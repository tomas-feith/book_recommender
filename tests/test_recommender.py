"""The adaptive-hybrid recommender's scoring and card-selection contracts."""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from app.recommender import POP_REF, Recommender, genre_distribution, kl_calibration
from app.store import Catalog


def test_cf_weight_monotone_in_popularity_and_bounded(tiny_catalog):
    rec = Recommender(tiny_catalog)
    w = rec.cf_weight
    assert np.all((w >= 0) & (w <= 1))
    # pop = [3000, 40, 500, 800, 5, 0] -> most-rated book trusts CF most, pop=0 least.
    assert w[0] == max(w)
    assert w[5] == 0.0  # a book with zero ratings is pure content


def test_recommend_excludes_reacted_and_caps_per_author(tiny_catalog):
    rec = Recommender(tiny_catalog)
    reactions = {"b0": "like"}  # liked Dune
    # n=3 <= number of distinct authors, so the per-author cap holds strictly
    # (with n larger than the author count, MMR relaxes the cap to still fill n).
    picks = rec.recommend(reactions, filters={}, n=3, per_author=1)
    ids = [s.book["id"] for s in picks]
    assert "b0" not in ids  # reacted book is not recommended back
    authors = [s.book["author"] for s in picks]
    assert authors.count("Herbert, Frank") <= 1
    assert authors.count("Austen, Jane") <= 1


def test_cold_user_falls_back_to_popularity(tiny_catalog):
    rec = Recommender(tiny_catalog)
    picks = rec.recommend({}, filters={}, n=3)
    # with no taste signal, the top pick should be the most popular book (b0, pop=3000).
    assert picks[0].book["id"] == "b0"


def test_filters_apply_around_scoring(tiny_catalog):
    rec = Recommender(tiny_catalog)
    picks = rec.recommend({"b0": "like"}, filters={"languages": ["fr"]}, n=5)
    assert {s.book["id"] for s in picks} == {"b5"}  # only the French book survives


def test_surprise_needs_positive_signal(tiny_catalog):
    rec = Recommender(tiny_catalog)
    assert rec.surprise({}, filters={}) == []  # cold user: no surprises
    out = rec.surprise({"b0": "like", "b2": "like"}, filters={})
    assert all(0.0 <= s.novelty <= 2.0 for s in out)


def test_scored_reports_cf_weight(tiny_catalog):
    rec = Recommender(tiny_catalog)
    picks = rec.recommend({"b0": "like"}, filters={}, n=3)
    assert all(0.0 <= s.cf_weight <= 1.0 for s in picks)


def test_pop_ref_constant_is_tuned_for_ease():
    # guards the retuned blend: EASE-R warranted a lower reference than the old KNN.
    assert POP_REF == 500.0


def test_similar_excludes_self_and_ranks_cf_neighbor_first(tiny_catalog):
    rec = Recommender(tiny_catalog)
    sims = rec.similar("b0", n=3)  # b0 is CF-warm (pop 3000) -> CF drives 'similar'
    ids = [s.book["id"] for s in sims]
    assert "b0" not in ids
    assert ids[0] == "b1"  # b1 is b0's strongest CF neighbor in the fixture
    assert all(s.explanation.startswith("Similar to") for s in sims)


def test_similar_unknown_book_is_empty(tiny_catalog):
    assert Recommender(tiny_catalog).similar("does-not-exist") == []


def test_recommendations_carry_explanations(tiny_catalog):
    picks = Recommender(tiny_catalog).recommend({"b0": "like"}, {}, n=3)
    assert all(s.explanation for s in picks)
    assert any("Dune" in s.explanation for s in picks)  # cites the liked title


def test_cold_user_gets_neutral_explanation(tiny_catalog):
    picks = Recommender(tiny_catalog).recommend({}, {}, n=2)
    assert picks[0].explanation == "A popular pick to get you started"


def _clustered_catalog() -> Catalog:
    """Two near-identical cluster-A books + one cluster-B book (distinct authors)."""
    books = [
        {
            "id": bid,
            "title": bid,
            "author": auth,
            "subjects": [sub],
            "language": "en",
            "year": 2000,
            "description": "",
        }
        for bid, auth, sub in [
            ("a1", "W1", "x"),
            ("a2", "W2", "x"),
            ("b1", "W3", "y"),
            ("seed", "W4", "x"),
        ]
    ]
    emb = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype=np.float32)  # a1,a2,seed=A; b1=B
    sim = sparse.csr_matrix((4, 4), dtype=np.float32)
    pop = np.zeros(4, dtype=np.float32)  # pop=0 -> pure content, so MMR distance bites
    return Catalog(books, emb, sim, pop, {b["id"]: i for i, b in enumerate(books)})


def test_mmr_lambda_trades_relevance_for_diversity():
    rec = Recommender(_clustered_catalog())
    reactions = {"seed": "like"}  # taste ~ cluster A; a1 & a2 both maximally relevant
    focused = {
        s.book["id"] for s in rec.recommend(reactions, {}, n=2, mmr_lambda=1.0, cal_lambda=0)
    }
    diverse = {
        s.book["id"] for s in rec.recommend(reactions, {}, n=2, mmr_lambda=0.2, cal_lambda=0)
    }
    assert "b1" not in focused  # pure relevance keeps both near-identical A books
    assert "b1" in diverse  # diversity swaps the redundant A book for cluster B


# ---- calibration primitives -------------------------------------------------


def test_genre_distribution_splits_mass_and_normalizes():
    d = genre_distribution([["a", "b"], ["a"]])  # book1: a,b half each; book2: a
    assert math.isclose(d["a"], 0.75) and math.isclose(d["b"], 0.25)


def test_genre_distribution_weights_and_empty():
    d = genre_distribution([["a"], ["b"]], weights=[1.0, 3.0])
    assert math.isclose(d["a"], 0.25) and math.isclose(d["b"], 0.75)
    assert genre_distribution([[], []]) == {}


def test_kl_calibration_zero_when_matched_positive_when_not():
    assert kl_calibration({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    off = kl_calibration({"a": 0.5, "b": 0.5}, {"a": 1.0})
    assert off > 0.0


def _two_taste_catalog() -> Catalog:
    """User likes genre a AND b; candidates: 3 highly-relevant 'a', 2 'b'."""
    rows = [
        ("as", "W0", "a", [1.0, 0.0]),  # liked seed, genre a
        ("bs", "W1", "b", [0.0, 1.0]),  # liked seed, genre b
        ("a1", "W2", "a", [0.8, 0.6]),  # a candidates: closest to the taste centroid
        ("a2", "W3", "a", [0.8, 0.6]),
        ("a3", "W4", "a", [0.8, 0.6]),
        ("b1", "W5", "b", [0.0, 1.0]),  # b candidates: relevant but less so
        ("b2", "W6", "b", [0.0, 1.0]),
    ]
    books = [
        {
            "id": bid,
            "title": bid,
            "author": auth,
            "subjects": [g],
            "language": "en",
            "year": 2000,
            "description": "",
        }
        for bid, auth, g, _ in rows
    ]
    emb = np.array([v for *_, v in rows], dtype=np.float32)
    sim = sparse.csr_matrix((len(rows), len(rows)), dtype=np.float32)
    pop = np.zeros(len(rows), dtype=np.float32)
    return Catalog(books, emb, sim, pop, {b["id"]: i for i, b in enumerate(books)})


def _n_genre_b(scored):
    return sum(1 for s in scored if s.book["subjects"] == ["b"])


def test_calibration_covers_a_minority_taste():
    rec = Recommender(_two_taste_catalog())
    reactions = {"as": "like", "bs": "like"}  # taste is 50% a, 50% b
    # Pure relevance floods the list with the higher-scoring 'a' cluster...
    uncalibrated = rec.recommend(reactions, {}, n=3, mmr_lambda=1.0, cal_lambda=0.0)
    assert _n_genre_b(uncalibrated) == 0
    # ...strong calibration overrides relevance to cover the user's 'b' taste.
    calibrated = rec.recommend(reactions, {}, n=3, mmr_lambda=1.0, cal_lambda=5.0)
    assert _n_genre_b(calibrated) >= 1
