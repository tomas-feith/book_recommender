"""The adaptive-hybrid recommender's scoring and card-selection contracts."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from app.recommender import POP_REF, Recommender
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
    focused = {s.book["id"] for s in rec.recommend(reactions, {}, n=2, mmr_lambda=1.0)}
    diverse = {s.book["id"] for s in rec.recommend(reactions, {}, n=2, mmr_lambda=0.2)}
    assert "b1" not in focused  # pure relevance keeps both near-identical A books
    assert "b1" in diverse  # diversity swaps the redundant A book for cluster B
