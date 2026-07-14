"""The adaptive-hybrid recommender's scoring and card-selection contracts."""

from __future__ import annotations

import numpy as np

from app.recommender import POP_REF, Recommender


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
    picks = rec.recommend(reactions, filters={}, n=5, per_author=1)
    ids = [s.book["id"] for s in picks]
    assert "b0" not in ids  # reacted book is not recommended back
    # per_author=1 -> at most one Austen and one Herbert in the list
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
