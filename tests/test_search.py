"""Fuzzy title resolution for onboarding."""

from __future__ import annotations

from app.search import TitleIndex


def test_exact_title_is_best_match(tiny_catalog):
    idx = TitleIndex(tiny_catalog)
    match = idx.best("Neuromancer")
    assert match is not None
    assert match.book_id == "b2"


def test_fuzzy_and_case_insensitive(tiny_catalog):
    idx = TitleIndex(tiny_catalog)
    match = idx.best("pride and prejudice")
    assert match is not None
    assert match.book_id == "b3"


def test_search_returns_ranked_matches(tiny_catalog):
    idx = TitleIndex(tiny_catalog)
    hits = idx.search("Dune", k=3)
    assert hits[0].book_id == "b0"  # exact wins over "Dune Messiah"
    assert len(hits) <= 3
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_gibberish_below_threshold_returns_none(tiny_catalog):
    idx = TitleIndex(tiny_catalog)
    assert idx.best("zzzzqqqq xxxx", threshold=0.5) is None


def test_empty_query_returns_no_matches(tiny_catalog):
    assert TitleIndex(tiny_catalog).search("   ") == []
