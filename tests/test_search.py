"""Fuzzy title resolution for onboarding."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from app.search import TitleIndex
from app.store import Catalog


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


def test_search_by_author(tiny_catalog):
    hits = TitleIndex(tiny_catalog).search("Jane Austen", k=2)
    assert all(h.author.startswith("Austen") for h in hits)  # both top hits are Austen books


def test_popularity_breaks_near_ties(tiny_catalog):
    # "Dune" and "Dune Messiah" both match; the far more popular "Dune" (b0) wins.
    assert TitleIndex(tiny_catalog).best("Dune").book_id == "b0"


def _alias_catalog() -> Catalog:
    books = [
        {
            "id": "x",
            "title": "The Three-Body Problem",
            "author": "Liu Cixin",
            "orig_title": "三体",
            "subjects": [],
            "language": "en",
            "year": 2008,
            "description": "",
        },
        {
            "id": "y",
            "title": "Neuromancer",
            "author": "William Gibson",
            "subjects": [],
            "language": "en",
            "year": 1984,
            "description": "",
        },
    ]
    return Catalog(
        books,
        np.zeros((2, 4), dtype=np.float32),
        sparse.csr_matrix((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        {b["id"]: i for i, b in enumerate(books)},
    )


def test_original_language_title_is_searchable():
    idx = TitleIndex(_alias_catalog())
    assert idx.best("three body problem").book_id == "x"  # English name
    assert idx.best("三体").book_id == "x"  # original-language alias
