"""Representation programs: the book_to_text variants swept in eval.repr_sweep."""

from __future__ import annotations

from eval.data import book_to_text
from eval.representations import (
    REGISTRY,
    rep_full,
    rep_repeat_subjects,
    rep_strip_boilerplate,
    strip_boilerplate,
)

BOOK = {
    "title": "River God",
    "author": "Wilbur Smith",
    "description": "For Tanus, the young warrior, the gods have decreed war.",
    "subjects": ["fiction", "historical"],
}


def test_strip_removes_leading_series_bookkeeping():
    d = "Book two, a continuation of the Tom Gray series. After the explosion, Tom hides."
    out = strip_boilerplate(d)
    assert out.startswith("After the explosion")
    assert "series" not in out.lower()


def test_strip_removes_from_the_author_marketing():
    out = strip_boilerplate("From the best-selling creator of Percy Jackson. Old enemies awaken.")
    assert "Percy Jackson" not in out
    assert "Old enemies awaken." in out


def test_strip_removes_bestseller_and_award_tokens():
    out = strip_boilerplate("A New York Times bestselling tale of an award-winning chef.").lower()
    assert "bestselling" not in out
    assert "award-winning" not in out
    assert "chef" in out


def test_strip_leaves_clean_text_untouched():
    d = "While their father is away at war, four sisters grow up."
    assert strip_boilerplate(d) == d


def test_strip_is_idempotent():
    once = strip_boilerplate("Book one of the Foo series. The bestselling saga begins.")
    assert strip_boilerplate(once) == once


def test_strip_handles_empty():
    assert strip_boilerplate("") == ""


def test_rep_full_matches_production_book_to_text():
    assert rep_full(BOOK) == book_to_text(BOOK, mode="full")


def test_rep_repeat_subjects_doubles_the_themes_clause():
    assert rep_repeat_subjects(BOOK).count("Themes:") == 2


def test_rep_strip_boilerplate_composes_title_and_themes():
    book = dict(BOOK, description="The bestselling epic. A desert war unfolds.")
    text = rep_strip_boilerplate(book)
    assert text.startswith("River God by Wilbur Smith.")
    assert "bestselling" not in text.lower()
    assert "Themes: fiction, historical." in text


def test_registry_keys_are_stable_and_callable():
    assert set(REGISTRY) == {"full", "strip_boilerplate", "repeat_subjects", "strip_repeat"}
    for fn in REGISTRY.values():
        assert isinstance(fn(BOOK), str)
