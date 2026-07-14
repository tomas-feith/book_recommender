"""book_to_text: what text represents a book for embedding."""

from __future__ import annotations

import pytest

from eval.data import book_to_text

BOOK = {
    "title": "Dune",
    "author": "Frank Herbert",
    "description": "Desert planet epic.",
    "subjects": ["science fiction", "epic"],
}


def test_full_includes_subjects():
    text = book_to_text(BOOK, mode="full")
    assert "Dune" in text and "Frank Herbert" in text
    assert "Desert planet epic." in text
    assert "science fiction" in text


def test_no_subjects_drops_genre_words():
    text = book_to_text(BOOK, mode="no-subjects")
    assert "Dune" in text
    assert "science fiction" not in text


def test_missing_optional_fields_dont_crash():
    text = book_to_text({"title": "X", "author": "Y"}, mode="full")
    assert "X" in text and "Y" in text


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown text mode"):
        book_to_text(BOOK, mode="sideways")
