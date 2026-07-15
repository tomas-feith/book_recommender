"""Tests for the Open Library subject normalizer (scripts/refresh_subjects.py).

Parsing only -- no network. This rewrites the subjects of every ``ol:`` book in
the catalog, so the shape of what it emits is pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from refresh_subjects import goodbooks_vocab, normalize_subjects  # noqa: E402

VOCAB = {
    "science-fiction",
    "sci-fi",
    "fiction",
    "fantasy",
    "romance",
    "young-adult",
    "classics",
    "historical-fiction",
}


def test_maps_library_headings_into_catalog_vocabulary():
    """'science fiction' (OL) and 'science-fiction' (goodbooks) are the same genre.

    They must end up as the same string, or genre calibration scores an OL book
    at zero against a user whose taste vector says 'science-fiction'.
    """
    assert "science-fiction" in normalize_subjects(["science fiction"], VOCAB)


def test_known_vocab_tags_come_first():
    """The cap is 5; vocabulary matches must not be crowded out by noise."""
    raw = ["obscure heading", "another one", "third", "fourth", "science fiction"]
    assert normalize_subjects(raw, VOCAB)[0] == "science-fiction"


def test_drops_machine_tags():
    raw = ["nyt:hardcover-fiction=2021-05-23", "science fiction"]
    assert normalize_subjects(raw, VOCAB) == ["science-fiction"]


def test_splits_bisac_headings():
    """OL carries BISAC strings: one heading is really several genres."""
    got = normalize_subjects(["fiction, fantasy, general"], VOCAB)
    assert "fiction" in got and "fantasy" in got
    assert "general" not in got  # a filler component, not a genre


def test_keeps_hyphenated_tags():
    """The catalog's own tags are hyphenated; an isalpha() filter dropped them."""
    got = normalize_subjects(["science-fiction", "sci-fi"], VOCAB)
    assert got == ["science-fiction", "sci-fi"]


def test_drops_non_ascii_and_punctuation():
    got = normalize_subjects(["fantasía", "dragons & mythical creatures", "fantasy"], VOCAB)
    assert got == ["fantasy"]


def test_dedupes():
    got = normalize_subjects(["fiction", "Fiction", "fiction, general"], VOCAB)
    assert got == ["fiction"]


def test_respects_cap():
    raw = [f"heading {i}" for i in range(20)]
    assert len(normalize_subjects(raw, VOCAB, cap=5)) == 5


def test_empty_input_gives_empty_list():
    assert normalize_subjects([], VOCAB) == []


def test_goodbooks_vocab_excludes_ol_books():
    books = [
        {"id": "123", "subjects": ["science-fiction"]},
        {"id": "ol:OL1W", "subjects": ["science fiction"]},
    ]
    # The vocabulary is the *target* dialect, so it must come only from the
    # goodbooks half -- seeding it with ol: tags would entrench the split.
    assert goodbooks_vocab(books) == {"science-fiction"}
