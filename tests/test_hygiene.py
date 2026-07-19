"""Ingest-time hygiene: dedup + language guess."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from hygiene import dedup_key, dedup_records, guess_language, norm_title


def test_norm_title_strips_series_and_punctuation():
    assert norm_title("Dune (Dune, #1)") == "dune"
    assert norm_title("The Lord of the Rings!") == "the lord of the rings"
    assert norm_title("Salem's Lot") == "salem s lot"


def test_dedup_key_uses_title_and_first_author():
    a = {"title": "Coraline", "author": "Neil Gaiman"}
    b = {"title": "Coraline (Deluxe)", "author": "Gaiman, Neil, illus. Somebody"}
    # same title, but different first-credited author -> different keys
    assert dedup_key(a) == ("coraline", "neil gaiman")
    assert dedup_key(b) == ("coraline", "gaiman")


def test_dedup_keeps_canonical_edition():
    recs = [
        {
            "id": "1",
            "title": "The Firm",
            "author": "John Grisham",
            "description": "",
            "subjects": [],
        },
        {
            "id": "2",
            "title": "The Firm",
            "author": "John Grisham",
            "description": "A long blurb about a law firm.",
            "subjects": ["thriller"],
            "image": "x",
        },
        {"id": "3", "title": "Dune", "author": "Frank Herbert", "description": "d", "subjects": []},
    ]
    out = dedup_records(recs)
    assert len(out) == 2  # the two Firms collapse
    firm = next(b for b in out if b["title"] == "The Firm")
    assert firm["id"] == "2"  # the fuller record wins


def test_dedup_keeps_untitled_records():
    recs = [{"id": "a", "title": "", "author": ""}, {"id": "b", "title": "", "author": ""}]
    assert len(dedup_records(recs)) == 2  # empty titles can't be deduped


def test_dedup_is_order_stable():
    recs = [
        {"id": "1", "title": "B", "author": "x"},
        {"id": "2", "title": "A", "author": "y"},
        {"id": "3", "title": "B", "author": "x"},
    ]
    assert [b["title"] for b in dedup_records(recs)] == ["B", "A"]


def test_guess_language_by_script():
    assert guess_language("The Great Gatsby") == "en"
    assert guess_language("三体") == "zh"
    assert guess_language("ノルウェイの森") == "ja"  # kana forces Japanese
    assert guess_language("مئة عام من العزلة") == "ar"
    assert guess_language("Преступление и наказание") == "ru"
    assert guess_language("", default="en") == "en"
