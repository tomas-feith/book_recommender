"""On-demand Open Library fetch: doc -> record shaping (no network)."""

from __future__ import annotations

from app.external import enrich_description, to_record


def test_to_record_shapes_an_ol_doc():
    doc = {
        "key": "/works/OL123W",
        "title": "Dune",
        "author_name": ["Frank Herbert", "someone"],
        "first_publish_year": 1965,
        "subject": ["Science Fiction", "epic"],
        "language": ["eng"],
        "cover_i": 42,
    }
    rec = to_record(doc)
    assert rec is not None
    assert rec["id"] == "ol:OL123W"
    assert rec["title"] == "Dune"
    assert rec["author"] == "Frank Herbert, someone"
    assert rec["year"] == 1965
    assert "science fiction" in rec["subjects"]
    assert rec["language"] == "en"
    assert rec["image"].endswith("42-M.jpg")


def test_to_record_rejects_incomplete_docs():
    assert to_record({"key": "/works/OL1W"}) is None  # no title
    assert to_record({"key": "/works/OL1W", "title": "X"}) is None  # no author
    assert to_record({"title": "X", "author_name": ["A"]}) is None  # not a work key


def test_enrich_description_is_noop_when_present_or_not_ol():
    present = {"id": "ol:OL1W", "description": "already here"}
    assert enrich_description(present) is present
    non_ol = {"id": "123", "description": ""}
    assert enrich_description(non_ol) is non_ol
