"""Tests for the bulk-dump description enricher (scripts/enrich_bulk.py).

No multi-GB downloads: tiny gzipped fixtures stand in for the real dumps, so
these pin the *join keys and fill logic* -- which is the whole point of the script.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import enrich_bulk  # noqa: E402


def _write_gz(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _catalog(tmp_path: Path, books: list[dict]) -> Path:
    (tmp_path / "real_books.json").write_text(json.dumps(books), encoding="utf-8")
    return tmp_path


# ---- goodreads join ----------------------------------------------------------


def test_goodreads_targets_joins_catalog_id_via_gb_csv(tmp_path):
    gb = tmp_path / "gb.csv"
    gb.write_text("book_id,goodreads_book_id\n106,9418327\n999,55555\n", encoding="utf-8")
    # 106 is missing (needs a fill); 999 has a description already so must be skipped.
    missing = [{"id": "106", "title": "Bossypants", "description": ""}]
    targets = enrich_bulk.goodreads_targets(missing, gb_csv=gb)
    assert set(targets) == {"9418327"}  # keyed by goodreads_book_id, not catalog id
    assert targets["9418327"]["id"] == "106"


def test_fill_from_goodreads_fills_only_matches_with_a_description(tmp_path):
    book = {"id": "106", "title": "Bossypants", "description": "", "subjects": [], "image": ""}
    targets = {"9418327": book}
    dump = tmp_path / "gr.json.gz"
    _write_gz(
        dump,
        [
            json.dumps(
                {
                    "book_id": "9418327",
                    "description": "A funny memoir.",
                    "image_url": "http://img/x.jpg",
                    "popular_shelves": [{"name": "Humor", "count": "50"}],
                }
            ),
            json.dumps({"book_id": "0000", "description": "unrelated"}),
        ],
    )
    changed = enrich_bulk.fill_from_goodreads(targets, dump)
    assert set(changed) == {"106"}
    assert book["description"] == "A funny memoir."
    assert book["subjects"] == ["humor"]  # filled from popular_shelves, lowercased
    assert book["image"] == "http://img/x.jpg"


def test_fill_from_goodreads_skips_empty_description(tmp_path):
    book = {"id": "106", "title": "X", "description": ""}
    dump = tmp_path / "gr.json.gz"
    _write_gz(dump, [json.dumps({"book_id": "9418327", "description": ""})])
    assert enrich_bulk.fill_from_goodreads({"9418327": book}, dump) == {}
    assert book["description"] == ""


# ---- OL works join -----------------------------------------------------------


def test_ol_targets_strips_prefix(tmp_path):
    missing = [
        {"id": "ol:OL123W", "description": ""},
        {"id": "106", "description": ""},  # not an OL id
    ]
    assert set(enrich_bulk.ol_targets(missing)) == {"OL123W"}


def test_fill_from_ol_works_matches_work_key(tmp_path):
    book = {"id": "ol:OL123W", "title": "T", "description": "", "subjects": [], "image": ""}
    dump = tmp_path / "works.txt.gz"
    # OL dump rows are TSV with the JSON record in the 5th column.
    row = "\t".join(
        [
            "type",
            "/works/OL123W",
            "1",
            "date",
            json.dumps(
                {
                    "key": "/works/OL123W",
                    "description": {"type": "/type/text", "value": "A classic."},
                    "subjects": ["Fiction"],
                    "covers": [42],
                }
            ),
        ]
    )
    _write_gz(dump, [row, "malformed\trow"])
    changed = enrich_bulk.fill_from_ol_works({"OL123W": book}, dump)
    assert set(changed) == {"ol:OL123W"}
    assert book["description"] == "A classic."  # unwrapped from the {type,value} dict
    assert book["subjects"] == ["fiction"]
    assert "42-M.jpg" in book["image"]


# ---- title-match fallback (fuzzy join; precision guards are the point) --------


def test_norm_title_strips_series_and_punctuation():
    assert enrich_bulk._norm_title("The Color of Magic (Discworld, #1)") == "the color of magic"
    assert enrich_bulk._norm_title("Guards! Guards!") == "guards guards"


def _gr(book_id, title, desc, year, rc, work_id):
    return json.dumps(
        {
            "book_id": book_id,
            "title": title,
            "title_without_series": title,
            "description": desc,
            "publication_year": str(year),
            "ratings_count": str(rc),
            "work_id": work_id,
            "popular_shelves": [],
        }
    )


def test_title_match_fills_by_title_and_year(tmp_path):
    book = {
        "id": "ol:OL9W",
        "title": "Ancillary Justice",
        "year": 2013,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(dump, [_gr("1", "Ancillary Justice", "Radch space opera.", 2013, 5000, "w1")])
    targets = enrich_bulk.title_targets([book])
    changed = enrich_bulk.fill_from_goodreads_titles(targets, dump)
    assert set(changed) == {"ol:OL9W"}
    assert book["description"] == "Radch space opera."


def test_title_match_rejects_year_mismatch(tmp_path):
    # Same title, but the only candidate is 30 years off -> almost certainly a
    # different book, so reject rather than mis-fill.
    book = {
        "id": "ol:OL9W",
        "title": "Foundation",
        "year": 2021,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(dump, [_gr("1", "Foundation", "Asimov's classic.", 1951, 9000, "w1")])
    targets = enrich_bulk.title_targets([book])
    assert enrich_bulk.fill_from_goodreads_titles(targets, dump) == {}
    assert book["description"] == ""


def test_title_match_no_year_requires_single_work(tmp_path):
    # Our book has no year; two different works share the title -> ambiguous -> skip.
    book = {
        "id": "ol:OL9W",
        "title": "Passage",
        "year": None,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(
        dump,
        [
            _gr("1", "Passage", "Connie Willis novel.", 2001, 8000, "wA"),
            _gr("2", "Passage", "A different Passage.", 2014, 100, "wB"),
        ],
    )
    targets = enrich_bulk.title_targets([book])
    assert enrich_bulk.fill_from_goodreads_titles(targets, dump) == {}


def test_title_match_rejects_generic_short_title_with_few_ratings(tmp_path):
    # "Alice" + right year but an obscure 11-rating match -> too risky, skip.
    book = {
        "id": "ol:OL9W",
        "title": "Alice",
        "year": 2018,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(dump, [_gr("1", "Alice", "Some other Alice.", 2018, 11, "w1")])
    targets = enrich_bulk.title_targets([book])
    assert enrich_bulk.fill_from_goodreads_titles(targets, dump) == {}
    assert book["description"] == ""


def test_title_match_accepts_short_title_when_well_rated(tmp_path):
    # Same short title, but a clearly-canonical 900k-rating edition -> safe to fill.
    book = {
        "id": "ol:OL9W",
        "title": "Damnation",
        "year": 2015,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(dump, [_gr("1", "Damnation", "The real one.", 2015, 997, "w1")])
    targets = enrich_bulk.title_targets([book])
    changed = enrich_bulk.fill_from_goodreads_titles(targets, dump)
    assert set(changed) == {"ol:OL9W"}


def test_title_match_picks_most_rated_edition(tmp_path):
    book = {
        "id": "ol:OL9W",
        "title": "Dune",
        "year": 1965,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(
        dump,
        [
            _gr("1", "Dune", "Reissue blurb.", 1965, 50, "w1"),
            _gr("2", "Dune", "Canonical blurb.", 1965, 900000, "w1"),
        ],
    )
    targets = enrich_bulk.title_targets([book])
    enrich_bulk.fill_from_goodreads_titles(targets, dump)
    assert book["description"] == "Canonical blurb."


def test_title_match_drops_two_books_colliding_on_one_work(tmp_path):
    # Two of OUR books normalize to the same title and both resolve to the same
    # goodreads work -> we can't tell them apart -> neither is filled.
    b1 = {
        "id": "ol:A",
        "title": "Twins",
        "year": 2010,
        "description": "",
        "subjects": [],
        "image": "",
    }
    b2 = {
        "id": "ol:B",
        "title": "Twins",
        "year": 2010,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(dump, [_gr("1", "Twins", "Only one blurb.", 2010, 500, "w1")])
    targets = enrich_bulk.title_targets([b1, b2])
    assert enrich_bulk.fill_from_goodreads_titles(targets, dump) == {}
    assert b1["description"] == "" and b2["description"] == ""


def test_title_match_dry_run_reports_without_mutating(tmp_path):
    book = {
        "id": "ol:OL9W",
        "title": "Spin",
        "year": 2005,
        "description": "",
        "subjects": [],
        "image": "",
    }
    dump = tmp_path / "gr.json.gz"
    _write_gz(dump, [_gr("1", "Spin", "Time dilation.", 2005, 4000, "w1")])
    targets = enrich_bulk.title_targets([book])
    changed = enrich_bulk.fill_from_goodreads_titles(targets, dump, dry_run=True)
    assert set(changed) == {"ol:OL9W"}  # reported as would-fill
    assert book["description"] == ""  # but NOT mutated


# ---- end to end (both dumps, then the shared re-embed is skipped) -------------


def test_enrich_end_to_end_writes_back_without_embedding(tmp_path):
    books = [
        {"id": "106", "title": "Bossypants", "description": "", "subjects": [], "image": ""},
        {"id": "ol:OL9W", "title": "Iliad", "description": "", "subjects": [], "image": ""},
        {"id": "200", "title": "Has one", "description": "already here"},
    ]
    _catalog(tmp_path, books)
    gb = tmp_path / "gb.csv"
    gb.write_text("book_id,goodreads_book_id\n106,777\n", encoding="utf-8")
    enrich_bulk.GB_CSV = gb  # point the join at the fixture csv

    gr = tmp_path / "gr.json.gz"
    _write_gz(gr, [json.dumps({"book_id": "777", "description": "Memoir."})])
    ol = tmp_path / "works.txt.gz"
    _write_gz(
        ol,
        [
            "\t".join(
                [
                    "t",
                    "/works/OL9W",
                    "1",
                    "d",
                    json.dumps({"key": "/works/OL9W", "description": "Epic poem."}),
                ]
            )
        ],
    )

    changed = enrich_bulk.enrich(data_dir=tmp_path, goodreads=gr, ol_works=ol, re_embed=False)
    assert set(changed) == {"106", "ol:OL9W"}
    on_disk = json.loads((tmp_path / "real_books.json").read_text(encoding="utf-8"))
    by_id = {b["id"]: b for b in on_disk}
    assert by_id["106"]["description"] == "Memoir."
    assert by_id["ol:OL9W"]["description"] == "Epic poem."
    assert by_id["200"]["description"] == "already here"  # untouched
