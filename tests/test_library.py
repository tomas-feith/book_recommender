"""Library import: file parsing (pure) and catalog matching (via the service)."""

from __future__ import annotations

import io

from openpyxl import Workbook

from app.library import LibraryEntry, parse_library
from app.recommender import Recommender
from app.search import TitleIndex
from app.service import BookRecommenderService
from app.store import SwipeStore

# ---- parsing ---------------------------------------------------------------


def test_csv_with_header():
    raw = b"Title,Author\nDune,Frank Herbert\nEmma,Jane Austen\n"
    entries = parse_library("list.csv", raw)
    assert entries == [LibraryEntry("Dune", "Frank Herbert"), LibraryEntry("Emma", "Jane Austen")]


def test_csv_without_header_first_column_is_title():
    raw = b"Dune,Frank Herbert\nNeuromancer,William Gibson\n"
    entries = parse_library("list.csv", raw)
    assert entries[0] == LibraryEntry("Dune", "Frank Herbert")
    assert len(entries) == 2


def test_header_order_independent():
    raw = b"author,title\nFrank Herbert,Dune\n"
    assert parse_library("l.csv", raw) == [LibraryEntry("Dune", "Frank Herbert")]


def test_txt_lines_and_by_separator():
    raw = b"Dune by Frank Herbert\nNeuromancer\n\n  Emma by Jane Austen  \n"
    entries = parse_library("list.txt", raw)
    assert entries == [
        LibraryEntry("Dune", "Frank Herbert"),
        LibraryEntry("Neuromancer", ""),
        LibraryEntry("Emma", "Jane Austen"),
    ]


def test_tsv_tab_delimited():
    raw = b"Title\tAuthor\nDune\tFrank Herbert\n"
    assert parse_library("l.tsv", raw) == [LibraryEntry("Dune", "Frank Herbert")]


def test_xlsx_roundtrip():
    wb = Workbook()
    ws = wb.active
    ws.append(["Title", "Author"])
    ws.append(["Dune", "Frank Herbert"])
    ws.append(["Emma", None])
    buf = io.BytesIO()
    wb.save(buf)
    entries = parse_library("shelf.xlsx", buf.getvalue())
    assert entries == [LibraryEntry("Dune", "Frank Herbert"), LibraryEntry("Emma", "")]


def test_bom_is_stripped():
    raw = "Title,Author\nDune,Frank Herbert\n".encode("utf-8-sig")
    assert parse_library("l.csv", raw)[0].title == "Dune"


def test_duplicates_removed():
    raw = b"Dune,Frank Herbert\ndune,frank herbert\nEmma,Jane Austen\n"
    entries = parse_library("l.csv", raw)
    assert len(entries) == 2


# ---- matching to the catalog (service) -------------------------------------


def _service(catalog, tmp_path) -> BookRecommenderService:
    """A service wired to a synthetic catalog without touching real data files."""
    svc = BookRecommenderService.__new__(BookRecommenderService)
    svc.catalog = catalog
    svc.recommender = Recommender(catalog)
    svc.titles = TitleIndex(catalog)
    svc.store = SwipeStore(db_path=tmp_path / "app.db")
    svc.data_dir = tmp_path
    svc._encoder_loaded = True  # pretend the encoder is unavailable (no torch in tests)
    svc._encoder_cache = None
    return svc


def test_import_matches_and_reports_unmatched(tiny_catalog, tmp_path):
    svc = _service(tiny_catalog, tmp_path)
    uid = svc.store.create_user()
    entries = [
        LibraryEntry("Dune", "Frank Herbert"),
        LibraryEntry("Neuromancer", "William Gibson"),
        LibraryEntry("A Book We Do Not Have", "Nobody"),
    ]
    result = svc.import_library(uid, entries)

    matched_ids = {m.match.book_id for m in result.matched}
    assert matched_ids == {"b0", "b2"}
    assert [e.title for e in result.unmatched] == ["A Book We Do Not Have"]
    # matched books were recorded as likes (an import seeds taste)
    assert svc.store.reactions(uid) == {"b0": "like", "b2": "like"}
    svc.store.close()


def test_author_disambiguates_and_records_reaction(tiny_catalog, tmp_path):
    svc = _service(tiny_catalog, tmp_path)
    uid = svc.store.create_user()
    # "Emma" by Austen should resolve to b4; record as interested instead of like
    result = svc.import_library(uid, [LibraryEntry("Emma", "Jane Austen")], reaction="interested")
    assert result.n_matched == 1
    assert result.matched[0].match.book_id == "b4"
    assert svc.store.reactions(uid) == {"b4": "interested"}
    svc.store.close()


def test_similar_books_via_service(tiny_catalog, tmp_path):
    svc = _service(tiny_catalog, tmp_path)
    sims = svc.similar_books("b0", n=3)
    assert sims and "b0" not in {s.book["id"] for s in sims}
    svc.store.close()


def test_semantic_search_falls_back_without_encoder(tiny_catalog, tmp_path):
    svc = _service(tiny_catalog, tmp_path)  # encoder forced unavailable
    assert svc.semantic_search("books about the desert") == []
    assert svc.semantic_search("") == []
    svc.store.close()


def test_add_external_book_is_idempotent_for_known_ids(tiny_catalog, tmp_path):
    svc = _service(tiny_catalog, tmp_path)
    n0 = len(svc.catalog.books)
    # b0 is already in the catalog -> returns its id without re-adding or fetching
    assert svc.add_external_book({"id": "b0", "title": "Dune"}) == "b0"
    assert len(svc.catalog.books) == n0
    svc.store.close()
