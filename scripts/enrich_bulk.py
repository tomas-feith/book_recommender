"""Backfill missing descriptions from **bulk dataset dumps** -- no API, no quota.

``enrich_google_books.py`` fills one book per HTTP request against a rate-limited,
frequently-503ing endpoint; draining the ~10k description-less catalog that way
takes months. But the same descriptions already exist, complete, in the bulk
datasets this project ingests -- so match against those offline instead:

* goodbooks entries (numeric ids) -> the **UCSD Goodreads** ``goodreads_books.json.gz``.
  We don't store the goodreads id, but the cached ``gb_books.csv`` maps our catalog
  id -> ``goodreads_book_id`` (== the UCSD ``book_id``), and UCSD carries a
  ``description`` field.
* Open Library works (``ol:OL...W`` ids) -> the **OL works dump**, matched on the
  work key directly.

Each dump is streamed once (never loaded whole -- they are multi-GB), collecting
only the descriptions for books we still need, then the changed rows are re-embedded
in place so ``real_embeddings.npz`` stays consistent (shared with the API enricher).

Download the dumps once, then run whichever you have:
    # UCSD: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
    # OL:   https://openlibrary.org/developers/dumps  (ol_dump_works_latest.txt.gz)
    uv run --no-sync python scripts/enrich_bulk.py \
        --goodreads goodreads_books.json.gz \
        --ol-works  ol_dump_works_latest.txt.gz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # sibling-script imports

from enrich_google_books import _reembed  # noqa: E402  (reuse the row-wise re-embed)
from ingest_goodreads_ucsd import stream_jsonl_gz  # noqa: E402
from ingest_openlibrary_dump import _text, stream_dump  # noqa: E402

DATA = ROOT / "data"
GB_CSV = ROOT / ".cache" / "gb_books.csv"
DESC_CAP = 800  # match the API enricher's cap so embeddings see the same text budget


def _missing(books: list[dict]) -> list[dict]:
    return [b for b in books if not (b.get("description") or "").strip()]


def goodreads_targets(missing: list[dict], gb_csv: Path | None = None) -> dict[str, dict]:
    """goodreads_book_id -> catalog book, for numeric-id books that need a description.

    The join hop is catalog id -> gb_books.csv -> goodreads_book_id (the UCSD key).
    """
    gb_csv = gb_csv or GB_CSV  # resolved at call time so callers/tests can override GB_CSV
    if not gb_csv.exists():
        return {}
    by_catalog_id = {b["id"]: b for b in missing if b["id"].isdigit()}
    if not by_catalog_id:
        return {}
    targets: dict[str, dict] = {}
    with open(gb_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            book = by_catalog_id.get(row["book_id"])
            gr_id = (row.get("goodreads_book_id") or "").strip()
            if book and gr_id:
                targets[gr_id] = book
    return targets


def ol_targets(missing: list[dict]) -> dict[str, dict]:
    """bare OLID (e.g. 'OL123W') -> catalog book, for ol: works needing a description."""
    return {b["id"].split(":", 1)[1]: b for b in missing if b["id"].startswith("ol:")}


def _apply(book: dict, desc: str, subjects: list[str], image: str) -> None:
    """Fill the blanks on a book in place (description, and subjects/image if empty)."""
    book["description"] = desc.strip()[:DESC_CAP]
    if not book.get("subjects") and subjects:
        book["subjects"] = [s.lower() for s in subjects[:5] if isinstance(s, str)]
    if not (book.get("image") or "").strip() and image:
        book["image"] = image


def fill_from_goodreads(targets: dict[str, dict], books_gz: Path) -> dict[str, dict]:
    """Stream the UCSD books dump once, filling ``targets`` that have a description."""
    changed: dict[str, dict] = {}
    for raw in stream_jsonl_gz(books_gz):
        book = targets.get(str(raw.get("book_id")))
        if book is None:
            continue
        desc = (raw.get("description") or "").strip()
        if desc:
            shelves = [
                s.get("name", "")
                for s in sorted(
                    raw.get("popular_shelves", []),
                    key=lambda s: -int(s.get("count", 0) or 0),
                )
            ]
            _apply(book, desc, shelves, raw.get("image_url", ""))
            changed[book["id"]] = book
            if len(changed) == len(targets):  # every needed book found -> stop streaming
                break
    return changed


def fill_from_ol_works(targets: dict[str, dict], works_gz: Path) -> dict[str, dict]:
    """Stream the OL works dump once, filling ``targets`` that have a description."""
    changed: dict[str, dict] = {}
    for work in stream_dump(works_gz):
        key = work.get("key", "")  # "/works/OL...W"
        book = targets.get(key.rsplit("/", 1)[-1]) if key else None
        if book is None:
            continue
        desc = _text(work.get("description"))
        if desc:
            subjects = [s for s in work.get("subjects", []) if isinstance(s, str)]
            covers = [c for c in work.get("covers", []) if isinstance(c, int) and c > 0]
            image = f"https://covers.openlibrary.org/b/id/{covers[0]}-M.jpg" if covers else ""
            _apply(book, desc, subjects, image)
            changed[book["id"]] = book
            if len(changed) == len(targets):
                break
    return changed


def enrich(
    data_dir: Path = DATA,
    goodreads: Path | None = None,
    ol_works: Path | None = None,
    re_embed: bool = True,
) -> dict[str, dict]:
    books = json.loads((data_dir / "real_books.json").read_text(encoding="utf-8"))
    missing = _missing(books)
    gr_t = goodreads_targets(missing)
    ol_t = ol_targets(missing)
    print(
        f"{len(missing)} books missing a description "
        f"({len(gr_t)} matchable to goodreads, {len(ol_t)} to OL works)."
    )

    changed: dict[str, dict] = {}
    if goodreads:
        got = fill_from_goodreads(gr_t, goodreads)
        print(f"  goodreads dump: filled {len(got)}")
        changed.update(got)
    if ol_works:
        got = fill_from_ol_works(ol_t, ol_works)
        print(f"  OL works dump:  filled {len(got)}")
        changed.update(got)

    if not changed:
        print("Nothing filled (no dumps given, or no matches with a description).")
        return {}

    (data_dir / "real_books.json").write_text(
        json.dumps(books, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Filled {len(changed)} descriptions from bulk dumps.")
    if re_embed:
        _reembed(data_dir, changed)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill descriptions from bulk dataset dumps.")
    ap.add_argument("--goodreads", type=Path, help="UCSD goodreads_books.json.gz")
    ap.add_argument("--ol-works", type=Path, help="OL ol_dump_works_latest.txt.gz")
    ap.add_argument("--no-embed", action="store_true", help="Skip re-embedding.")
    args = ap.parse_args()
    if not args.goodreads and not args.ol_works:
        ap.error("give at least one of --goodreads / --ol-works")
    enrich(goodreads=args.goodreads, ol_works=args.ol_works, re_embed=not args.no_embed)


if __name__ == "__main__":
    main()
