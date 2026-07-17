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
    #       (direct: https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/)
    # OL:   https://openlibrary.org/developers/dumps  (ol_dump_works_latest.txt.gz)
    uv run --no-sync python scripts/enrich_bulk.py \
        --goodreads goodreads_books.json.gz \
        --ol-works  ol_dump_works_latest.txt.gz

Add ``--title-match --works goodreads_book_works.json.gz`` to also match the no-key
books by title; the works dump (75MB) supplies the original publication year that
makes that match trustworthy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # sibling-script imports

from enrich_google_books import _reembed  # noqa: E402  (reuse the row-wise re-embed)
from ingest_goodreads_ucsd import _to_int, stream_jsonl_gz  # noqa: E402
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


# --- title-match fallback (for books with no key join, e.g. OL works) ---------------
#
# The ol: books have no goodreads_book_id and no ISBN, so they can't key-join to the
# goodreads dump -- but many are common titles that ARE in it. Matching on title is
# fuzzy, and a *wrong* description is worse than a blank one (it poisons the book's
# embedding), so this is deliberately precision-over-recall: it requires the matched
# editions to be a single work, rejects anything ambiguous, and never guesses.

_PARENS = re.compile(r"[(\[].*?[)\]]")  # "(Discworld, #5)", "[UK edition]"
_NONWORD = re.compile(r"[^a-z0-9]+")
# Our ``year`` is the WORK's first-publication year (what Open Library records). The books
# dump only carries the EDITION's ``publication_year`` -- a reprint of an 1895 novel is
# stamped 2009 -- so on its own the year can only rule out candidates that predate the work
# (an edition cannot print a book that doesn't exist yet); it cannot confirm identity.
# The *works* dump does carry ``original_publication_year``, which is the same quantity as
# ours and so compares directly. Pass ``--works`` to get that stronger test; without it we
# fall back to the edition floor. This tolerance covers data noise in either comparison.
YEAR_TOL = 1
# A distinctive multi-word title ("Nemesis Games") is safe on its own, but a short
# generic one ("Alice", "Self Defense") can land on the wrong book even with a year
# match -- so require such a title to resolve to a well-rated (i.e. canonical) edition,
# since a wrong obscure match has few ratings. Tunable precision/recall knob.
SHORT_TITLE_WORDS = 2
MIN_RATINGS_SHORT = 50


def _norm_title(title: str) -> str:
    """Aggressively normalize a title for matching: drop series/edition parens and
    punctuation, lowercase, collapse whitespace. Favors precision -- a subtitle
    mismatch just misses rather than mis-matches."""
    t = _PARENS.sub(" ", title.lower())
    return " ".join(_NONWORD.sub(" ", t).split())


def title_targets(missing: list[dict]) -> dict[str, list[dict]]:
    """normalized-title -> the still-missing books wanting it (a list: titles collide)."""
    out: dict[str, list[dict]] = {}
    for b in missing:
        key = _norm_title(b.get("title", ""))
        if key:
            out.setdefault(key, []).append(b)
    return out


def load_work_years(works_gz: Path, keep: set[str]) -> dict[str, int]:
    """work_id -> original_publication_year, for the work_ids in ``keep``.

    ``keep`` bounds this to the candidate works (a few thousand) rather than all ~2.3M.
    """
    out: dict[str, int] = {}
    for work in stream_jsonl_gz(works_gz):
        wid = str(work.get("work_id"))
        if wid in keep and (y := _to_int(work.get("original_publication_year"))):
            out[wid] = y
    return out


def _year_ok(rec: dict, yr: int, work_years: dict[str, int]) -> bool:
    """Is ``rec`` plausibly an edition of a work first published in ``yr``?"""
    if wy := work_years.get(str(rec.get("work_id"))):
        return abs(wy - yr) <= YEAR_TOL  # like-for-like: both are first-publication years
    # No work year (no --works, or the dump omits it): fall back to the edition floor.
    # This also rejects records the books dump leaves undated, which is why --works
    # recovers books whose only candidates carry no publication_year at all.
    return bool(py := _to_int(rec.get("publication_year"))) and py >= yr - YEAR_TOL


def _pick_record(book: dict, recs: list[dict], work_years: dict[str, int] | None = None) -> dict | None:
    """Choose the safest goodreads record for ``book``, or None if unsafe/ambiguous.

    Identity is settled by ``work_id``, not by the year: every edition of one work shares
    a work_id and describes the same book, so several editions are fine to choose among,
    but several *works* is a title collision we cannot resolve -- reject rather than guess.
    The year (see ``_year_ok``) prunes candidates that cannot be our book at all.
    Ties break to the most-rated edition (the canonical one).
    """
    yr = book.get("year")
    if yr:
        pool = [r for r in recs if _year_ok(r, yr, work_years or {})]
    else:
        pool = list(recs)
    if not pool or len({r.get("work_id") for r in pool}) != 1:
        return None
    best = max(pool, key=lambda r: _to_int(r.get("ratings_count")))
    n_words = len(_norm_title(book.get("title", "")).split())
    if n_words <= SHORT_TITLE_WORDS and _to_int(best.get("ratings_count")) < MIN_RATINGS_SHORT:
        return None  # generic short title without a popular canonical match -> too risky
    return best


def fill_from_goodreads_titles(
    targets: dict[str, list[dict]],
    books_gz: Path,
    works_gz: Path | None = None,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Stream the goodreads dump once, collect description-bearing records per needed
    title, then resolve each book to one record (or skip if ambiguous)."""
    cand: dict[str, list[dict]] = {}
    for raw in stream_jsonl_gz(books_gz):
        if not (raw.get("description") or "").strip():
            continue
        key = _norm_title(raw.get("title_without_series") or raw.get("title") or "")
        if key in targets:
            cand.setdefault(key, []).append(raw)

    # Only the candidate works are needed, so this is loaded after ``cand`` narrows them.
    work_years: dict[str, int] = {}
    if works_gz:
        keep = {str(r.get("work_id")) for recs in cand.values() for r in recs}
        work_years = load_work_years(works_gz, keep)
        print(f"  work years: {len(work_years)} of {len(keep)} candidate works dated")

    # Resolve every book to a record FIRST (no mutation), so collisions can be dropped
    # before anything is written -- otherwise a discarded match would still have left a
    # wrong description on the book.
    resolved: dict[str, tuple[dict, dict]] = {}  # book id -> (book, chosen record)
    assigned: dict[str, list[str]] = {}  # work_id -> book ids sharing it
    for key, books in targets.items():
        recs = cand.get(key)
        if not recs:
            continue
        for b in books:
            rec = _pick_record(b, recs, work_years)
            if rec is None:
                continue
            resolved[b["id"]] = (b, rec)
            assigned.setdefault(str(rec.get("work_id")), []).append(b["id"])

    # Two of our books resolving to the SAME goodreads work means we couldn't tell them
    # apart -- drop both rather than assign a maybe-wrong description.
    for ids in assigned.values():
        if len(ids) > 1:
            for bid in ids:
                resolved.pop(bid, None)

    changed: dict[str, dict] = {}
    for bid, (b, rec) in resolved.items():
        shelves = [
            s.get("name", "")
            for s in sorted(rec.get("popular_shelves", []), key=lambda s: -_to_int(s.get("count")))
        ]
        if not dry_run:
            _apply(b, (rec["description"] or "").strip(), shelves, rec.get("image_url", ""))
        changed[bid] = b
    return changed


def enrich(
    data_dir: Path = DATA,
    goodreads: Path | None = None,
    ol_works: Path | None = None,
    title_match: bool = False,
    works: Path | None = None,
    re_embed: bool = True,
    dry_run: bool = False,
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
    if goodreads and gr_t:
        got = fill_from_goodreads(gr_t, goodreads)
        print(f"  goodreads (key):   filled {len(got)}")
        changed.update(got)
    if ol_works and ol_t:
        got = fill_from_ol_works(ol_t, ol_works)
        print(f"  OL works (key):    filled {len(got)}")
        changed.update(got)
    if title_match and goodreads:
        # Books with no key join (the OL works) matched by title+year against the same
        # goodreads dump. Runs on what's STILL missing after the key fills above.
        tt = title_targets(_missing(books))
        got = fill_from_goodreads_titles(tt, goodreads, works_gz=works, dry_run=dry_run)
        print(f"  goodreads (title): {'would fill' if dry_run else 'filled'} {len(got)}")
        changed.update(got)

    if not changed:
        print("Nothing filled (no dumps given, or no matches with a description).")
        return {}

    if dry_run:
        print(f"[dry-run] {len(changed)} descriptions would be filled; nothing written.")
        return changed

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
    ap.add_argument(
        "--title-match",
        action="store_true",
        help="Also match no-key books (OL works) by title+year against --goodreads.",
    )
    ap.add_argument(
        "--works",
        type=Path,
        help="UCSD goodreads_book_works.json.gz. Supplies original_publication_year, "
        "which compares like-for-like with our year (the books dump only has edition "
        "years). Improves --title-match precision and recall.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what the title-match would fill without writing or re-embedding.",
    )
    ap.add_argument("--no-embed", action="store_true", help="Skip re-embedding.")
    args = ap.parse_args()
    if not args.goodreads and not args.ol_works:
        ap.error("give at least one of --goodreads / --ol-works")
    if args.title_match and not args.goodreads:
        ap.error("--title-match needs --goodreads (it matches against that dump)")
    if args.works and not args.title_match:
        ap.error("--works only affects --title-match")
    enrich(
        goodreads=args.goodreads,
        ol_works=args.ol_works,
        title_match=args.title_match,
        works=args.works,
        re_embed=not args.no_embed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
