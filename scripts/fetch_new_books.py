"""Pull genuinely-new books from Open Library, mapped to the catalog schema.

This is the *source* side of catalog freshness: goodbooks-10k is a frozen dump,
so new releases have to come from a live feed. We query Open Library's search
API by subject, newest first, keep recent-year results, and shape each into the
record format ``add_books.py`` / ``refresh.py`` consume.

New books get ``ol:``-prefixed ids (e.g. ``ol:OL12345W`` from the work key) so
they never collide with goodbooks' numeric ids. They carry no ratings, so once
ingested they start CF-cold and are ranked by content until reactions accrue.

Network failures degrade gracefully: a book whose description can't be fetched
keeps title+author+subjects; a subject query that fails is skipped.

Run:
    uv run --no-sync python scripts/fetch_new_books.py --out new_books.json --want 20
Then ingest with:
    uv run --no-sync python scripts/refresh.py --add new_books.json
or in one step:
    uv run --no-sync python scripts/refresh.py --fetch-new 20
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UA = {"User-Agent": "book-rec/0.1 (catalog refresh)"}
SEARCH = "https://openlibrary.org/search.json"

LANG_MAP = {"eng": "en", "en": "en", "fre": "fr", "spa": "es", "ger": "de", "ita": "it"}

# Default subjects to pull from -- broad, matches the catalog's popular genres.
DEFAULT_SUBJECTS = (
    "fantasy",
    "science fiction",
    "romance",
    "mystery",
    "historical fiction",
    "young adult",
    "thriller",
    "horror",
)


def _get_json(url: str, timeout: int = 20) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception:
        return None


def _work_description(work_key: str) -> str:
    """Fetch a work's description text (best-effort)."""
    data = _get_json(f"https://openlibrary.org{work_key}.json", timeout=12)
    if not data:
        return ""
    desc = data.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")
    return (desc or "").split("--")[0].strip()[:800] if isinstance(desc, str) else ""


def _search_subject(subject: str, limit: int, min_year: int) -> list[dict]:
    # Recent English books in this subject, ranked by reader count (a quality
    # proxy) rather than raw recency -- 'sort=new' surfaces obscure/foreign noise.
    query = f"subject:{subject} AND language:eng AND first_publish_year:[{min_year} TO 2035]"
    params = urllib.parse.urlencode(
        {
            "q": query,
            "sort": "readinglog",
            "limit": limit,
            "fields": "key,title,author_name,first_publish_year,subject,language,cover_i",
        }
    )
    data = _get_json(f"{SEARCH}?{params}")
    docs = (data or {}).get("docs", [])
    # Require an author and a cover -- cheap signals of a real, catalog-worthy book.
    return [
        d
        for d in docs
        if (d.get("first_publish_year") or 0) >= min_year
        and d.get("author_name")
        and d.get("cover_i")
    ]


def _to_record(doc: dict, with_description: bool = True) -> dict | None:
    key = doc.get("key", "")  # "/works/OL...W"
    title = doc.get("title")
    if not key.startswith("/works/") or not title:
        return None
    subjects = [s.lower() for s in doc.get("subject", []) if s.replace(" ", "").isalpha()][:5]
    langs = doc.get("language") or ["eng"]
    language = "en" if "eng" in langs else LANG_MAP.get(langs[0], "en")
    cover = doc.get("cover_i")
    return {
        "id": "ol:" + key.rsplit("/", 1)[-1],
        "title": title,
        "author": ", ".join(doc.get("author_name", [])[:2]),
        "subjects": subjects,
        "language": language,
        "year": doc.get("first_publish_year"),
        "image": f"https://covers.openlibrary.org/b/id/{cover}-M.jpg" if cover else "",
        "description": _work_description(key) if with_description else "",
    }


def _existing(data_dir: Path):
    """(ids, normalized title|author keys) already in the catalog, for dedup."""
    books = json.loads((data_dir / "real_books.json").read_text(encoding="utf-8"))
    ids = {b["id"] for b in books}
    titles = {
        f"{b['title'].strip().lower()}|{(b.get('author') or '').split(',')[0].strip().lower()}"
        for b in books
    }
    return ids, titles


def fetch_new_books(
    subjects=DEFAULT_SUBJECTS,
    want: int = 20,
    min_year: int = 2022,
    per_subject: int = 15,
    data_dir: Path = DATA,
) -> list[dict]:
    """Return up to ``want`` new-book records not already in the catalog."""
    known_ids, known_titles = _existing(data_dir)
    out: list[dict] = []
    seen: set = set()
    for subject in subjects:
        if len(out) >= want:
            break
        for doc in _search_subject(subject, per_subject, min_year):
            rec = _to_record(doc, with_description=False)  # cheap pass first
            if not rec:
                continue
            tkey = f"{rec['title'].strip().lower()}|{rec['author'].split(',')[0].strip().lower()}"
            if rec["id"] in known_ids or rec["id"] in seen or tkey in known_titles:
                continue
            rec["description"] = _work_description(doc["key"])  # enrich only the keepers
            seen.add(rec["id"])
            out.append(rec)
            if len(out) >= want:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch new books from Open Library.")
    ap.add_argument("--out", required=True, help="Path to write the JSON records.")
    ap.add_argument("--want", type=int, default=20, help="How many books to collect.")
    ap.add_argument("--min-year", type=int, default=2022, help="Earliest publish year.")
    ap.add_argument("--subjects", help="Comma-separated subjects (default: broad set).")
    args = ap.parse_args()

    subjects = (
        tuple(s.strip() for s in args.subjects.split(",")) if args.subjects else DEFAULT_SUBJECTS
    )
    records = fetch_new_books(subjects=subjects, want=args.want, min_year=args.min_year)
    Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fetched {len(records)} new book(s) -> {args.out}")


if __name__ == "__main__":
    main()
