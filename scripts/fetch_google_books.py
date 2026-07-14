"""Fetch NEW books from the Google Books API -> records for add_books.

The source-side companion to enrich_google_books.py (which fills gaps on books
we already have). This searches Google Books by subject and returns catalog
records for books not yet in the catalog, so add_books / refresh can ingest them
(CF-cold, content-ranked until reactions accrue).

Needs a Google Books API key: put GOOGLE_BOOKS_API_KEY in a repo-root .env, or
pass --api-key. The anonymous quota is a shared, usually-exhausted pool.

Run (writes a JSON list add_books understands):
    uv run --no-sync python scripts/fetch_google_books.py --out gb_new.json --want 40
    uv run --no-sync python scripts/refresh.py --add gb_new.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
API = "https://www.googleapis.com/books/v1/volumes"
UA = {"User-Agent": "book-rec/0.1 (catalog source)"}
_YEAR_RE = re.compile(r"(19|20)\d{2}")

DEFAULT_SUBJECTS = (
    "fiction",
    "science fiction",
    "fantasy",
    "mystery",
    "romance",
    "history",
    "biography",
    "poetry",
)


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _search(subject: str, start: int, key: str, retries: int = 3) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "q": f'subject:"{subject}"',
            "startIndex": start,
            "maxResults": 40,
            "langRestrict": "en",
            "country": "US",
            "orderBy": "relevance",
            "key": key,
        }
    )
    for attempt in range(retries):
        try:
            data = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(f"{API}?{params}", headers=UA), timeout=20
                )
            )
            return data.get("items") or []
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise SystemExit(
                f"Google Books error {e.code} for subject {subject!r}. "
                "429 = quota (need a key); 503 = search backend outage (retry later)."
            ) from e
        except Exception:
            return []
    return []


def to_record(vol: dict) -> dict | None:
    info = vol.get("volumeInfo", {})
    if not info.get("title") or not vol.get("id"):
        return None
    year = None
    if m := _YEAR_RE.search(info.get("publishedDate", "")):
        year = int(m.group(0))
    return {
        "id": "gb:" + vol["id"],
        "title": info["title"],
        "author": ", ".join(info.get("authors", [])[:2]),
        "subjects": [c.lower() for c in info.get("categories", [])][:5],
        "language": info.get("language", "en"),
        "year": year,
        "image": (info.get("imageLinks") or {}).get("thumbnail", ""),
        "description": (info.get("description") or "").strip()[:800],
    }


def _existing(data_dir: Path):
    books = json.loads((data_dir / "real_books.json").read_text(encoding="utf-8"))
    ids = {b["id"] for b in books}
    titles = {
        f"{b['title'].strip().lower()}|{(b.get('author') or '').split(',')[0].strip().lower()}"
        for b in books
    }
    return ids, titles


def fetch_google_books(
    subjects=DEFAULT_SUBJECTS, want: int = 40, api_key: str | None = None, data_dir: Path = DATA
) -> list[dict]:
    _load_dotenv()
    api_key = api_key or os.environ.get("GOOGLE_BOOKS_API_KEY")
    if not api_key:
        raise SystemExit("No API key. Set GOOGLE_BOOKS_API_KEY in .env or pass --api-key.")

    known_ids, known_titles = _existing(data_dir)
    out: list[dict] = []
    seen: set = set()
    for subject in subjects:
        start = 0
        while len(out) < want and start < 200:  # API caps deep paging
            items = _search(subject, start, api_key)
            if not items:
                break
            for vol in items:
                rec = to_record(vol)
                if not rec:
                    continue
                tkey = (
                    f"{rec['title'].strip().lower()}|{rec['author'].split(',')[0].strip().lower()}"
                )
                if rec["id"] in known_ids or rec["id"] in seen or tkey in known_titles:
                    continue
                if not rec["description"]:
                    continue  # only add books with real content
                seen.add(rec["id"])
                out.append(rec)
                if len(out) >= want:
                    break
            start += 40
            time.sleep(0.2)
        if len(out) >= want:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch new books from Google Books.")
    ap.add_argument("--out", required=True, help="Path to write the JSON records.")
    ap.add_argument("--want", type=int, default=40)
    ap.add_argument("--subjects", help="Comma-separated subjects (default: broad set).")
    ap.add_argument("--api-key")
    args = ap.parse_args()
    subjects = (
        tuple(s.strip() for s in args.subjects.split(",")) if args.subjects else DEFAULT_SUBJECTS
    )
    records = fetch_google_books(subjects=subjects, want=args.want, api_key=args.api_key)
    Path(args.out).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fetched {len(records)} new book(s) -> {args.out}")


if __name__ == "__main__":
    main()
