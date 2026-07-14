"""Build a REAL evaluation dataset from goodbooks-10k + Open Library.

Produces two files the harness can consume directly:

* ``data/real_books.json``    -- top-N most-rated books, with real reader-sourced
  genre tags (from Goodreads shelves) and descriptions enriched from Open
  Library by ISBN.
* ``data/real_profiles.json`` -- real users' like/dislike sets, derived from
  their 1-5 star ratings of those books (>=4 = like, <=2 = dislike, 3 ignored).

Everything downloaded is cached in the scratchpad so re-runs are fast. Network
failures during Open Library enrichment degrade gracefully (book falls back to
title+author+tags).

Run:  python scripts/build_real_dataset.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = Path(os.environ["SCRATCHPAD"]) if "SCRATCHPAD" in os.environ else ROOT / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)

GB_BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
UA = {"User-Agent": "book-rec-eval/0.1"}

N_BOOKS = 10000  # top books by rating count (goodbooks-10k is ~10k total)
MIN_RATED = 10  # a user must have rated at least this many of the subset
MAX_RATED = 40  # ...but not be an omnivore who rated a huge share of it
MIN_LIKES = 6  # enough positive signal to build a profile + hold out
MAX_LIKES = 25  # exclude users who "like" nearly everything
MAX_USERS = 120  # cap profiles
OL_WORKERS = 12  # threads for Open Library enrichment

# Goodreads shelves that are not genres -- excluded from a book's subjects.
NON_GENRE = (
    "to-read",
    "currently-reading",
    "read",
    "favorites",
    "favourites",
    "owned",
    "own",
    "books-i-own",
    "i-own",
    "wish",
    "library",
    "kindle",
    "ebook",
    "e-book",
    "audio",
    "audiobook",
    "series",
    "re-read",
    "reread",
    "default",
    "my-books",
    "to-buy",
    "buy",
    "shelf",
    "have",
    "want",
    "unfinished",
    "dnf",
    "abandoned",
    "all-time",
    "book-club",
    "not-read",
    "maybe",
    "tbr",
)


def fetch(url: str, cache_name: str) -> str:
    """Download text with an on-disk cache."""
    cached = CACHE / cache_name
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers=UA)
    text = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace")
    cached.write_text(text, encoding="utf-8")
    return text


def load_csv(text: str):
    return list(csv.DictReader(io.StringIO(text)))


def pad_isbn10(raw: str) -> str:
    raw = (raw or "").strip()
    return raw.zfill(10) if raw and raw.upper().isalnum() else ""


def ol_description(isbn: str) -> str | None:
    if not isbn:
        return None
    try:
        req = urllib.request.Request(f"https://openlibrary.org/isbn/{isbn}.json", headers=UA)
        ed = json.load(urllib.request.urlopen(req, timeout=12))
    except Exception:
        return None
    desc = ed.get("description")
    if not desc and ed.get("works"):
        try:
            wk_url = "https://openlibrary.org" + ed["works"][0]["key"] + ".json"
            wk = json.load(
                urllib.request.urlopen(urllib.request.Request(wk_url, headers=UA), timeout=12)
            )
            desc = wk.get("description")
        except Exception:
            desc = None
    if isinstance(desc, dict):
        desc = desc.get("value")
    if isinstance(desc, str):
        # OL descriptions often trail with source blurbs after a divider.
        return desc.split("--")[0].strip()[:800] or None
    return None


def enrich_descriptions(books):
    """Fill each book's description via Open Library, with an on-disk cache."""
    cache_file = CACHE / "ol_descriptions.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    todo = [b for b in books if b["isbn10"] and b["isbn10"] not in cache]
    print(
        f"Enriching {len(todo)} descriptions from Open Library ({len(books) - len(todo)} cached)..."
    )

    def work(isbn):
        return isbn, ol_description(isbn)

    done = 0
    with ThreadPoolExecutor(max_workers=OL_WORKERS) as pool:
        for isbn, desc in pool.map(work, [b["isbn10"] for b in todo]):
            cache[isbn] = desc or ""
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{len(todo)}")
                cache_file.write_text(json.dumps(cache))
    cache_file.write_text(json.dumps(cache))

    hits = 0
    for b in books:
        d = cache.get(b["isbn10"], "")
        if d:
            hits += 1
        b["description"] = d
    print(f"Descriptions found for {hits}/{len(books)} books.")


def build_books():
    print("Downloading goodbooks-10k metadata...")
    books_rows = load_csv(fetch(GB_BASE + "books.csv", "gb_books.csv"))
    tags_rows = load_csv(fetch(GB_BASE + "tags.csv", "gb_tags.csv"))
    book_tags_rows = load_csv(fetch(GB_BASE + "book_tags.csv", "gb_book_tags.csv"))

    # Top-N by work rating count.
    for r in books_rows:
        r["_rc"] = int(float(r.get("work_ratings_count") or 0))
    top = sorted(books_rows, key=lambda r: r["_rc"], reverse=True)[:N_BOOKS]

    # tag_id -> name, then goodreads_book_id -> ranked genre tags.
    tag_name = {r["tag_id"]: r["tag_name"] for r in tags_rows}

    def is_genre(name: str) -> bool:
        n = name.lower()
        return not any(bad in n for bad in NON_GENRE) and n.replace("-", "").isalpha()

    per_gbid = defaultdict(list)
    for r in book_tags_rows:
        per_gbid[r["goodreads_book_id"]].append((int(r["count"]), tag_name.get(r["tag_id"], "")))

    def subjects_for(gbid: str):
        ranked = sorted(per_gbid.get(gbid, []), reverse=True)
        out = []
        for _, name in ranked:
            if name and is_genre(name) and name.lower() not in out:
                out.append(name.lower())
            if len(out) == 5:
                break
        return out

    lang_map = {"eng": "en", "en-US": "en", "en-GB": "en", "en-CA": "en"}
    books = []
    for r in top:
        year = r.get("original_publication_year") or ""
        try:
            year = int(float(year))
        except ValueError:
            year = None
        books.append(
            {
                "id": r["book_id"],
                "title": r.get("original_title") or r.get("title") or "",
                "author": r.get("authors", ""),
                "subjects": subjects_for(r["goodreads_book_id"]),
                "language": lang_map.get(
                    r.get("language_code", ""), r.get("language_code") or "en"
                ),
                "year": year,
                "image": r.get("image_url", ""),
                "isbn10": pad_isbn10(r.get("isbn", "")),
            }
        )

    enrich_descriptions(books)
    for b in books:
        b.pop("isbn10", None)  # drop helper field

    out = DATA / "real_books.json"
    out.write_text(json.dumps(books, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(books)} books -> {out}")
    return [b["id"] for b in books]  # catalog order


def build_profiles(order):
    print("Streaming ratings.csv (this is the big one)...")
    text = fetch(GB_BASE + "ratings.csv", "gb_ratings.csv")
    subset_ids = set(order)

    by_user = defaultdict(dict)  # user_id -> {book_id: rating}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        bid = row["book_id"]
        if bid in subset_ids:
            by_user[row["user_id"]][bid] = int(row["rating"])

    profiles = []
    for uid, ratings in by_user.items():
        # Focused, moderate readers -- not omnivores who rated a huge share of
        # the catalog. Those are collaborative-filtering territory; a content
        # recommender is meant to help users with a discernible taste.
        if not (MIN_RATED <= len(ratings) <= MAX_RATED):
            continue
        likes = [b for b, s in ratings.items() if s >= 4]
        dislikes = [b for b, s in ratings.items() if s <= 2]
        if not (MIN_LIKES <= len(likes) <= MAX_LIKES):
            continue
        profiles.append({"user": f"gr_{uid}", "likes": likes, "dislikes": dislikes})

    # Deterministic sample, preferring users who also gave dislikes (richer signal).
    profiles.sort(key=lambda p: (len(p["dislikes"]) > 0, p["user"]), reverse=True)
    profiles = profiles[:MAX_USERS]

    out = DATA / "real_profiles.json"
    out.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    n_dis = sum(1 for p in profiles if p["dislikes"])
    avg_likes = sum(len(p["likes"]) for p in profiles) / max(len(profiles), 1)
    print(
        f"Wrote {len(profiles)} profiles -> {out} "
        f"({n_dis} have dislikes, avg {avg_likes:.1f} likes/user)"
    )

    build_cf(order, by_user, {p["user"].removeprefix("gr_") for p in profiles})
    return profiles


def build_cf(order, by_user, profile_uids):
    """EASE-R item-item CF matrix + popularity, saved as .npz.

    Learned ONLY from users NOT in our evaluation profiles, so the CF baseline
    never sees the held-out users' ratings -- a fair comparison against the
    content models. Uses EASE-R (closed-form regularized item-item, top-k
    truncated; see scripts/cf_build.py), which measured +35% Recall@10 over the
    adjusted-cosine KNN builder while keeping the same sparse serving format.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from cf_build import ease_cf

    from app.store import save_cf

    train = {u: r for u, r in by_user.items() if u not in profile_uids}
    sim, pop = ease_cf(order, train)

    out = DATA / "real_cf.npz"
    save_cf(out, order, sim, pop)
    print(
        f"Wrote EASE-R CF ({sim.shape[0]}x{sim.shape[1]}, {sim.nnz} nnz) "
        f"from {len(train)} non-eval users -> {out}"
    )


if __name__ == "__main__":
    order = build_books()
    build_profiles(order)
    print("Done.")
