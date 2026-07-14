"""Ingest the UCSD Goodreads dataset -> our catalog artifacts.

goodbooks-10k is a frozen 10k snapshot. The UCSD Goodreads dataset
(https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) is its natural
superset: ~2.3M books and ~876M interactions **with ratings**, so scaling to it
keeps the collaborative signal strong instead of dumping everything into
cold-start. This adapter maps that source into the exact three artifacts the
serving app already loads.

It streams the files (never loads them whole), selects the top-N books by
rating count, and builds:

* ``data/real_books.json``      -- metadata, ids prefixed ``gr:`` (no goodbooks clash)
* ``data/real_embeddings.npz``  -- bge-small vectors (needs the torch env)
* ``data/real_cf.npz``          -- sparse top-k CF from the real interactions

Inputs (download once from the UCSD page; gzipped JSON-lines / CSV):
    --books        goodreads_books.json.gz              (required)
    --interactions goodreads_interactions_dedup.json.gz (required; string ids)
    --genres       goodreads_book_genres_initial.json.gz (optional, better genres)
    --authors      goodreads_book_authors.json.gz       (optional, for author names)

A single-genre subset (e.g. goodreads_books_fantasy_paranormal.json.gz + its
interactions) is a far smaller, self-contained way to try this first.

Run (in the uv/torch env):
    uv run --no-sync python scripts/ingest_goodreads_ucsd.py \
        --books goodreads_books.json.gz \
        --interactions goodreads_interactions_dedup.json.gz \
        --genres goodreads_book_genres_initial.json.gz \
        --authors goodreads_book_authors.json.gz \
        --top-n 25000
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.data import book_to_text  # noqa: E402

DATA = ROOT / "data"

LANG_MAP = {"eng": "en", "en-US": "en", "en-GB": "en", "en-CA": "en", "": "en"}

# Swipe/rating scale is 1-5; UCSD ratings are already 0-5 (0 == no rating).
MIN_RATED_PER_USER = 3  # users with too little signal add noise to CF
MAX_USERS = 200_000  # cap CF training users to bound memory


def stream_jsonl_gz(path: Path) -> Iterable[dict]:
    """Yield one dict per line from a gzipped JSON-lines file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- metadata ----------------------------------------------------------------


def load_authors(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    return {str(a["author_id"]): a.get("name", "") for a in stream_jsonl_gz(path)}


def load_genres(path: Path | None, keep: set | None = None) -> dict[str, list[str]]:
    """book_id -> ranked genre list (UCSD genres file is {book_id, genres:{g:count}}).

    The genres file covers ALL ~2.3M books; pass ``keep`` (a set of raw book_ids)
    to hold only the selected subset in memory.
    """
    if not path:
        return {}
    out: dict[str, list[str]] = {}
    for row in stream_jsonl_gz(path):
        bid = str(row["book_id"])
        if keep is not None and bid not in keep:
            continue
        genres = row.get("genres") or {}
        out[bid] = [g for g, _ in sorted(genres.items(), key=lambda kv: -kv[1])][:5]
    return out


def _to_int(x, default=0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def select_top_books(books_path: Path, top_n: int) -> list[dict]:
    """Stream the books file, keep the ``top_n`` with the most ratings (min-heap)."""
    heap: list[tuple] = []  # (ratings_count, book_id, raw)
    for i, raw in enumerate(stream_jsonl_gz(books_path)):
        rc = _to_int(raw.get("ratings_count"))
        if not raw.get("title"):
            continue
        key = (rc, str(raw.get("book_id", i)))
        if len(heap) < top_n:
            heapq.heappush(heap, (rc, key[1], raw))
        elif rc > heap[0][0]:
            heapq.heapreplace(heap, (rc, key[1], raw))
    return [raw for _, _, raw in sorted(heap, key=lambda t: -t[0])]


def to_record(raw: dict, authors: dict[str, str], genres: dict[str, list[str]]) -> dict:
    bid = str(raw["book_id"])
    author_names = [authors.get(str(a.get("author_id")), "") for a in raw.get("authors", [])]
    author_names = [a for a in author_names if a][:2]
    subs = genres.get(bid) or [
        s["name"].lower()
        for s in sorted(raw.get("popular_shelves", []), key=lambda s: -_to_int(s.get("count")))[:5]
    ]
    year = _to_int(raw.get("publication_year")) or None
    return {
        "id": "gr:" + bid,
        "title": raw.get("title", ""),
        "author": ", ".join(author_names),
        "subjects": subs,
        "language": LANG_MAP.get(raw.get("language_code", ""), raw.get("language_code") or "en"),
        "year": year,
        "image": raw.get("image_url", ""),
        "description": (raw.get("description") or "").strip()[:800],
    }


# ---- interactions ------------------------------------------------------------


def build_interactions(path: Path, keep: set) -> dict[str, dict[str, float]]:
    """Stream interactions, keeping rated ones for selected books.

    Returns {user_id: {gr_book_id: rating}}. Users with < MIN_RATED_PER_USER
    ratings are dropped; the set is capped at MAX_USERS.
    """
    by_user: dict[str, dict[str, float]] = defaultdict(dict)
    for row in stream_jsonl_gz(path):
        bid = "gr:" + str(row.get("book_id"))
        rating = _to_int(row.get("rating"))
        if rating >= 1 and bid in keep:
            by_user[str(row["user_id"])][bid] = float(rating)
    filtered = {u: r for u, r in by_user.items() if len(r) >= MIN_RATED_PER_USER}
    if len(filtered) > MAX_USERS:
        # deterministic cap: users with the most ratings (richest CF signal)
        top = sorted(filtered.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:MAX_USERS]
        filtered = dict(top)
    return filtered


# ---- driver ------------------------------------------------------------------


def ingest(books_path, interactions_path, genres_path, authors_path, top_n, data_dir=DATA):
    from cf_build import sparse_topk_cf

    from app.store import save_cf
    from eval.embedders import SentenceTransformerEmbedder

    print(f"Selecting top {top_n} books by rating count...")
    raw_books = select_top_books(books_path, top_n)
    keep_raw = {str(r["book_id"]) for r in raw_books}
    authors = load_authors(authors_path)
    genres = load_genres(genres_path, keep=keep_raw)
    books = [to_record(r, authors, genres) for r in raw_books]
    order = [b["id"] for b in books]
    keep = set(order)
    print(
        f"  selected {len(books)} books "
        f"({sum(1 for b in books if b['description'])} with descriptions)"
    )

    print("Streaming interactions (the big one)...")
    by_user = build_interactions(interactions_path, keep)
    print(f"  {len(by_user)} users after filtering")

    model = "BAAI/bge-small-en-v1.5"
    print(f"Embedding {len(books)} books with {model}...")
    emb = SentenceTransformerEmbedder(model).encode([book_to_text(b) for b in books])

    print("Building sparse top-k CF...")
    sim, pop = sparse_topk_cf(order, by_user)

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "real_books.json").write_text(
        json.dumps(books, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        data_dir / "real_embeddings.npz",
        ids=np.array(order, dtype=str),
        emb=emb.astype(np.float32),
        model=np.array(model),
    )
    save_cf(data_dir / "real_cf.npz", order, sim, pop)
    print(
        f"Done. Wrote {len(books)} books, sparse CF ({sim.shape[0]}x{sim.shape[1]}, "
        f"{sim.nnz} nnz) from {len(by_user)} users -> {data_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the UCSD Goodreads dataset.")
    ap.add_argument("--books", type=Path, required=True)
    ap.add_argument("--interactions", type=Path, required=True)
    ap.add_argument("--genres", type=Path)
    ap.add_argument("--authors", type=Path)
    ap.add_argument("--top-n", type=int, default=25000)
    args = ap.parse_args()
    ingest(args.books, args.interactions, args.genres, args.authors, args.top_n)


if __name__ == "__main__":
    main()
