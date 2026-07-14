"""Ingest the Amazon Reviews 2023 (Books) dataset -> our catalog artifacts.

Another CF source at scale, and structurally the twin of the UCSD Goodreads
adapter: book metadata + rated interactions -> our three artifacts. The Amazon
Books category has tens of millions of ratings, so it's a strong collaborative
signal to complement (or replace) goodbooks.

Source: https://amazon-reviews-2023.github.io/ (same McAuley lab host). Two
gzipped JSON-lines files:
    --meta     meta_Books.jsonl.gz   (per-book metadata)
    --reviews  Books.jsonl.gz        (per-review ratings; user_id, parent_asin)

Books get ``az:`` ids (from parent_asin). Everything downstream -- embeddings,
``sparse_topk_cf`` -- is shared with the other adapters.

Run (in the uv/torch env):
    uv run --no-sync python scripts/ingest_amazon_reviews.py \
        --meta meta_Books.jsonl.gz --reviews Books.jsonl.gz --top-n 25000
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_goodreads_ucsd import _to_int, stream_jsonl_gz  # noqa: E402  (shared helpers)

from eval.data import book_to_text  # noqa: E402

DATA = ROOT / "data"
MIN_RATED_PER_USER = 3
MAX_USERS = 200_000
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def select_top_books(meta_path: Path, top_n: int) -> list[dict]:
    """Keep the ``top_n`` books with the most ratings (by ``rating_number``)."""
    heap: list[tuple] = []
    for raw in stream_jsonl_gz(meta_path):
        if not raw.get("title") or not raw.get("parent_asin"):
            continue
        rc = _to_int(raw.get("rating_number"))
        key = str(raw["parent_asin"])
        if len(heap) < top_n:
            heapq.heappush(heap, (rc, key, raw))
        elif rc > heap[0][0]:
            heapq.heapreplace(heap, (rc, key, raw))
    return [raw for _, _, raw in sorted(heap, key=lambda t: -t[0])]


def _author(raw: dict) -> str:
    details = raw.get("details") or {}
    for k in ("Author", "Authors"):
        if details.get(k):
            return str(details[k])
    return raw.get("store", "") or ""


def _year(raw: dict):
    details = raw.get("details") or {}
    for v in (details.get("Publication date"), details.get("Publication_date")):
        if v and (m := _YEAR_RE.search(str(v))):
            return int(m.group(0))
    return None


def to_record(raw: dict) -> dict:
    desc = raw.get("description")
    if isinstance(desc, list):
        desc = " ".join(desc)
    # Amazon categories start with "Books"; drop that and keep the specifics.
    cats = [c.lower() for c in raw.get("categories", []) if c.lower() != "books"][:5]
    images = raw.get("images") or []
    img = ""
    if images and isinstance(images[0], dict):
        img = images[0].get("large") or images[0].get("thumb") or ""
    return {
        "id": "az:" + str(raw["parent_asin"]),
        "title": raw.get("title", ""),
        "author": _author(raw),
        "subjects": cats,
        "language": "en",
        "year": _year(raw),
        "image": img,
        "description": (desc or "").strip()[:800],
    }


def build_interactions(reviews_path: Path, keep: set) -> dict[str, dict[str, float]]:
    by_user: dict[str, dict[str, float]] = defaultdict(dict)
    for row in stream_jsonl_gz(reviews_path):
        bid = "az:" + str(row.get("parent_asin"))
        rating = _to_int(row.get("rating"))
        uid = row.get("user_id")
        if rating >= 1 and uid and bid in keep:
            by_user[str(uid)][bid] = float(rating)
    filtered = {u: r for u, r in by_user.items() if len(r) >= MIN_RATED_PER_USER}
    if len(filtered) > MAX_USERS:
        filtered = dict(sorted(filtered.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:MAX_USERS])
    return filtered


def ingest(meta_path, reviews_path, top_n, data_dir=DATA):
    from cf_build import sparse_topk_cf

    from app.store import save_cf
    from eval.embedders import SentenceTransformerEmbedder

    print(f"Selecting top {top_n} books by rating count...")
    books = [to_record(r) for r in select_top_books(meta_path, top_n)]
    order = [b["id"] for b in books]
    keep = set(order)
    print(f"  {len(books)} books ({sum(1 for b in books if b['description'])} with descriptions)")

    print("Streaming reviews...")
    by_user = build_interactions(reviews_path, keep)
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
        f"Done. {len(books)} books, sparse CF ({sim.shape[0]}x{sim.shape[1]}, "
        f"{sim.nnz} nnz) from {len(by_user)} users -> {data_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Amazon Reviews 2023 (Books).")
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--reviews", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=25000)
    args = ap.parse_args()
    ingest(args.meta, args.reviews, args.top_n)


if __name__ == "__main__":
    main()
