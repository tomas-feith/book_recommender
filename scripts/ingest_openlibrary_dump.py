"""Ingest the Open Library bulk data dumps -> our catalog artifacts.

The breadth source: Open Library publishes CC0 dumps of its whole catalog
(~30M works). Unlike the ratings datasets, these have **no interactions**, so
every book arrives CF-cold and is ranked purely by content — which the adaptive
blend already handles (``cf_weight`` -> 0 when ``pop`` is 0). Use this to widen
the catalog massively; pair it with a ratings source (Goodreads/Amazon) or the
app's own swipe log to grow CF over time.

Source (https://openlibrary.org/developers/dumps), gzipped TSV with 5 columns:
``type <TAB> key <TAB> revision <TAB> last_modified <TAB> JSON``.
    --works    ol_dump_works_latest.txt.gz    (required; title/description/subjects)
    --authors  ol_dump_authors_latest.txt.gz  (optional; resolves author names)

Books get ``ol:`` ids (from the work key). We keep only works with a title and a
description (so embeddings have real content), capped at --top-n.

Run (in the uv/torch env):
    uv run --no-sync python scripts/ingest_openlibrary_dump.py \
        --works ol_dump_works_latest.txt.gz \
        --authors ol_dump_authors_latest.txt.gz --top-n 50000
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.data import book_to_text  # noqa: E402

DATA = ROOT / "data"
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def stream_dump(path: Path) -> Iterable[dict]:
    """Yield the JSON record from each row of an OL TSV dump (5th column)."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                try:
                    yield json.loads(parts[4])
                except json.JSONDecodeError:
                    continue


def _text(val) -> str:
    """OL text fields are a str or a {'type','value'} dict."""
    if isinstance(val, dict):
        val = val.get("value")
    return (val or "").strip() if isinstance(val, (str, type(None))) else ""


def _author_keys(work: dict) -> list[str]:
    out = []
    for a in work.get("authors", []):
        key = (a.get("author") or {}).get("key") if isinstance(a, dict) else None
        if key:
            out.append(key)
    return out


def select_works(works_path: Path, top_n: int) -> list[dict]:
    """First ``top_n`` works that have a title AND a description."""
    out: list[dict] = []
    for work in stream_dump(works_path):
        if work.get("title") and _text(work.get("description")):
            out.append(work)
            if len(out) >= top_n:
                break
    return out


def load_author_names(authors_path: Path | None, needed: set) -> dict[str, str]:
    if not authors_path or not needed:
        return {}
    names: dict[str, str] = {}
    for a in stream_dump(authors_path):
        key = a.get("key")
        if key in needed:
            names[key] = a.get("name", "")
            if len(names) == len(needed):
                break
    return names


def to_record(work: dict, authors: dict[str, str]) -> dict:
    key = work["key"]  # "/works/OL...W"
    author_names = [authors.get(k, "") for k in _author_keys(work)]
    author_names = [a for a in author_names if a][:2]
    covers = [c for c in work.get("covers", []) if isinstance(c, int) and c > 0]
    year = None
    if m := _YEAR_RE.search(_text(work.get("first_publish_date"))):
        year = int(m.group(0))
    return {
        "id": "ol:" + key.rsplit("/", 1)[-1],
        "title": work.get("title", ""),
        "author": ", ".join(author_names),
        "subjects": [s.lower() for s in work.get("subjects", []) if isinstance(s, str)][:5],
        "language": "en",
        "year": year,
        "image": f"https://covers.openlibrary.org/b/id/{covers[0]}-M.jpg" if covers else "",
        "description": _text(work.get("description"))[:800],
    }


def ingest(works_path, authors_path, top_n, data_dir=DATA):
    from app.store import save_cf
    from eval.embedders import SentenceTransformerEmbedder

    print(f"Selecting up to {top_n} works with a title + description...")
    works = select_works(works_path, top_n)
    needed = {k for w in works for k in _author_keys(w)}
    print(f"  {len(works)} works; resolving {len(needed)} authors...")
    authors = load_author_names(authors_path, needed)
    books = [to_record(w, authors) for w in works]
    order = [b["id"] for b in books]

    model = "BAAI/bge-small-en-v1.5"
    print(f"Embedding {len(books)} books with {model}...")
    emb = SentenceTransformerEmbedder(model).encode([book_to_text(b) for b in books])

    # No interactions in the OL dump: every book is CF-cold (empty matrix, pop 0).
    n = len(books)
    sim = sparse.csr_matrix((n, n), dtype=np.float32)
    pop = np.zeros(n, dtype=np.float32)

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
        f"Done. {len(books)} content-only books (CF-cold) -> {data_dir}. "
        "Add a ratings source or accumulate swipes to grow CF."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Open Library bulk dumps.")
    ap.add_argument("--works", type=Path, required=True)
    ap.add_argument("--authors", type=Path)
    ap.add_argument("--top-n", type=int, default=50000)
    args = ap.parse_args()
    ingest(args.works, args.authors, args.top_n)


if __name__ == "__main__":
    main()
