"""Incrementally add books to the serving catalog -- without a full rebuild.

A book is three aligned artifacts, all keyed by book id (see ``app/store.py``):

* ``data/real_books.json``      -- metadata
* ``data/real_embeddings.npz``  -- content vector (needs the torch env)
* ``data/real_cf.npz``          -- item-item CF similarity + popularity

This script appends new books to all three, idempotently:

* Metadata is normalized to the catalog schema and appended.
* Embeddings are computed ONLY for the new books, with the *same* model the
  existing vectors were built with (read back from the .npz), so old and new
  vectors live in one comparable space.
* CF is the honest part: a genuinely new book has **no ratings**, so it gets
  ``pop = 0`` and an all-zero similarity row/column. The recommender's adaptive
  blend (``cf_weight`` scales with ``log(pop)``) then ranks it purely by content
  until real reactions accumulate -- exactly the intended cold-start behavior.
  Later, ``scripts/refresh.py`` rebuilds CF from accumulated swipe data.

Books already present (by id) are skipped, so re-running is safe.

Run (in the uv/torch env):
    uv run --no-sync python scripts/add_books.py path/to/new_books.json

The input is a JSON list of book records. Minimum fields: ``id`` and ``title``;
``author``, ``subjects``, ``language``, ``year``, ``image``, ``description`` are
filled with sensible defaults when absent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `uv run python scripts/add_books.py`

from eval.data import book_to_text  # noqa: E402

DATA = ROOT / "data"

# Catalog schema: field -> default for records that omit it.
DEFAULTS = {
    "author": "",
    "subjects": [],
    "language": "en",
    "year": None,
    "image": "",
    "description": "",
}


# ---- artifact I/O ------------------------------------------------------------

def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays) -> None:
    # Temp name must still end in .npz, or np.savez_compressed appends another.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _load_books(data_dir: Path) -> List[dict]:
    return json.loads((data_dir / "real_books.json").read_text(encoding="utf-8"))


# ---- helpers -----------------------------------------------------------------

def _normalize(record: dict) -> dict:
    if not record.get("id") or not record.get("title"):
        raise ValueError(f"book record needs 'id' and 'title': {record!r}")
    out = {"id": str(record["id"]), "title": record["title"]}
    for field, default in DEFAULTS.items():
        out[field] = record.get(field, default)
    return out


def _embed(texts: List[str], model: str) -> np.ndarray:
    from eval.embedders import SentenceTransformerEmbedder  # lazy: needs torch

    return SentenceTransformerEmbedder(model).encode(texts)


# ---- core --------------------------------------------------------------------

def add_books(new_records: Iterable[dict], data_dir: Path = DATA, model: str | None = None) -> int:
    """Append genuinely-new books to the catalog's three artifacts.

    Returns the number of books actually added (0 if all were already present).
    """
    books = _load_books(data_dir)
    known = {b["id"] for b in books}

    to_add, seen = [], set()
    for rec in new_records:
        norm = _normalize(rec)
        if norm["id"] in known or norm["id"] in seen:
            continue  # idempotent: skip existing and in-batch duplicates
        to_add.append(norm)
        seen.add(norm["id"])

    if not to_add:
        print("Nothing to add -- all books already in the catalog.")
        return 0

    from scipy import sparse
    from app.store import load_cf, save_cf

    emb_path = data_dir / "real_embeddings.npz"
    cf_path = data_dir / "real_cf.npz"
    # Materialize arrays and close the handle: on Windows an open NpzFile keeps
    # the .npz locked, which blocks the atomic os.replace below.
    with np.load(emb_path, allow_pickle=True) as z:
        old_emb, old_emb_ids, stored_model = z["emb"], z["ids"].astype(str), str(z["model"])
    old_cf_ids, sim, pop = load_cf(cf_path)

    model = model or stored_model
    if model != stored_model:
        raise SystemExit(
            f"Refusing to mix embedding spaces: existing vectors use {stored_model!r}, "
            f"requested {model!r}. Re-embed the whole catalog to change models."
        )

    print(f"Embedding {len(to_add)} new book(s) with {model} ...")
    new_emb = _embed([book_to_text(b) for b in to_add], model)

    # --- metadata: append
    books.extend(to_add)

    # --- embeddings: stack new rows, extend ids
    new_ids = [b["id"] for b in to_add]
    emb = np.vstack([old_emb, new_emb]).astype(np.float32)
    emb_ids = np.concatenate([old_emb_ids, np.array(new_ids, dtype=str)])

    # --- CF: grow N x N -> M x M with empty rows/cols (cold-start), pop = 0.
    # block_diag places the existing sparse matrix top-left; new books get no
    # neighbors and zero popularity until refresh.py folds in real ratings.
    k = len(to_add)
    grown = sparse.block_diag([sim, sparse.csr_matrix((k, k))], format="csr")
    new_pop = np.concatenate([pop, np.zeros(k, dtype=np.float32)]).astype(np.float32)
    cf_ids = list(old_cf_ids) + new_ids

    # --- write all three (json last: it's the source of truth for membership)
    _atomic_savez(emb_path, ids=emb_ids, emb=emb, model=np.array(model))
    save_cf(cf_path, cf_ids, grown, new_pop)
    _atomic_write_json(data_dir / "real_books.json", books)

    print(f"Added {k} book(s). Catalog is now {len(books)} books "
          f"(new books start CF-cold: pop=0, content-ranked).")
    return k


def main(argv: List[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: add_books.py <new_books.json>")
    records = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("input JSON must be a list of book records")
    add_books(records)


if __name__ == "__main__":
    main(sys.argv[1:])
