"""Precompute and cache book embeddings for the serving app.

Encoding 400 books is fast, but the app should not pay model load + encode time
on every start (and shouldn't require torch at serve time at all). This bakes the
vectors to disk once; the app loads a plain numpy array.

Run in the torch-capable env:
    .venv39/Scripts/python.exe scripts/build_embeddings.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.data import book_to_text, load_books
from eval.embedders import SentenceTransformerEmbedder

MODEL = "BAAI/bge-small-en-v1.5"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    books = load_books(DATA / "real_books.json")
    print(f"Embedding {len(books)} books with {MODEL} ...")
    emb = SentenceTransformerEmbedder(MODEL).encode([book_to_text(b) for b in books])
    out = DATA / "real_embeddings.npz"
    np.savez_compressed(
        out,
        ids=np.array([b["id"] for b in books]),
        emb=emb.astype(np.float32),
        model=np.array(MODEL),
    )
    print(f"Wrote {emb.shape} embeddings -> {out}")


if __name__ == "__main__":
    main()
