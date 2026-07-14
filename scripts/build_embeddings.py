"""Precompute and cache book embeddings for the serving app.

Encoding is one-time work the app should not pay on every start (and shouldn't
require torch at serve time at all). This bakes the vectors to disk once; the app
loads a plain numpy array.

By default this uses the **co-read fine-tuned** encoder in ``data/coread-encoder``
when present -- bge-small further trained to place co-read books near each other,
which improves cold-start (see scripts/finetune_coread.py). If that directory is
absent it falls back to stock ``bge-small``, so the pipeline still works before
the encoder is built.

Run in the torch-capable env:
    uv run --no-sync python scripts/build_embeddings.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.data import book_to_text, load_books
from eval.embedders import SentenceTransformerEmbedder

BASE_MODEL = "BAAI/bge-small-en-v1.5"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
COREAD_ENCODER = DATA / "coread-encoder"


def main() -> None:
    books = load_books(DATA / "real_books.json")
    model = str(COREAD_ENCODER) if COREAD_ENCODER.exists() else BASE_MODEL
    label = "coread-finetuned bge-small" if COREAD_ENCODER.exists() else BASE_MODEL
    print(f"Embedding {len(books)} books with {label} ...")
    emb = SentenceTransformerEmbedder(model).encode([book_to_text(b) for b in books])
    out = DATA / "real_embeddings.npz"
    np.savez_compressed(
        out,
        ids=np.array([b["id"] for b in books]),
        emb=emb.astype(np.float16),  # half the file + resident footprint; fp16 ranking is fine
        model=np.array(label),
    )
    print(f"Wrote {emb.shape} {emb.dtype} embeddings -> {out}")


if __name__ == "__main__":
    main()
