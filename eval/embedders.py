"""Pluggable embedders.

Every embedder turns a list of strings into an (N, D) float32 matrix of
L2-normalized row vectors, so cosine similarity is just a dot product.

Two are provided:

* ``HashingEmbedder`` -- pure numpy, no ML dependencies. A bag-of-words hashing
  vectorizer with sub-linear term weighting. It is only a *baseline*: good
  enough to prove the harness works and to give real models something to beat,
  but it has no semantic understanding beyond shared vocabulary.

* ``SentenceTransformerEmbedder`` -- wraps any ``sentence-transformers`` model
  (e.g. ``BAAI/bge-m3``, ``Qwen/Qwen3-Embedding-0.6B``,
  ``sentence-transformers/all-MiniLM-L6-v2``). Only imported if you actually
  request it, so the baseline runs even when torch is not installed.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbedder:
    """Dependency-free bag-of-words baseline (feature hashing)."""

    name = "hashing-baseline"

    def __init__(self, dim: int = 2048, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in self._tokenize(text):
                # Deterministic hash independent of PYTHONHASHSEED.
                h = (hash((self.seed, tok)) & 0x7FFFFFFF) % self.dim
                out[row, h] += 1.0
        # Sub-linear term frequency dampening, then normalize.
        np.log1p(out, out=out)
        return _l2_normalize(out)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Imported lazily."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


def build_embedder(spec: str):
    """Resolve a CLI spec to an embedder instance.

    ``"hashing"`` -> the numpy baseline.
    Anything else is treated as a sentence-transformers model name.
    """
    if spec == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(spec)
