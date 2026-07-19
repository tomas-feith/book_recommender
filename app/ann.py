"""Approximate nearest-neighbour retrieval for the content channel.

At 1M books a full ``emb @ profile`` scan (plus argsort) per request is the serving
wall (§C in docs/scaling-to-1m.md). This wraps a FAISS **IVF-PQ** index so the
recommender retrieves a few hundred content candidates in ~1 ms instead of scanning
the catalog, then reranks them with the *exact* embeddings (the PQ codes are lossy, so
retrieval is approximate but the final ranking is exact). PQ also compresses the index
~30x (fp32 1.5 GB -> ~50 MB at 1M), so it's cheap to persist and reload.

**Import-guarded and size-gated.** If faiss isn't installed, or the catalog is
smaller than ``ANN_MIN`` (where an exact scan is faster than IVF overhead), ``build``
returns ``None`` and the recommender falls back to the exact full scan -- so serving
still works numpy-only, just linearly. Embeddings are L2-normalized, so inner product
is cosine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:  # faiss is a serving dep, but keep it optional so the numpy-only path still runs
    import faiss
except Exception:  # pragma: no cover - exercised only where faiss is absent
    faiss = None  # type: ignore[assignment]

# Below this the exact scan (a single GEMV) beats IVF training + probe overhead.
ANN_MIN = 50_000


def _pq_subquantizers(d: int) -> int:
    """PQ subquantizer count -- a divisor of ``d`` near d/8 (8 dims/subquantizer)."""
    for m in (48, 64, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1):
        if d % m == 0:
            return m
    return 1


class ANNIndex:
    """A trained FAISS IVF-PQ index over the catalog embeddings, in catalog-row order."""

    def __init__(self, index: Any, dim: int):
        self._index: Any = index  # faiss ships no stubs; treat as Any
        self.dim = dim

    @classmethod
    def build(cls, emb: np.ndarray, nprobe: int = 48) -> ANNIndex | None:
        """Build an IVF-PQ index over ``emb`` (N, D), or ``None`` if ANN doesn't apply."""
        n = len(emb)
        if faiss is None or n < ANN_MIN:
            return None
        d = int(emb.shape[1])
        nlist = int(min(4096, max(64, 4 * np.sqrt(n))))  # ~4*sqrt(N) cells
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(
            quant, d, nlist, _pq_subquantizers(d), 8, faiss.METRIC_INNER_PRODUCT
        )
        x = np.ascontiguousarray(emb, dtype=np.float32)
        index.train(x)
        index.add(x)
        index.nprobe = int(min(nlist, nprobe))
        return cls(index, d)

    @classmethod
    def load(cls, path: Path) -> ANNIndex | None:
        """Load a persisted index (so serving boot doesn't retrain), or None."""
        if faiss is None or not path.exists():
            return None
        index = faiss.read_index(str(path))
        return cls(index, int(index.d))

    def save(self, path: Path) -> None:
        faiss.write_index(self._index, str(path))

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        """Row indices of the ~``k`` nearest catalog books to ``query`` (best first)."""
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        _, ids = self._index.search(q, int(k))
        out = ids[0]
        return out[out >= 0]  # faiss pads short results with -1

    def add(self, vec: np.ndarray) -> None:
        """Append one catalog vector (keeps ANN row-ids aligned with the catalog)."""
        self._index.add(np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(1, -1)))

    def __len__(self) -> int:
        return int(self._index.ntotal)
