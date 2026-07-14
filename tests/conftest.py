"""Shared fixtures: tiny synthetic catalogs and rating sets, no data files or torch."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from app.store import Catalog


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@pytest.fixture
def tiny_catalog() -> Catalog:
    """A 6-book catalog with deterministic embeddings and a hand-built CF matrix."""
    books = [
        {
            "id": "b0",
            "title": "Dune",
            "author": "Herbert, Frank",
            "subjects": ["science fiction"],
            "language": "en",
            "year": 1965,
            "description": "Desert planet epic.",
        },
        {
            "id": "b1",
            "title": "Dune Messiah",
            "author": "Herbert, Frank",
            "subjects": ["science fiction"],
            "language": "en",
            "year": 1969,
            "description": "The sequel on Arrakis.",
        },
        {
            "id": "b2",
            "title": "Neuromancer",
            "author": "Gibson, William",
            "subjects": ["science fiction", "cyberpunk"],
            "language": "en",
            "year": 1984,
            "description": "Cyberspace cowboy.",
        },
        {
            "id": "b3",
            "title": "Pride and Prejudice",
            "author": "Austen, Jane",
            "subjects": ["romance", "classics"],
            "language": "en",
            "year": 1813,
            "description": "A romance of manners.",
        },
        {
            "id": "b4",
            "title": "Emma",
            "author": "Austen, Jane",
            "subjects": ["romance", "classics"],
            "language": "en",
            "year": 1815,
            "description": "Matchmaking in a village.",
        },
        {
            "id": "b5",
            "title": "Le Petit Prince",
            "author": "Saint-Exupery, Antoine de",
            "subjects": ["fable"],
            "language": "fr",
            "year": 1943,
            "description": "",
        },
    ]
    rng = np.random.default_rng(0)
    emb = _l2(rng.standard_normal((6, 8)).astype(np.float32))
    # Item-item CF: sci-fi trio (0,1,2) and Austen pair (3,4) are neighbors.
    dense = np.zeros((6, 6), dtype=np.float32)
    for a, b, w in [(0, 1, 0.9), (0, 2, 0.5), (1, 2, 0.4), (3, 4, 0.8)]:
        dense[a, b] = dense[b, a] = w
    sim = sparse.csr_matrix(dense)
    pop = np.array([3000, 40, 500, 800, 5, 0], dtype=np.float32)
    id_to_idx = {b["id"]: i for i, b in enumerate(books)}
    return Catalog(books, emb, sim, pop, id_to_idx)


@pytest.fixture
def coread_ratings() -> tuple[list[str], dict[str, dict[str, float]]]:
    """Two reader camps: (i0,i1) fans and (i2,i3) fans; everyone also rates i4 low.

    Per-user rating variance is deliberate so adjusted-cosine (which mean-centers
    each user) has signal, not just the binary EASE co-occurrence.
    """
    order = [f"i{i}" for i in range(5)]
    by_user: dict[str, dict[str, float]] = {}
    for u in range(40):
        if u % 2 == 0:
            by_user[f"u{u}"] = {"i0": 5.0, "i1": 5.0, "i4": 2.0}
        else:
            by_user[f"u{u}"] = {"i2": 5.0, "i3": 5.0, "i4": 2.0}
    return order, by_user
