"""Build a user taste vector from the books they liked / disliked.

Two strategies, both returning a single L2-normalized vector so recommendation
is a cosine-similarity search:

* ``mean``    -- centroid of liked-book vectors. Ignores dislikes entirely.
* ``rocchio`` -- alpha * mean(liked) - beta * mean(disliked). The classic
  relevance-feedback formula; beta < alpha because a dislike is a noisier
  signal than a like (could be genre, mood, or just that one book).

Both are cheap enough to recompute on every swipe. When the profile needs to
represent genuinely multi-modal taste (literary fiction AND hard sci-fi), a
single centroid is the wrong model -- that is the point at which you move to
per-cluster centroids or a small per-user classifier. This module is where that
upgrade will live.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def build_profile(
    liked_vecs: np.ndarray,
    disliked_vecs: np.ndarray | None = None,
    strategy: str = "rocchio",
    alpha: float = 1.0,
    beta: float = 0.5,
) -> np.ndarray:
    """Return a single L2-normalized taste vector.

    ``liked_vecs`` / ``disliked_vecs`` are (n, D) arrays of already-normalized
    book vectors.
    """
    if liked_vecs.shape[0] == 0:
        raise ValueError("cannot build a profile with no liked books")

    profile = alpha * liked_vecs.mean(axis=0)

    if strategy == "rocchio" and disliked_vecs is not None and disliked_vecs.shape[0] > 0:
        profile = profile - beta * disliked_vecs.mean(axis=0)
    elif strategy not in ("mean", "rocchio"):
        raise ValueError(f"unknown profile strategy: {strategy!r}")

    return _l2_normalize(profile.astype(np.float32))


def rank_candidates(
    profile: np.ndarray,
    catalog_vecs: np.ndarray,
    candidate_idx: Sequence[int],
) -> list[int]:
    """Rank ``candidate_idx`` (rows of ``catalog_vecs``) by cosine to profile.

    Returns candidate indices ordered best-first. All vectors are assumed
    L2-normalized, so cosine == dot product.
    """
    cand = np.asarray(candidate_idx)
    scores = catalog_vecs[cand] @ profile
    order = np.argsort(-scores)
    return cand[order].tolist()
