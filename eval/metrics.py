"""Ranking metrics for held-out evaluation.

Given a ranked list of candidate book ids (best-first) and the set of ids that
were held out (books we know the user liked but hid when building the profile),
we ask: did the recommender rank the held-out books highly?
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: Sequence, relevant: set, k: int) -> float:
    """Normalized discounted cumulative gain (binary relevance)."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: Sequence, relevant: set) -> float:
    """Mean reciprocal rank of the first relevant hit."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


# ---- beyond-accuracy: how varied is a recommendation LIST? -------------------


def intra_list_distance(vectors: np.ndarray) -> float:
    """Mean pairwise cosine *distance* (1 - cos) within a list of L2-normalized
    embeddings. Higher = more internally diverse (the classic ILD metric)."""
    n = len(vectors)
    if n < 2:
        return 0.0
    sims = np.asarray(vectors, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T
    iu = np.triu_indices(n, k=1)
    return float(np.clip(1.0 - sims[iu], 0.0, 2.0).mean())


def genre_entropy(subject_lists: Sequence[Sequence[str]]) -> float:
    """Shannon entropy (bits) of the genre/subject distribution across a list --
    a low value means the list piles onto one or two genres."""
    counts = Counter(s for subs in subject_lists for s in subs)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())
