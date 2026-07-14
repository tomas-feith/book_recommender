"""Ranking metrics for held-out evaluation.

Given a ranked list of candidate book ids (best-first) and the set of ids that
were held out (books we know the user liked but hid when building the profile),
we ask: did the recommender rank the held-out books highly?
"""

from __future__ import annotations

import math
from collections.abc import Sequence


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
