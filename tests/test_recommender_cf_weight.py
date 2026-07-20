"""cf_weight must track CF *evidence*, not popularity."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from app.recommender import POP_REF, Recommender


class _Cat:
    """Minimal Catalog stand-in: only pop/sim/emb are needed to build a Recommender."""

    def __init__(self, pop, sim):
        self.pop = np.asarray(pop, dtype=np.float32)
        self.sim = sim
        self.emb = np.zeros((len(self.pop), 4), dtype=np.float32)


def test_cf_weight_zero_when_the_book_has_no_cf_row():
    # Book 0 is popular AND in the EASE solve; book 1 is just as popular but fell
    # outside max_items, so its sim row is empty; book 2 has no ratings at all.
    sim = sparse.csr_matrix(
        (np.array([0.5], dtype=np.float32), (np.array([0]), np.array([2]))), shape=(3, 3)
    )
    rec = Recommender(_Cat([1000.0, 1000.0, 0.0], sim))
    assert rec.cf_weight[0] > 0.9  # has evidence -> trusted
    assert rec.cf_weight[1] == 0.0  # popular but no CF row -> no CF term at all
    assert rec.cf_weight[2] == 0.0  # genuinely cold


def test_cf_weight_still_scales_with_popularity_when_evidence_exists():
    rows = np.array([0, 1])
    sim = sparse.csr_matrix(
        (np.array([0.5, 0.5], dtype=np.float32), (rows, np.array([1, 0]))), shape=(2, 2)
    )
    rec = Recommender(_Cat([10.0, 1000.0], sim))
    assert rec.cf_weight[0] < rec.cf_weight[1]
    assert np.isclose(rec.cf_weight[0], np.log1p(10.0) / np.log1p(POP_REF), atol=1e-6)
