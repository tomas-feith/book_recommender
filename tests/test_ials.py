"""iALS: does the solver recover structure, and does it keep the serving contract?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cf_build import ials_cf


def _planted(n_users=400, n_items=60, per_user=12, seed=0):
    """Two disjoint user communities reading two disjoint item blocks."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    half = n_items // 2
    for u in range(n_users):
        grp = u % 2
        for i in rng.choice(np.arange(grp * half, grp * half + half), per_user, replace=False):
            rows.append(u)
            cols.append(int(i))
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )
    return X, np.asarray(X.sum(axis=0)).ravel().astype(np.float32)


def _purity(sim, n_items=60):
    half = n_items // 2
    ok = tot = 0
    for i in range(n_items):
        nb = sim.getrow(i).indices
        ok += sum((int(j) < half) == (i < half) for j in nb)
        tot += len(nb)
    return ok / max(tot, 1)


def test_ials_recovers_planted_communities():
    X, pop = _planted()
    sim, _ = ials_cf(X, pop, k=5, factors=16, iters=8)
    # Every neighbour should come from the item's own community.
    assert _purity(sim) > 0.95, _purity(sim)


def test_ials_covers_every_interacted_item():
    # The whole point vs EASE: coverage is not capped by a dense inverse.
    X, pop = _planted()
    sim, _ = ials_cf(X, pop, k=5, factors=16, iters=8)
    assert (np.diff(sim.indptr) > 0).sum() == X.shape[1]


def test_ials_keeps_the_serving_contract():
    X, pop = _planted()
    sim, out_pop = ials_cf(X, pop, k=5, factors=8, iters=4)
    assert sparse.isspmatrix_csr(sim)
    assert sim.shape == (X.shape[1], X.shape[1])
    assert sim.dtype == np.float32
    assert np.array_equal(out_pop, pop)  # pop passes through untouched
    assert sim.diagonal().sum() == 0.0  # no self-similarity
    assert (sim.data > 0).all()  # negative similarity is dropped, not stored


def test_ials_handles_items_with_no_interactions():
    X, pop = _planted()
    X = sparse.hstack([X, sparse.csr_matrix((X.shape[0], 5))]).tocsr()
    pop = np.concatenate([pop, np.zeros(5, dtype=np.float32)])
    sim, _ = ials_cf(X, pop, k=5, factors=8, iters=4)
    assert sim.shape == (65, 65)
    assert np.diff(sim.indptr)[-5:].sum() == 0  # cold items get no neighbours
