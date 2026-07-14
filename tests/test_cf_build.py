"""Item-item CF builders: adjusted-cosine KNN and EASE-R."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cf_build import ease_cf, sparse_topk_cf


def test_sparse_topk_shape_symmetry_and_popularity(coread_ratings):
    order, by_user = coread_ratings
    sim, pop = sparse_topk_cf(order, by_user)
    assert sim.shape == (5, 5)
    dense = sim.toarray()
    assert np.allclose(np.diag(dense), 0.0)  # no self-similarity
    assert np.allclose(dense, dense.T)  # symmetrized
    # everyone rated i4 (40), the camps rated their pair (20 each), i-of-camp only.
    assert pop.tolist() == [20.0, 20.0, 20.0, 20.0, 40.0]


def test_sparse_topk_co_read_items_are_neighbors(coread_ratings):
    order, by_user = coread_ratings
    dense = sparse_topk_cf(order, by_user)[0].toarray()
    # i0's only positive neighbor is its co-read partner i1 (cross-camp share no users).
    assert dense[0].argmax() == 1
    assert dense[0, 1] > 0
    assert dense[0, 2] == 0.0


def test_ease_shape_and_popularity(coread_ratings):
    order, by_user = coread_ratings
    sim, pop = ease_cf(order, by_user, lam=10.0, k=50)
    assert sim.shape == (5, 5)
    assert pop.tolist() == [20.0, 20.0, 20.0, 20.0, 40.0]


def test_ease_ranks_co_read_partner_highest(coread_ratings):
    order, by_user = coread_ratings
    B = ease_cf(order, by_user, lam=10.0, k=50)[0].toarray()
    # the co-read partner outweighs a cross-camp item...
    assert B[0, 1] > B[0, 2]
    # ...and outweighs the ubiquitous i4, which EASE explains away as popularity.
    assert B[0, 1] > B[0, 4]


def test_ease_topk_truncation(coread_ratings):
    order, by_user = coread_ratings
    B = ease_cf(order, by_user, lam=10.0, k=2)[0].toarray()
    assert (B != 0).sum(axis=1).max() <= 2  # each row keeps <= k neighbors
