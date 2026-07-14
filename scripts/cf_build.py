"""Sparse top-k item-item CF -- the scalable replacement for the dense N x N matrix.

A dense similarity matrix is O(N^2): at 10k books that's a 400 MB array that
neither fits in a git repo (>100 MB) nor builds comfortably in ~10 GB RAM. But
the matrix is overwhelmingly near-zero -- almost no two books are strong CF
neighbors. So we keep only each book's **top-k** neighbors (k~100), which turns a
400 MB dense matrix into a ~10 MB sparse one with no meaningful loss for ranking.

The similarity itself is unchanged: adjusted (user-mean-centered) cosine, the
standard item-item measure. We just build it block-wise and truncate, so peak
memory is one (block x N) slice rather than the whole thing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse


def sparse_topk_cf(
    order: List[str],
    by_user: Dict[str, Dict[str, float]],
    k: int = 100,
    block: int = 500,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Return (sim, pop): a symmetric CSR top-k similarity matrix + rating counts.

    ``order`` is the canonical book id order; ``by_user`` maps user -> {book_id:
    rating}. Only books in ``order`` are considered.
    """
    idx = {bid: i for i, bid in enumerate(order)}
    n = len(order)

    # Sparse, user-mean-centered ratings: rows = books, cols = users.
    rows, cols, vals = [], [], []
    pop = np.zeros(n, dtype=np.float32)
    users = [u for u, r in by_user.items() if r]
    for col, u in enumerate(users):
        ratings = by_user[u]
        mean = sum(ratings.values()) / len(ratings)
        for bid, r in ratings.items():
            i = idx[bid]
            rows.append(i)
            cols.append(col)
            vals.append(r - mean)
            pop[i] += 1.0
    R = sparse.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)),
        shape=(n, len(users)), dtype=np.float32,
    )

    # L2-normalize each book row so cosine == dot.
    norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    Rn = sparse.diags(1.0 / norms) @ R
    RnT = Rn.T.tocsr()

    # Block-wise similarity + per-row top-k truncation.
    data, indices, indptr = [], [], [0]
    for start in range(0, n, block):
        stop = min(start + block, n)
        S = np.asarray((Rn[start:stop] @ RnT).todense())  # (b, n) dense block
        np.fill_diagonal(S[:, start:stop], 0.0)           # drop self-similarity
        for r_local in range(stop - start):
            row = S[r_local]
            if k < n:
                cand = np.argpartition(-row, k)[:k]
            else:
                cand = np.arange(n)
            cand = cand[row[cand] > 0.0]                   # only positive neighbors
            cand = cand[np.argsort(-row[cand])]
            indices.extend(int(c) for c in cand)
            data.extend(float(row[c]) for c in cand)
            indptr.append(len(indices))

    sim = sparse.csr_matrix(
        (np.array(data, dtype=np.float32),
         np.array(indices, dtype=np.int32),
         np.array(indptr, dtype=np.int64)),
        shape=(n, n),
    )
    # Adjusted cosine is symmetric; top-k per row is not, so union the two views
    # (values agree where both present, since sim(i,j) == sim(j,i)).
    return sim.maximum(sim.T).tocsr(), pop
