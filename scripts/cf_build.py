"""Item-item CF builders -- both emit the same sparse top-k ``(sim, pop)`` shape.

Two builders share the serving format (a symmetric-ish CSR similarity matrix +
per-book rating counts), so ``store.save_cf`` / ``load_cf`` and the recommender's
``_cf_sum`` don't care which one produced the matrix:

* :func:`ease_cf` -- **EASE-R** (Embarrassingly Shallow Auto-Encoder). A single
  closed-form regularized solve, ``B = -P/diag(P)`` with ``P = (XᵀX + λI)⁻¹`` on
  the binary user-item matrix. Measured **+35% Recall@10** over the KNN builder on
  the 10k catalog (0.262 -> 0.355), and truncating B to each item's top-k columns
  keeps ~all of it at ~4 MB. This is the default.

* :func:`sparse_topk_cf` -- the older **adjusted-cosine KNN** builder (kept for
  comparison and as a dependency-light fallback). A dense N×N matrix is O(N²) and
  mostly near-zero, so we keep each book's top-k neighbors, built block-wise.

A dense 10k×10k matrix would be ~400 MB (>git's 100 MB limit, ~2 GB RAM); the
sparse top-k output of either builder is ~5-10 MB with no ranking loss.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse


def _binary_user_item(order: List[str], by_user: Dict[str, Dict[str, float]]):
    """Return (X, pop): binary users×items CSR and per-item rating counts."""
    idx = {bid: i for i, bid in enumerate(order)}
    n = len(order)
    rows, cols = [], []
    for ui, ratings in enumerate(r for r in by_user.values() if r):
        for bid in ratings:
            j = idx.get(bid)
            if j is not None:
                rows.append(ui)
                cols.append(j)
    n_users = sum(1 for r in by_user.values() if r)
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n), dtype=np.float32,
    )
    pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    return X, pop


def _topk_rows(B: np.ndarray, k: int) -> sparse.csr_matrix:
    """Keep each row's top-k entries by value; return a CSR matrix."""
    n = B.shape[0]
    if k >= n:
        keep = B
    else:
        idx = np.argpartition(-B, k, axis=1)[:, :k]        # k largest per row
        keep = np.zeros_like(B)
        ri = np.arange(n)[:, None]
        keep[ri, idx] = B[ri, idx]
    return sparse.csr_matrix(keep.astype(np.float32))


def ease_cf(
    order: List[str],
    by_user: Dict[str, Dict[str, float]],
    lam: float = 1000.0,
    k: int = 50,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Return (sim, pop): EASE-R item-item weights truncated to top-k per row.

    ``order`` is the canonical book id order; ``by_user`` maps user -> {book_id:
    rating} (ratings are binarized -- EASE uses implicit co-occurrence). ``lam`` is
    the L2 regularizer (~1000 was optimal at 10k); ``k`` caps neighbors per item
    so the stored matrix stays sparse (~4 MB at k=50). Only books in ``order``
    count. Scored exactly like the KNN matrix: ``sim[cand][:, seed].sum(axis=1)``.
    """
    n = len(order)
    X, pop = _binary_user_item(order, by_user)

    # Closed-form EASE: G = XᵀX (item Gram); B[i,j] = -P[i,j]/P[j,j], diag 0.
    G = np.asarray((X.T @ X).todense(), dtype=np.float64)
    G[np.diag_indices(n)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0.0)
    return _topk_rows(B, k), pop


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
