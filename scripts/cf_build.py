"""Item-item CF builders -- both emit the same sparse top-k ``(sim, pop)`` shape.

Two builders share the serving format (a symmetric-ish CSR similarity matrix +
per-book rating counts), so ``store.save_cf`` / ``load_cf`` and the recommender's
``_cf_sum`` don't care which one produced the matrix:

* :func:`ease_cf` -- **EASE-R** (Embarrassingly Shallow Auto-Encoder). A single
  closed-form regularized solve, ``B = -P/diag(P)`` with ``P = (XᵀX + λI)⁻¹`` on
  the binary user-item matrix. Measured **+35% Recall@10** over the KNN builder on
  the 10k catalog (0.262 -> 0.355), and truncating B to each item's top-k columns
  keeps ~all of it at ~4 MB. This is the default.

* :func:`ials_cf` -- **implicit ALS** (Hu/Koren/Volinsky 2008). Learns low-rank user
  and item factors by alternating ridge solves, then converts the item factors into the
  same top-k item-item matrix. Its reason to exist is **coverage**: EASE inverts an
  item×item Gram, which caps at ~10k items on this hardware, while iALS is O(nnz·k² +
  N·k³) time and O(N·k) memory -- so it covers the *whole* catalog. Measured on the
  real 100k catalog, CF beats content 2.6x on the books it covers (Recall@10 0.201 vs
  0.078) and reached only 10% of them, which is what made this the binding constraint
  rather than the encoder or the ANN (docs §E2).

* :func:`sparse_topk_cf` -- the older **adjusted-cosine KNN** builder (kept for
  comparison and as a dependency-light fallback). A dense N×N matrix is O(N²) and
  mostly near-zero, so we keep each book's top-k neighbors, built block-wise.

A dense 10k×10k matrix would be ~400 MB (>git's 100 MB limit, ~2 GB RAM); the
sparse top-k output of any builder is ~5-10 MB with no ranking loss.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

# EASE holds two h×h float64 arrays live at peak (the Gram and its inverse), so it costs
# ~16·h² bytes on top of the interaction matrix: 10k -> 1.6 GB, 20k -> 6.4 GB,
# 30k -> 14.4 GB. This is the single knob deciding whether a rebuild finishes or dies,
# so it lives here rather than per caller (refresh.py carried its own 30k default).
#
# 10k is set from *measured* headroom, not the machine's nominal RAM: a 10 GB box shows
# only ~3-5 GB actually available, and at h=20000 the build paged ~2.3 GB while still
# assembling the Gram, before the O(h³) inverse began. At h=10000 a worst-case build
# (200k users, 40M interactions, near-dense Gram) completes in 193 s. It also leaves the
# current catalog bit-identical, whose warm set is exactly 10000.
#
# The real lesson is that a dense inverse is the wrong algorithm past ~10k items on this
# hardware, which makes MF/iALS for the tail (docs §B1) the binding constraint on CF
# coverage rather than an optimization: at 250k this covers the top 4% of the catalog.
EASE_MAX_ITEMS = 10_000


def _binary_user_item(order: list[str], by_user: dict[str, dict[str, float]]):
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
        shape=(n_users, n),
        dtype=np.float32,
    )
    pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    return X, pop


def _topk_rows(B: np.ndarray, k: int, block: int = 2048) -> sparse.csr_matrix:
    """Keep each row's top-k entries by value; return a CSR matrix.

    Row-blocked on purpose. The obvious whole-matrix form allocates ``-B``, an int64
    ``argpartition`` result and a ``zeros_like`` -- three more h×h arrays on top of B
    itself, i.e. ~12.8 GB at h=20000 where each array is 3.2 GB. Blocking bounds the
    temporaries to ``block`` rows and emits the CSR pieces directly, so cost is O(h·k)
    instead of O(h²).
    """
    n = B.shape[0]
    if k >= n:
        return sparse.csr_matrix(B.astype(np.float32))
    rows, cols, vals = [], [], []
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        chunk = B[lo:hi]
        idx = np.argpartition(-chunk, k, axis=1)[:, :k]  # k largest per row
        ri = np.arange(hi - lo)[:, None]
        rows.append(np.repeat(np.arange(lo, hi), k))
        cols.append(idx.ravel())
        vals.append(chunk[ri, idx].ravel().astype(np.float32))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
        dtype=np.float32,
    )


def _item_gram(Xw: sparse.csr_matrix, h: int, block: int = 2048) -> np.ndarray:
    """Dense item-item Gram ``Xw.T @ Xw``, filled a column block at a time.

    ``(Xw.T @ Xw).todense()`` materializes a sparse intermediate first, and at 20k
    popular items that product is nearly dense -- ~4e8 stored entries, ~4.8 GB of CSR,
    before the dense copy. Writing straight into a preallocated dense array skips it.
    """
    G = np.empty((h, h), dtype=np.float64)
    XwT = Xw.T.tocsr()
    Xc = Xw.tocsc()  # column slicing is what CSC is for
    for lo in range(0, h, block):
        hi = min(lo + block, h)
        G[:, lo:hi] = (XwT @ Xc[:, lo:hi]).toarray()
    return G


def ease_cf(
    order: list[str],
    by_user: dict[str, dict[str, float]],
    lam: float = 1000.0,
    k: int = 50,
    max_items: int = EASE_MAX_ITEMS,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return (sim, pop): EASE-R item-item weights truncated to top-k per row.

    ``order`` is the canonical book id order; ``by_user`` maps user -> {book_id:
    rating} (ratings are binarized -- EASE uses implicit co-occurrence). ``lam`` is
    the L2 regularizer (~1000 was optimal at 10k); ``k`` caps neighbors per item
    so the stored matrix stays sparse (~4 MB at k=50). Only books in ``order``
    count. Scored exactly like the KNN matrix: ``sim[cand][:, seed].sum(axis=1)``.

    **Scale: solve over the warm head only.** EASE's closed form inverts an
    item×item Gram -- O(H²) memory, O(H³) time -- so a dense N×N solve is impossible
    past ~30-50k items (a 1M×1M Gram alone is 8 TB). But an item with no
    interactions is *decoupled* in the Gram: its user column is empty, so its
    off-diagonal Gram entries are zero and its EASE row/column come out all-zero
    anyway. So we solve EASE only over the **warm** items (``pop > 0``), capped at
    the ``max_items`` most-rated, and scatter that block back into the full N×N
    matrix with cold items left empty. When ``warm <= max_items`` this is **exact**
    -- bit-for-bit the same neighbors as solving the whole matrix -- while bounding
    the dense inverse to H×H regardless of catalog size, which is what lets
    refresh/rebuild run at 1M items. (When the warm set outgrows the dense budget,
    the dropped tail loses CF and falls to content; that's the point to add
    MF/iALS for the tail -- see docs/scaling-to-1m.md §B1.)
    """
    X, pop = _binary_user_item(order, by_user)
    return ease_from_X(X, pop, lam=lam, k=k, max_items=max_items)


def ease_from_X(
    X: sparse.csr_matrix,
    pop: np.ndarray,
    lam: float = 1000.0,
    k: int = 50,
    max_items: int = EASE_MAX_ITEMS,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """EASE-R over a prebuilt binary users×items matrix -- see :func:`ease_cf`.

    Split out so large ingests can stream ``X`` straight into numpy/CSR instead of
    materializing a ``{user: {book: rating}}`` dict first. At 100M+ interactions that
    dict is hundreds of GB of Python objects; the CSR is ~8 bytes per interaction.
    """
    n = X.shape[1]
    warm = np.where(pop > 0)[0]
    if len(warm) == 0:
        return sparse.csr_matrix((n, n), dtype=np.float32), pop
    if len(warm) > max_items:  # keep the most-rated items (richest CF signal)
        keep = np.argsort(-pop[warm], kind="stable")[:max_items]
        warm = np.sort(warm[keep])
    h = len(warm)

    # Closed-form EASE on the warm sub-catalog: G = XᵀX (item Gram); B[i,j] = -P[i,j]/P[j,j].
    # Written to keep only TWO h×h arrays live at once (~16·h² bytes): the naive form
    # holds G, P and B together and then three more inside the top-k, which is ~9.6 GB
    # and ~12.8 GB at h=20000.
    Xw = X[:, warm].tocsr()
    G = _item_gram(Xw, h)
    G[np.diag_indices(h)] += lam
    P = np.linalg.inv(G)
    del G  # free before deriving B, which is the peak
    diag = np.diag(P).copy()
    P /= -diag  # in place: P becomes B = -P / diag(P)
    np.fill_diagonal(P, 0.0)
    block = _topk_rows(P, k).tocoo()  # (h, h) top-k per warm row
    del P

    # Scatter the warm block back to full catalog coordinates; cold items stay empty.
    sim = sparse.csr_matrix(
        (block.data, (warm[block.row], warm[block.col])),
        shape=(n, n),
        dtype=np.float32,
    )
    return sim, pop


def _als_half(
    Y: np.ndarray,
    Xr: sparse.csr_matrix,
    reg: float,
    alpha: float,
    out: np.ndarray,
    weighted_reg: bool = True,
) -> None:
    """One ALS half-step: solve for every row of ``out`` given fixed factors ``Y``.

    The Hu/Koren/Volinsky trick is that the normal equations for row ``u`` are
    ``(YᵀY + α·Y_uᵀY_u + λI) x_u = (1+α)·Σ_{i∈u} y_i``, where ``Y_u`` is just the rows
    ``u`` interacted with. ``YᵀY`` is computed once for all rows (k×k), so each row only
    pays for its own non-zeros -- which is what makes this O(nnz·k²) instead of
    O(users·items·k).

    ``weighted_reg`` scales the ridge by each row's observation count (``λ·n_u·I``,
    ALS-WR / Zhou et al. 2008) instead of a flat ``λI``. A single global λ cannot be
    right for both halves: instrumenting the solver on the real 100k catalog showed the
    two Gram matrices ~21x apart (diag ~93 for the user step, ~1940 for the item step),
    so λ was 0.80% of the user solve and 0.048% of the item solve -- regularizing users
    ~17x harder than items. Scaling by ``n_u`` makes the penalty track how much evidence
    a row actually has, which balances the halves and stops rows with one interaction
    being fit as confidently as rows with five hundred.
    """
    k = Y.shape[1]
    G = Y.T @ Y
    eye = np.eye(k, dtype=np.float64)
    Greg = G + reg * eye  # flat-λ path: fold the ridge in once
    indptr, indices = Xr.indptr, Xr.indices
    for u in range(Xr.shape[0]):
        lo, hi = indptr[u], indptr[u + 1]
        if lo == hi:
            out[u] = 0.0
            continue
        Yu = Y[indices[lo:hi]]  # (n_u, k)
        A = G + (reg * (hi - lo)) * eye if weighted_reg else Greg
        A = A + alpha * (Yu.T @ Yu)
        b = (1.0 + alpha) * Yu.sum(axis=0)
        out[u] = np.linalg.solve(A, b)


def ials_cf(
    X: sparse.csr_matrix,
    pop: np.ndarray,
    k: int = 50,
    factors: int = 64,
    iters: int = 12,
    reg: float = 10.0,
    alpha: float = 40.0,
    seed: int = 0,
    block: int = 512,
    weighted_reg: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return (sim, pop): implicit-ALS item factors reduced to top-k item-item.

    Same output contract as :func:`ease_cf`, so nothing downstream changes -- but it
    covers **every item with an interaction** instead of the ``EASE_MAX_ITEMS`` head,
    because it never forms an item×item matrix. Memory is ``(users + items) × factors``
    floats (a few tens of MB at 100k items) rather than ``16·h²`` bytes.

    ``alpha`` is the confidence weight on observed interactions (implicit feedback has
    no negatives -- unobserved is *low confidence* zero, not a dislike). ``factors`` and
    ``iters`` trade fit against build time.

    **alpha=40 is tuned, reg is not load-bearing.** Swept on the real 100k catalog
    (50k-user subsample, CF-only Recall@10 in a tail-only pool): alpha 1/10/40 gives
    0.283/0.308/0.327 and then falls away sharply -- 0.290 at 80, 0.275 at 160, 0.264 at
    320. So 40 is a genuine optimum, worth **+7.5%** over the alpha=10 this used to
    default to. ``reg`` is nearly inert by comparison: across 0.1 -> 100 at matched
    alpha it moves Recall by less than adjacent configs wiggle, so it is left at 10
    rather than chased.

    Note this walks back to the Hu/Koren/Volinsky paper's own alpha=40. It had been set
    to 10 on the strength of a 60-item planted synthetic where alpha=40 looked like
    overfitting -- that toy validates the *solver* (see tests) but gave actively wrong
    hyperparameter guidance, which is why these are now tuned on real data.

    The final step converts factors to the served format: a blocked ``V @ Vᵀ`` keeping
    each row's top-``k``. That is O(N²·f) work but streams in row blocks, so memory is
    ``block × N`` rather than N².
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = X.shape
    Xr = X.tocsr()
    Xc = X.T.tocsr()  # items × users, for the item half-step

    U = np.zeros((n_users, factors), dtype=np.float64)
    V = rng.standard_normal((n_items, factors)) * 0.01

    for _ in range(iters):
        _als_half(V, Xr, reg, alpha, U, weighted_reg)
        _als_half(U, Xc, reg, alpha, V, weighted_reg)

    return _topk_from_factors(V, k=k, block=block), pop


def _row_normalize(sim: sparse.csr_matrix) -> sparse.csr_matrix:
    """Scale each row to unit sum, so a row's *shape* carries the signal, not its scale.

    Required before mixing builders. Measured on the real 100k catalog, EASE weights
    average 0.021 with row sums ~1.06, while iALS cosines average 0.898 with row sums
    ~44.9 -- a 43x difference. ``_cf_sum`` adds these across a user's likes, so an
    unnormalized merge would let whichever block has the larger scale win every
    comparison regardless of fit.
    """
    out = sim.copy().astype(np.float32)
    rs = np.asarray(out.sum(axis=1)).ravel()
    rs[rs <= 0] = 1.0
    out = sparse.diags((1.0 / rs).astype(np.float32)) @ out
    return out.tocsr()


def hybrid_cf(
    X: sparse.csr_matrix,
    pop: np.ndarray,
    k: int = 50,
    max_items: int = EASE_MAX_ITEMS,
    **ials_kwargs,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """EASE rows for the popular head, iALS rows for everything else.

    Neither builder wins outright. Measured on the real 100k catalog with a fixed
    popularity split at 10k, EASE gets Recall@10 0.242 on the head and 0.008 on the
    tail; iALS gets 0.169 and 0.104. EASE is the stronger model where its O(N³) solve
    fits, and iALS is the only one that reaches the other 90% -- so use each where it
    wins, which is what docs §B1 meant by "add MF/iALS *for the tail*".

    Rows are the unit of choice because ``_cf_sum`` scores a candidate from its own row.
    Both blocks are row-normalized first (see :func:`_row_normalize`); without that the
    iALS block's ~43x larger values would bury the EASE block.
    """
    warm = np.where(pop > 0)[0]
    head = warm[np.argsort(-pop[warm], kind="stable")[:max_items]]

    ease, _ = ease_from_X(X, pop, k=k, max_items=max_items)
    ials, _ = ials_cf(X, pop, k=k, **ials_kwargs)
    ease, ials = _row_normalize(ease), _row_normalize(ials)

    # Keep EASE rows for the head, iALS rows for the rest.
    is_head = np.zeros(X.shape[1], dtype=bool)
    is_head[head] = True
    keep_e = sparse.diags(is_head.astype(np.float32))
    keep_i = sparse.diags((~is_head).astype(np.float32))
    sim = ((keep_e @ ease) + (keep_i @ ials)).tocsr()
    sim.eliminate_zeros()
    return sim, pop


def _topk_from_factors(V: np.ndarray, k: int, block: int = 512) -> sparse.csr_matrix:
    """Top-k item-item similarity from factors, in row blocks.

    Cosine over factors (not raw dot) so a popular item with a large-norm vector does
    not dominate every row's neighbour list -- the served ``_cf_sum`` adds these up
    across a user's likes, so unnormalized magnitudes would reintroduce a popularity
    bias the adaptive blend is meant to control.
    """
    n = V.shape[0]
    Vn = np.ascontiguousarray(V, dtype=np.float32)
    norms = np.linalg.norm(Vn, axis=1, keepdims=True)
    np.divide(Vn, np.maximum(norms, 1e-8), out=Vn)

    rows, cols, vals = [], [], []
    kk = min(k, n - 1)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        S = Vn[lo:hi] @ Vn.T  # (b, n)
        S[np.arange(hi - lo), np.arange(lo, hi)] = -np.inf  # no self-similarity
        idx = np.argpartition(-S, kk, axis=1)[:, :kk]
        ri = np.arange(hi - lo)[:, None]
        rows.append(np.repeat(np.arange(lo, hi), kk))
        cols.append(idx.ravel())
        vals.append(S[ri, idx].ravel())
    v = np.concatenate(vals)
    keep = v > 0  # negative "similarity" is not evidence of co-reading
    return sparse.csr_matrix(
        (v[keep], (np.concatenate(rows)[keep], np.concatenate(cols)[keep])),
        shape=(n, n),
        dtype=np.float32,
    )


def sparse_topk_cf(
    order: list[str],
    by_user: dict[str, dict[str, float]],
    k: int = 100,
    block: int = 500,
) -> tuple[sparse.csr_matrix, np.ndarray]:
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
        shape=(n, len(users)),
        dtype=np.float32,
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
        np.fill_diagonal(S[:, start:stop], 0.0)  # drop self-similarity
        for r_local in range(stop - start):
            row = S[r_local]
            if k < n:
                cand = np.argpartition(-row, k)[:k]
            else:
                cand = np.arange(n)
            cand = cand[row[cand] > 0.0]  # only positive neighbors
            cand = cand[np.argsort(-row[cand])]
            indices.extend(int(c) for c in cand)
            data.extend(float(row[c]) for c in cand)
            indptr.append(len(indices))

    sim = sparse.csr_matrix(
        (
            np.array(data, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int64),
        ),
        shape=(n, n),
    )
    # Adjusted cosine is symmetric; top-k per row is not, so union the two views
    # (values agree where both present, since sim(i,j) == sim(j,i)).
    return sim.maximum(sim.T).tocsr(), pop
