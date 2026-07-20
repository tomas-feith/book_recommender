"""Rebuild the CF matrix for an existing catalog, without re-ingesting.

The ingest streams 10.7 GB of interactions to build the user-item matrix, then throws
it away once EASE has run. That makes every CF experiment cost ~50 minutes of I/O,
which is the wrong price for the thing we most want to iterate on -- CF coverage is the
binding constraint on tail quality (docs §E2).

So: build the matrix once, cache it beside the catalog, and rebuild CF from the cache
in minutes. The catalog's own id order is authoritative, so the rebuilt ``sim`` lines up
with the embeddings and metadata already on disk.

    # first run streams the interactions and caches the matrix
    uv run --no-sync python scripts/rebuild_cf.py --data data_100k \
        --interactions .cache/goodreads/goodreads_interactions_dedup.json.gz --method ials

    # subsequent runs are cache hits -- sweep freely
    uv run --no-sync python scripts/rebuild_cf.py --data data_100k --method ials --factors 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cf_build import EASE_MAX_ITEMS, ease_from_X, hybrid_cf, ials_cf  # noqa: E402
from ingest_goodreads_ucsd import (  # noqa: E402
    build_user_item,
    choose_users,
    count_ratings_per_user,
    pick_eval_users,
)

from app.store import save_cf  # noqa: E402

CACHE = "interactions.npz"


def load_or_build_matrix(data_dir: Path, interactions: Path | None):
    """Return (X, pop, order) for the catalog in ``data_dir``, caching X on first build."""
    books = json.loads((data_dir / "real_books.json").read_text(encoding="utf-8"))
    order = [b["id"] for b in books]
    cache = data_dir / CACHE

    if cache.exists():
        print(f"Loading cached interaction matrix from {cache} ...")
        X = sparse.load_npz(cache).tocsr()
        if X.shape[1] != len(order):
            raise SystemExit(
                f"{cache} has {X.shape[1]} items but the catalog has {len(order)}. "
                "The catalog changed; delete the cache to rebuild."
            )
        pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        print(f"  X = {X.shape[0]}x{X.shape[1]}, {X.nnz} interactions")
        return X, pop, order

    if interactions is None:
        raise SystemExit(f"No cache at {cache}; pass --interactions to build it.")

    keep, col_of = set(order), {b: i for i, b in enumerate(order)}
    print("Pass 1/2 over interactions: counting per user...")
    counts = count_ratings_per_user(interactions, keep)
    eval_users = pick_eval_users(counts)
    user_row = choose_users(counts, exclude=eval_users)
    print(f"  {len(counts)} users seen -> {len(user_row)} for CF, {len(eval_users)} held out")
    del counts

    print("Pass 2/2 over interactions: building the user-item matrix...")
    X, pop, _ = build_user_item(interactions, col_of, user_row, len(order), eval_users)
    print(f"  X = {X.shape[0]}x{X.shape[1]}, {X.nnz} interactions")
    sparse.save_npz(cache, X)
    print(f"  cached -> {cache} ({cache.stat().st_size / 1e6:.0f} MB)")
    return X, pop, order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="catalog dir")
    ap.add_argument("--interactions", type=Path, help="source file (only needed to build cache)")
    ap.add_argument("--method", choices=["ease", "ials", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=50, help="neighbours kept per item")
    ap.add_argument("--factors", type=int, default=64, help="iALS latent dimensions")
    ap.add_argument("--iters", type=int, default=12, help="iALS alternations")
    ap.add_argument("--reg", type=float, default=10.0, help="iALS ridge term")
    ap.add_argument("--alpha", type=float, default=10.0, help="iALS confidence weight")
    ap.add_argument("--max-items", type=int, default=EASE_MAX_ITEMS, help="EASE budget")
    ap.add_argument(
        "--out", type=Path, default=None, help="output npz (default <data>/real_cf.npz)"
    )
    args = ap.parse_args()

    X, pop, order = load_or_build_matrix(args.data, args.interactions)

    t = time.perf_counter()
    if args.method == "hybrid":
        print(
            f"hybrid: EASE head (cap {args.max_items}) + iALS tail, "
            f"factors={args.factors} iters={args.iters} reg={args.reg} alpha={args.alpha}"
        )
        sim, pop = hybrid_cf(
            X,
            pop,
            k=args.k,
            max_items=args.max_items,
            factors=args.factors,
            iters=args.iters,
            reg=args.reg,
            alpha=args.alpha,
        )
    elif args.method == "ials":
        print(
            f"iALS: factors={args.factors} iters={args.iters} "
            f"reg={args.reg} alpha={args.alpha} k={args.k}"
        )
        sim, pop = ials_cf(
            X,
            pop,
            k=args.k,
            factors=args.factors,
            iters=args.iters,
            reg=args.reg,
            alpha=args.alpha,
        )
    else:
        n_warm = int((pop > 0).sum())
        print(f"EASE: {n_warm} warm, cap {args.max_items} -> h={min(n_warm, args.max_items)}")
        sim, pop = ease_from_X(X, pop, k=args.k, max_items=args.max_items)
    dt = time.perf_counter() - t

    covered = int((np.diff(sim.indptr) > 0).sum())
    out = args.out or (args.data / "real_cf.npz")
    save_cf(out, order, sim, pop)
    print(
        f"Built {args.method} CF in {dt:.0f}s: {sim.shape[0]}x{sim.shape[1]}, {sim.nnz} nnz, "
        f"covering {covered}/{len(order)} items ({covered / len(order):.1%}) -> {out}"
    )


if __name__ == "__main__":
    main()
