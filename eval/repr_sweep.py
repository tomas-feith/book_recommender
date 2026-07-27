"""Sweep item-text *representation programs* on the held-out harness.

The one idea from AutoIndex (arXiv 2607.18603) that maps onto us: the book->text
transform is a tunable artifact. We hold the encoder fixed (the served co-read
bge-small) and re-embed the catalog under each representation in
``eval.representations.REGISTRY``, then measure whether the change helps -- and
specifically *where*.

Because a representation change only moves the CONTENT channel, we isolate it two
ways per variant:

* ``content R@10 / NDCG@10`` -- content-only ranking (no CF). Pure representation
  quality: is the text better on its own?
* ``hyb-cold R@10``          -- hybrid ranking of *cold* held-out targets, with CF
  blinded on cold books. Here content is the only signal that can see the target,
  so this is the served region a better representation is supposed to move.
* ``hyb R@10``               -- hybrid ranking with full CF and random held-out
  targets. The guardrail: a representation win must not cost the warm/served path.

Encoding is the cost (~3 books/s on this box), so we run on a *profile-anchored
subsample*: every book any profile references (guaranteeing full profile
coverage) plus a random background fill. Absolute recall differs from the 100k
serving catalog, but the *delta between representations* -- the only thing this
script claims -- is what transfers. Nothing here re-embeds the 100k catalog.

Run (real encoder, in the torch env):
    uv run --no-sync python -u -m eval.repr_sweep --n 2000 --seeds 5

Smoke-test the plumbing with zero ML deps (instant):
    python -m eval.repr_sweep --embedder hashing --n 800
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from .metrics import ndcg_at_k, recall_at_k
from .profiles import build_profile
from .representations import REGISTRY

DATA = Path(__file__).resolve().parent.parent / "data"
BETA = 0.5  # dislike down-weight in the CF sum, matching eval.recommenders


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else x - x.mean()


def build_embedder(spec: str):
    """``hashing`` -> numpy baseline; ``coread`` -> served encoder; else an ST name."""
    if spec == "hashing":
        from .embedders import HashingEmbedder

        return HashingEmbedder()
    if spec == "coread":
        from .embedders import SentenceTransformerEmbedder

        coread = DATA / "coread-encoder"
        model = str(coread) if coread.exists() else "BAAI/bge-small-en-v1.5"
        return SentenceTransformerEmbedder(model)
    from .embedders import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(spec)


def make_subsample(books, profiles, n: int, seed: int):
    """Every profile-referenced book (full coverage) + a random background fill."""
    referenced: set[str] = set()
    for p in profiles:
        referenced.update(p.get("likes", []))
        referenced.update(p.get("dislikes", []))
    anchors = [b for b in books if b["id"] in referenced]
    anchor_ids = {b["id"] for b in anchors}
    pool = [b for b in books if b["id"] not in anchor_ids]
    rng = random.Random(seed)
    fill = rng.sample(pool, max(0, min(len(pool), n - len(anchors))))
    sub = anchors + fill
    return sub, len(anchors)


def cf_for_subsample(npz_path: Path, sub: list[dict]):
    """Dense (N,N) CF similarity and popularity aligned to the subsample order."""
    z = np.load(npz_path, allow_pickle=True)
    npz_pos = {str(bid): i for i, bid in enumerate(z["ids"].tolist())}
    perm = np.array([npz_pos[b["id"]] for b in sub])
    sim = sparse.csr_matrix(
        (z["sim_data"], z["sim_indices"], z["sim_indptr"]), shape=tuple(z["sim_shape"])
    )
    sim_sub = np.asarray(sim[perm][:, perm].todense(), dtype=np.float32)
    pop_sub = np.asarray(z["pop"])[perm]
    return sim_sub, pop_sub


def _cf_sum(sim: np.ndarray, cands, seed_likes, dislikes) -> np.ndarray:
    s = sim[:, list(seed_likes)].sum(axis=1)
    if dislikes:
        s = s - BETA * sim[:, list(dislikes)].sum(axis=1)
    return s[cands]


def eval_rep(
    emb: np.ndarray,
    sim_full: np.ndarray,
    sim_blind: np.ndarray,
    cold: np.ndarray,
    prof_idx: list[tuple[list[int], list[int]]],
    k_hold: int,
    k_eval: int,
    seeds: list[int],
    w_cf: float,
) -> dict[str, float]:
    n = emb.shape[0]
    c_rec: list[float] = []
    c_ndcg: list[float] = []
    hyb_rec: list[float] = []
    hyb_cold: list[float] = []
    for likes, dislikes in prof_idx:
        if len(likes) - k_hold < 3:
            continue
        cold_likes = [i for i in likes if cold[i]]
        for s in seeds:
            rng = random.Random(s)
            # --- random-holdout regime: content-only + hybrid with FULL CF ---
            held = set(rng.sample(likes, k_hold))
            seed_likes = [i for i in likes if i not in held]
            reacted = set(seed_likes) | set(dislikes)
            cands = np.fromiter((i for i in range(n) if i not in reacted), dtype=np.int64)
            profile = build_profile(
                emb[seed_likes], emb[dislikes] if dislikes else None, strategy="rocchio"
            )
            content = emb[cands] @ profile
            order_c = cands[np.argsort(-content)].tolist()
            c_rec.append(recall_at_k(order_c, held, k_eval))
            c_ndcg.append(ndcg_at_k(order_c, held, k_eval))
            cf = _cf_sum(sim_full, cands, seed_likes, dislikes)
            hyb = (1 - w_cf) * _standardize(content) + w_cf * _standardize(cf)
            order_h = cands[np.argsort(-hyb)].tolist()
            hyb_rec.append(recall_at_k(order_h, held, k_eval))

            # --- cold-holdout regime: hybrid with BLINDED CF, cold targets ---
            if len(cold_likes) >= k_hold:
                held_c = set(rng.sample(cold_likes, k_hold))
                seed_c = [i for i in likes if i not in held_c]
                reacted_c = set(seed_c) | set(dislikes)
                cands_c = np.fromiter((i for i in range(n) if i not in reacted_c), dtype=np.int64)
                prof_c = build_profile(
                    emb[seed_c], emb[dislikes] if dislikes else None, strategy="rocchio"
                )
                content_c = emb[cands_c] @ prof_c
                cf_c = _cf_sum(sim_blind, cands_c, seed_c, dislikes)
                hyb_c = (1 - w_cf) * _standardize(content_c) + w_cf * _standardize(cf_c)
                order_hc = cands_c[np.argsort(-hyb_c)].tolist()
                hyb_cold.append(recall_at_k(order_hc, held_c, k_eval))

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "content_r": mean(c_rec),
        "content_ndcg": mean(c_ndcg),
        "hyb_r": mean(hyb_rec),
        "hyb_cold_r": mean(hyb_cold),
        "trials": len(c_rec),
        "cold_trials": len(hyb_cold),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", default="coread", help="coread | hashing | <ST model name>")
    ap.add_argument("--n", type=int, default=2000, help="subsample size (anchors + fill)")
    ap.add_argument("--cold-frac", type=float, default=0.40)
    ap.add_argument("--k-holdout", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--w-cf", type=float, default=0.5, help="hybrid CF weight (convex, as scoreboard)"
    )
    ap.add_argument("--fill-seed", type=int, default=13)
    ap.add_argument("--cold-seed", type=int, default=42)
    ap.add_argument(
        "--reps",
        nargs="*",
        default=list(REGISTRY),
        help=f"representations to sweep (default: all). choices: {list(REGISTRY)}",
    )
    ap.add_argument("--books", type=Path, default=DATA / "real_books.json")
    ap.add_argument("--profiles", type=Path, default=DATA / "real_profiles.json")
    ap.add_argument("--cf", type=Path, default=DATA / "real_cf.npz")
    args = ap.parse_args()

    import json

    with open(args.books, encoding="utf-8") as fh:
        books = json.load(fh)
    with open(args.profiles, encoding="utf-8") as fh:
        profiles = json.load(fh)

    sub, n_anchor = make_subsample(books, profiles, args.n, args.fill_seed)
    id_to_idx = {b["id"]: i for i, b in enumerate(sub)}
    sim_full, _pop = cf_for_subsample(args.cf, sub)

    rng = np.random.default_rng(args.cold_seed)
    cold = np.zeros(len(sub), dtype=bool)
    cold[rng.choice(len(sub), size=int(len(sub) * args.cold_frac), replace=False)] = True
    # Blinded CF: zero cold rows AND columns (D @ sim @ D), a just-added book.
    keep = (~cold).astype(np.float32)
    sim_blind = (keep[:, None] * sim_full) * keep[None, :]

    prof_idx: list[tuple[list[int], list[int]]] = []
    for p in profiles:
        likes = [id_to_idx[b] for b in p.get("likes", []) if b in id_to_idx]
        dislikes = [id_to_idx[b] for b in p.get("dislikes", []) if b in id_to_idx]
        if len(likes) - args.k_holdout >= 3:
            prof_idx.append((likes, dislikes))

    embedder = build_embedder(args.embedder)
    name = getattr(embedder, "name", args.embedder)
    seeds = list(range(args.seeds))

    print(
        f"\nREPR SWEEP | encoder={name} | subsample {len(sub)} "
        f"({n_anchor} profile-anchored + {len(sub) - n_anchor} fill) | "
        f"{int(cold.sum())} cold ({args.cold_frac:.0%}) | {len(prof_idx)} users | "
        f"hold-out={args.k_holdout} | eval@{args.k} | {args.seeds} splits\n"
    )
    header = (
        f"{'representation':<20} {'content R@10':>12} {'content NDCG':>13} "
        f"{'hyb R@10':>9} {'hyb-cold R@10':>14} {'trials':>7}"
    )
    print(header)
    print("-" * len(header))

    results: dict[str, dict[str, float]] = {}
    for rep in args.reps:
        fn = REGISTRY[rep]
        t0 = time.perf_counter()
        emb = embedder.encode([fn(b) for b in sub])
        m = eval_rep(
            emb, sim_full, sim_blind, cold, prof_idx, args.k_holdout, args.k, seeds, args.w_cf
        )
        results[rep] = m
        print(
            f"{rep:<20} {m['content_r']:>12.4f} {m['content_ndcg']:>13.4f} "
            f"{m['hyb_r']:>9.4f} {m['hyb_cold_r']:>14.4f} {m['trials']:>7d}"
            f"   ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    # --- deltas vs the production baseline ----------------------------------
    if "full" in results:
        base = results["full"]
        print("\nDelta vs full (production baseline):")
        dh = f"{'representation':<20} {'d content R':>12} {'d content NDCG':>15} {'d hyb R':>9} {'d hyb-cold R':>13}"
        print(dh)
        print("-" * len(dh))
        for rep in args.reps:
            if rep == "full":
                continue
            m = results[rep]
            print(
                f"{rep:<20} {m['content_r'] - base['content_r']:>+12.4f} "
                f"{m['content_ndcg'] - base['content_ndcg']:>+15.4f} "
                f"{m['hyb_r'] - base['hyb_r']:>+9.4f} "
                f"{m['hyb_cold_r'] - base['hyb_cold_r']:>+13.4f}"
            )
    print()


if __name__ == "__main__":
    main()
