"""Stratified evaluation of the **served** recommender, at (and toward) scale.

The existing scoreboards (``eval.run`` / ``compare_paradigms`` / ``cold_start``)
score the *research* recommenders on embeddings. This one drives the actual serving
stack -- ``app.store.Catalog`` + ``app.recommender.Recommender`` + the FAISS ANN --
so the retrieve-then-rerank path, the adaptive blend, and the ANN approximation are
all measured end to end.

Why it matters for 1M: at that scale the catalog is overwhelmingly **CF-cold**, so the
number that predicts the real experience is Recall on **cold** targets, not the warm
Recall the head-of-catalog comparison reports. We simulate cold by blinding a fraction
of books out of CF + popularity (their embeddings are untouched -- a description exists
the moment a book is added), hold out liked books that fall in the cold set, and ask
whether the served recommender still surfaces them. We report Recall@K split warm/cold,
catalog **coverage**, and per-call **latency**, for the exact scan vs the FAISS ANN.

Run:
    uv run --no-sync python -m eval.served_eval                 # real catalog
    uv run --no-sync python -m eval.served_eval --cold-frac 0.5 --k 10
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import app.ann as annmod
from app.ann import ANNIndex
from app.recommender import REC_POOL_MULT, Recommender
from app.store import Catalog

from .data import load_profiles
from .metrics import recall_at_k

DATA = Path(__file__).resolve().parent.parent / "data"


def blind_cold(cat: Catalog, cold_idx: np.ndarray) -> None:
    """Zero CF + popularity for the cold books, in place (embeddings untouched).

    pop -> 0 makes ``cf_weight`` 0 (pure content), and the CF rows/cols are cleared so
    a cold book contributes no collaborative signal -- exactly a just-added book.
    """
    cat.pop[cold_idx] = 0.0
    # Zero the cold rows AND columns via a warm-only diagonal mask: D @ sim @ D.
    keep = np.ones(len(cat), dtype=np.float32)
    keep[cold_idx] = 0.0
    d = sparse.diags(keep)
    cat.sim = (d @ cat.sim @ d).tocsr()
    cat.sim.eliminate_zeros()


def ranked_order(rec: Recommender, seed_likes: list[int], dislikes: list[int], k_pool: int):
    """Served retrieve-then-rerank order (indices, best first), before list assembly."""
    reactions = {rec.cat.books[i]["id"]: "like" for i in seed_likes}
    reactions.update({rec.cat.books[i]["id"]: "dislike" for i in dislikes})
    liked, disliked, interested = rec._split(reactions)
    cand = rec._candidates(liked, disliked, interested, reactions, {}, want=k_pool)
    if len(cand) == 0:
        return np.array([], dtype=np.int64)
    scores = rec._scores(liked, disliked, interested, cand)
    return cand[np.argsort(-scores)]


def evaluate(
    cat: Catalog,
    profiles: list[dict],
    cold: np.ndarray,
    k_holdout: int,
    k_eval: int,
    seeds: list[int],
) -> dict[str, float]:
    rec = Recommender(cat)  # recomputes cf_weight from the (blinded) pop
    warm_r: list[float] = []
    cold_r: list[float] = []
    all_r: list[float] = []
    shown: set[int] = set()
    t_total, n_calls = 0.0, 0
    for prof in profiles:
        likes = [cat.idx(b) for b in prof["likes"] if b in cat.id_to_idx]
        dislikes = [cat.idx(b) for b in prof.get("dislikes", []) if b in cat.id_to_idx]
        if len(likes) - k_holdout < 3:  # need enough left to seed a profile
            continue
        for s in seeds:
            rng = random.Random(s)
            held = set(rng.sample(likes, k_holdout))
            seed_likes = [i for i in likes if i not in held]
            t = time.perf_counter()
            order = ranked_order(rec, seed_likes, dislikes, REC_POOL_MULT * k_eval)
            t_total += time.perf_counter() - t
            n_calls += 1
            top = order[:k_eval].tolist()
            shown.update(top)
            ol = order.tolist()
            warm_held = {i for i in held if not cold[i]}
            cold_held = {i for i in held if cold[i]}
            if warm_held:
                warm_r.append(recall_at_k(ol, warm_held, k_eval))
            if cold_held:
                cold_r.append(recall_at_k(ol, cold_held, k_eval))
            all_r.append(recall_at_k(ol, held, k_eval))
    return {
        "warm_recall": float(np.mean(warm_r)) if warm_r else float("nan"),
        "cold_recall": float(np.mean(cold_r)) if cold_r else float("nan"),
        "recall": float(np.mean(all_r)) if all_r else float("nan"),
        "coverage": len(shown) / len(cat),
        "ms_per_call": t_total / max(n_calls, 1) * 1000,
        "trials": len(all_r),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified eval of the served recommender.")
    ap.add_argument("--cold-frac", type=float, default=0.4, help="share of catalog blinded cold")
    ap.add_argument("--k-holdout", type=int, default=3, help="liked books hidden per trial")
    ap.add_argument("--k", type=int, default=10, help="eval@K")
    ap.add_argument("--seeds", type=int, default=5, help="random hold-out splits per user")
    ap.add_argument("--seed", type=int, default=42, help="cold-mask seed")
    args = ap.parse_args()

    cat = Catalog.load(DATA)
    profiles = load_profiles(DATA / "real_profiles.json")
    rng = np.random.default_rng(args.seed)
    cold = np.zeros(len(cat), dtype=bool)
    cold[rng.choice(len(cat), size=int(len(cat) * args.cold_frac), replace=False)] = True
    blind_cold(cat, np.where(cold)[0])

    print(
        f"\nSERVED recommender | {len(cat)} books | {int(cold.sum())} blinded cold "
        f"({args.cold_frac:.0%}) | {len(profiles)} users | hold-out={args.k_holdout} | "
        f"eval@{args.k} | {args.seeds} splits/user"
    )
    header = (
        f"{'retrieval':<12} {'warm R@K':>9} {'cold R@K':>9} {'all R@K':>8} "
        f"{'coverage':>9} {'ms/call':>8} {'trials':>7}"
    )
    print(header)
    print("-" * len(header))
    seeds = list(range(args.seeds))

    cat.ann = None  # exact full scan
    m = evaluate(cat, profiles, cold, args.k_holdout, args.k, seeds)
    _row("exact", m)

    annmod.ANN_MIN = 0  # force FAISS on at this (sub-threshold) scale to compare
    cat.ann = ANNIndex.build(np.ascontiguousarray(cat.emb, dtype=np.float32))
    m = evaluate(cat, profiles, cold, args.k_holdout, args.k, seeds)
    _row("faiss-ann", m)
    print()


def _row(name: str, m: dict[str, float]) -> None:
    print(
        f"{name:<12} {m['warm_recall']:>9.3f} {m['cold_recall']:>9.3f} {m['recall']:>8.3f} "
        f"{m['coverage']:>9.3f} {m['ms_per_call']:>8.1f} {m['trials']:>7d}"
    )


if __name__ == "__main__":
    main()
