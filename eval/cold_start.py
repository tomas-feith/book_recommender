"""Cold-start simulation: can a recommender surface a RELEVANT unrated book?

We mark a random slice of the catalog as "just added" -- zero ratings -- by
zeroing those books out of the collaborative-filtering matrix and the popularity
counts (a brand-new book has neither co-rating data nor popularity). Content
embeddings are unaffected: a description exists the moment a book is added.

For each user we hold out books they liked that fall in the cold set, and ask
whether the recommender ranks those unrated-but-relevant books highly. This is
the exact regime the "Tinder" onboarding lives in, and the half of the story the
warm-user comparison (eval.compare_paradigms) structurally cannot measure.

Run:  python -m eval.cold_start
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import load_books, load_profiles
from .embedders import HashingEmbedder, SentenceTransformerEmbedder
from .metrics import mrr, ndcg_at_k, recall_at_k
from .recommenders import (
    EmbeddingRecommender,
    HybridRecommender,
    ItemItemCFRecommender,
    PopularityRecommender,
)

DATA = Path(__file__).resolve().parent.parent / "data"
COLD_FRAC = 0.40       # share of catalog treated as newly-added (no ratings)
K_HOLDOUT = 2          # cold liked books hidden per trial
K_EVAL = 10
SEEDS = list(range(5))


def make_cold_mask(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cold = np.zeros(n, dtype=bool)
    cold[rng.choice(n, size=int(n * COLD_FRAC), replace=False)] = True
    return cold


def apply_cold(rec, cold_idx: np.ndarray) -> None:
    """Blind the CF / popularity signal for cold books, in place."""
    if isinstance(rec, ItemItemCFRecommender):
        rec.sim[cold_idx, :] = 0.0
        rec.sim[:, cold_idx] = 0.0
    elif isinstance(rec, PopularityRecommender):
        rec.pop[cold_idx] = 0.0
    elif isinstance(rec, HybridRecommender):
        apply_cold(rec.cf, cold_idx)


def evaluate_cold(rec, books, profiles, cold: np.ndarray) -> Dict[str, float]:
    id_to_idx = {b["id"]: i for i, b in enumerate(books)}
    rec.prepare(books)
    apply_cold(rec, np.where(cold)[0])

    rc = nd = rr = 0.0
    n = 0
    for prof in profiles:
        likes = [id_to_idx[b] for b in prof["likes"] if b in id_to_idx]
        dislikes = [id_to_idx[b] for b in prof.get("dislikes", []) if b in id_to_idx]
        cold_likes = [i for i in likes if cold[i]]
        warm_likes = [i for i in likes if not cold[i]]
        # Need cold books to hide, and enough remaining signal to seed a profile.
        if len(cold_likes) < K_HOLDOUT or len(likes) - K_HOLDOUT < 3:
            continue
        for seed in SEEDS:
            rng = random.Random(seed)
            held = set(rng.sample(cold_likes, K_HOLDOUT))
            seed_likes = [i for i in likes if i not in held]
            reacted = set(seed_likes) | set(dislikes)
            cands = [i for i in range(len(books)) if i not in reacted]

            ranked = rec.rank(seed_likes, dislikes, cands)
            rc += recall_at_k(ranked, held, K_EVAL)
            nd += ndcg_at_k(ranked, held, K_EVAL)
            rr += mrr(ranked, held)
            n += 1
    if n == 0:
        return {"recall": 0, "ndcg": 0, "mrr": 0, "trials": 0}
    return {"recall": rc / n, "ndcg": nd / n, "mrr": rr / n, "trials": n}


def main() -> None:
    books = load_books(DATA / "real_books.json")
    profiles = load_profiles(DATA / "real_profiles.json")
    cf_npz = DATA / "real_cf.npz"
    cold = make_cold_mask(len(books))

    content = EmbeddingRecommender(
        SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"), strategy="rocchio"
    )
    cf = ItemItemCFRecommender(cf_npz)

    recommenders = [
        PopularityRecommender(cf_npz),
        content,
        cf,
        HybridRecommender(content, cf, w_cf=0.5),
    ]

    print(
        f"\nCOLD-START (item): {int(cold.sum())}/{len(books)} books marked "
        f"unrated | {len(profiles)} users | hold-out={K_HOLDOUT} cold likes | "
        f"eval@{K_EVAL}\nHeld-out books have ZERO ratings -- only content can "
        f"see them.\n"
    )
    header = f"{'recommender':<34} {'Recall@K':>9} {'NDCG@K':>8} {'MRR':>7} {'trials':>7}"
    print(header)
    print("-" * len(header))
    for rec in recommenders:
        m = evaluate_cold(rec, books, profiles, cold)
        print(
            f"{rec.name:<34} {m['recall']:>9.3f} {m['ndcg']:>8.3f} "
            f"{m['mrr']:>7.3f} {m['trials']:>7d}"
        )
    print()


if __name__ == "__main__":
    main()
