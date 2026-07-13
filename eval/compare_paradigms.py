"""Compare recommendation *paradigms* on the real goodbooks-10k data.

Content-based embeddings, item-item collaborative filtering, a popularity floor,
and a content+CF hybrid, all on the identical hold-out scoreboard. This is the
experiment that answers the architectural question: for warm users with a real
rating history, what actually wins -- and by how much?

Run (needs the real dataset built + sentence-transformers installed):
    python -m eval.compare_paradigms
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

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
K_HOLDOUT = 3
K_EVAL = 10
SEEDS = list(range(5))


def evaluate(rec, books, profiles) -> Dict[str, float]:
    id_to_idx = {b["id"]: i for i, b in enumerate(books)}
    rec.prepare(books)

    rc = nd = rr = 0.0
    n = 0
    for prof in profiles:
        likes = [id_to_idx[b] for b in prof["likes"] if b in id_to_idx]
        dislikes = [id_to_idx[b] for b in prof.get("dislikes", []) if b in id_to_idx]
        if len(likes) <= K_HOLDOUT:
            continue
        for seed in SEEDS:
            rng = random.Random(seed)
            held = set(rng.sample(likes, K_HOLDOUT))
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

    content = EmbeddingRecommender(
        SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"), strategy="rocchio"
    )
    cf = ItemItemCFRecommender(cf_npz)

    recommenders = [
        PopularityRecommender(cf_npz),
        EmbeddingRecommender(HashingEmbedder(), strategy="rocchio"),
        content,
        cf,
        HybridRecommender(content, cf, w_cf=0.5),
    ]

    print(
        f"\nReal data: {len(books)} books | {len(profiles)} users | "
        f"hold-out={K_HOLDOUT} | eval@{K_EVAL} | splits/user={len(SEEDS)}\n"
    )
    header = f"{'recommender':<34} {'Recall@K':>9} {'NDCG@K':>8} {'MRR':>7} {'trials':>7}"
    print(header)
    print("-" * len(header))
    for rec in recommenders:
        m = evaluate(rec, books, profiles)
        print(
            f"{rec.name:<34} {m['recall']:>9.3f} {m['ndcg']:>8.3f} "
            f"{m['mrr']:>7.3f} {m['trials']:>7d}"
        )
    print()


if __name__ == "__main__":
    main()
