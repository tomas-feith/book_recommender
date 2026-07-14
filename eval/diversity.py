"""The relevance <-> diversity frontier for the 'For You' list.

Runs the *actual served* recommender (`Recommender.recommend`, adaptive hybrid +
MMR) over the real profiles at several ``mmr_lambda`` values and reports, per
lambda: Recall@10 (relevance), intra-list distance and genre entropy (list
diversity), and catalog coverage (aggregate diversity). This makes the
relevance-for-diversity trade an evidence-based knob instead of a guess.

Run (needs the real dataset built):
    uv run --no-sync python -m eval.diversity
"""

from __future__ import annotations

import random
from pathlib import Path

from app.recommender import Recommender
from app.store import Catalog
from eval.data import load_profiles
from eval.metrics import genre_entropy, intra_list_distance

DATA = Path(__file__).resolve().parent.parent / "data"
K_HOLDOUT = 3
K_EVAL = 10
SEEDS = list(range(5))
LAMBDAS = [1.0, 0.9, 0.7, 0.5, 0.3]


def main() -> None:
    cat = Catalog.load(DATA)
    rec = Recommender(cat)
    profiles = load_profiles(DATA / "real_profiles.json")

    print(
        f"\nFor-You diversity frontier | {len(cat.books)} books | "
        f"{len(profiles)} users | hold-out={K_HOLDOUT} | list={K_EVAL}\n"
    )
    header = f"{'mmr_lambda':>10} {'Recall@10':>10} {'ILD':>7} {'genreH':>7} {'coverage':>9}"
    print(header)
    print("-" * len(header))

    for lam in LAMBDAS:
        rc = ild = ent = 0.0
        n = 0
        shown: set[str] = set()
        for prof in profiles:
            likes = [b for b in prof["likes"] if b in cat.id_to_idx]
            if len(likes) <= K_HOLDOUT:
                continue
            for seed in SEEDS:
                rng = random.Random(seed)
                held = set(rng.sample(likes, K_HOLDOUT))
                seed_likes = [b for b in likes if b not in held]
                reactions = dict.fromkeys(seed_likes, "like")
                recs = rec.recommend(reactions, filters={}, n=K_EVAL, mmr_lambda=lam)
                ids = [s.book["id"] for s in recs]
                idxs = [cat.idx(i) for i in ids]

                rc += len(held & set(ids)) / min(len(held), K_EVAL)
                ild += intra_list_distance(cat.emb[idxs]) if idxs else 0.0
                ent += genre_entropy([s.book.get("subjects", []) for s in recs])
                shown.update(ids)
                n += 1
        cov = len(shown) / len(cat.books)
        print(f"{lam:>10.1f} {rc / n:>10.3f} {ild / n:>7.3f} {ent / n:>7.2f} {cov:>9.3f}")
    print()


if __name__ == "__main__":
    main()
