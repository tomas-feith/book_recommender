"""Relevance vs diversity vs genre-calibration for the 'For You' list.

Runs the *actual served* recommender over the real profiles at several
(mmr_lambda, cal_lambda) settings and reports, per config: Recall@10 (relevance),
intra-list distance + genre entropy + catalog coverage (diversity), and
**miscalibration KL** (how far the list's genre mix is from the user's taste mix;
lower = better calibrated). Turns the diversity/calibration knobs into evidence.

Run (needs the real dataset built):
    uv run --no-sync python -m eval.diversity
"""

from __future__ import annotations

import random
from pathlib import Path

from app.recommender import Recommender, genre_distribution, kl_calibration
from app.store import Catalog
from eval.data import load_profiles
from eval.metrics import genre_entropy, intra_list_distance

DATA = Path(__file__).resolve().parent.parent / "data"
K_HOLDOUT = 3
K_EVAL = 10
SEEDS = list(range(5))
# (mmr_lambda, cal_lambda): pure relevance, MMR-only, then MMR + rising calibration.
CONFIGS = [(1.0, 0.0), (0.5, 0.0), (0.5, 0.4), (0.5, 0.8)]


def main() -> None:
    cat = Catalog.load(DATA)
    rec = Recommender(cat)
    profiles = load_profiles(DATA / "real_profiles.json")

    print(
        f"\nFor-You frontier | {len(cat.books)} books | {len(profiles)} users | "
        f"hold-out={K_HOLDOUT} | list={K_EVAL}\n"
    )
    header = (
        f"{'mmr':>4} {'cal':>4} {'Recall@10':>10} {'ILD':>6} "
        f"{'genreH':>7} {'miscalKL':>9} {'coverage':>9}"
    )
    print(header)
    print("-" * len(header))

    for mmr_lambda, cal_lambda in CONFIGS:
        rc = ild = ent = kl = 0.0
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
                recs = rec.recommend(
                    reactions, filters={}, n=K_EVAL, mmr_lambda=mmr_lambda, cal_lambda=cal_lambda
                )
                ids = [s.book["id"] for s in recs]
                idxs = [cat.idx(i) for i in ids]

                target = genre_distribution(
                    [cat.books[cat.idx(b)].get("subjects", []) for b in seed_likes]
                )
                list_dist = genre_distribution([s.book.get("subjects", []) for s in recs])

                rc += len(held & set(ids)) / min(len(held), K_EVAL)
                ild += intra_list_distance(cat.emb[idxs]) if idxs else 0.0
                ent += genre_entropy([s.book.get("subjects", []) for s in recs])
                kl += kl_calibration(target, list_dist) if target else 0.0
                shown.update(ids)
                n += 1
        cov = len(shown) / len(cat.books)
        print(
            f"{mmr_lambda:>4.1f} {cal_lambda:>4.1f} {rc / n:>10.3f} {ild / n:>6.3f} "
            f"{ent / n:>7.2f} {kl / n:>9.3f} {cov:>9.3f}"
        )
    print()


if __name__ == "__main__":
    main()
