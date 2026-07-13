"""Offline evaluation harness for the book recommender.

For every synthetic user we hold out ``k`` of their liked books, build a taste
profile from the *rest* of their signal (remaining likes + dislikes), then rank
the whole catalog (minus books they've already reacted to). A good recommender
ranks the held-out likes near the top. We report Recall@K, NDCG@K and MRR,
averaged over users and over several random hold-out splits.

This is a SCOREBOARD, not a product: it exists so you can compare embedding
models and profile strategies on your own data before building any UI.

Usage
-----
    # Runs today with zero ML deps (numpy baseline):
    python -m eval.run

    # Compare the baseline against real models (needs sentence-transformers):
    python -m eval.run --model hashing \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --model BAAI/bge-m3

    # Sweep profile strategies too:
    python -m eval.run --strategy both
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List

import numpy as np

from pathlib import Path

from .data import book_to_text, load_books, load_profiles
from .embedders import build_embedder
from .metrics import mrr, ndcg_at_k, recall_at_k
from .profiles import build_profile, rank_candidates


def evaluate_model(
    embedder,
    books: List[Dict],
    profiles: List[Dict],
    strategy: str,
    k_holdout: int,
    k_eval: int,
    seeds: List[int],
    text_mode: str = "full",
) -> Dict[str, float]:
    """Return averaged metrics for one embedder + one profile strategy."""
    id_to_idx = {b["id"]: i for i, b in enumerate(books)}
    texts = [book_to_text(b, mode=text_mode) for b in books]
    catalog = embedder.encode(texts)  # (N, D), normalized

    rec, ndcg, rr, n = 0.0, 0.0, 0.0, 0

    for prof in profiles:
        likes = [id_to_idx[bid] for bid in prof["likes"] if bid in id_to_idx]
        dislikes = [id_to_idx[bid] for bid in prof.get("dislikes", []) if bid in id_to_idx]
        if len(likes) <= k_holdout:
            continue  # need at least one liked book left to seed the profile

        for seed in seeds:
            rng = random.Random(seed)
            held_out = set(rng.sample(likes, k_holdout))
            seed_likes = [i for i in likes if i not in held_out]

            profile_vec = build_profile(
                liked_vecs=catalog[seed_likes],
                disliked_vecs=catalog[dislikes] if dislikes else None,
                strategy=strategy,
            )

            # Candidates = everything the user has NOT already reacted to.
            reacted = set(seed_likes) | set(dislikes)
            candidates = [i for i in range(len(books)) if i not in reacted]
            ranked = rank_candidates(profile_vec, catalog, candidates)

            rec += recall_at_k(ranked, held_out, k_eval)
            ndcg += ndcg_at_k(ranked, held_out, k_eval)
            rr += mrr(ranked, held_out)
            n += 1

    if n == 0:
        return {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "trials": 0}
    return {"recall": rec / n, "ndcg": ndcg / n, "mrr": rr / n, "trials": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Embedder spec. 'hashing' for the numpy baseline, or any "
        "sentence-transformers model name. Repeatable. Default: hashing.",
    )
    parser.add_argument(
        "--strategy",
        choices=["mean", "rocchio", "both"],
        default="rocchio",
        help="Profile-building strategy (default: rocchio).",
    )
    parser.add_argument("--k-holdout", type=int, default=2, help="Liked books hidden per trial.")
    parser.add_argument("--k-eval", type=int, default=5, help="Top-K cutoff for metrics.")
    parser.add_argument("--seeds", type=int, default=5, help="Random hold-out splits per user.")
    parser.add_argument("--books", type=Path, default=None, help="Path to a books JSON (default: sample).")
    parser.add_argument("--profiles", type=Path, default=None, help="Path to a profiles JSON (default: sample).")
    parser.add_argument(
        "--text-mode",
        choices=["full", "no-subjects"],
        default="full",
        help="What text to embed. 'no-subjects' drops literal genre words "
        "(diagnostic for lexical keyword leakage).",
    )
    args = parser.parse_args()

    models = args.models or ["hashing"]
    strategies = ["mean", "rocchio"] if args.strategy == "both" else [args.strategy]
    seeds = list(range(args.seeds))

    books = load_books(args.books)
    profiles = load_profiles(args.profiles)

    print(
        f"\nCatalog: {len(books)} books | Users: {len(profiles)} | "
        f"hold-out={args.k_holdout} | eval@{args.k_eval} | splits/user={args.seeds}\n"
    )
    header = f"{'model':<42} {'strategy':<9} {'Recall@K':>9} {'NDCG@K':>8} {'MRR':>7} {'trials':>7}"
    print(header)
    print("-" * len(header))

    for spec in models:
        try:
            embedder = build_embedder(spec)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{spec:<42} !! could not load: {exc}")
            continue
        for strat in strategies:
            m = evaluate_model(
                embedder, books, profiles, strat, args.k_holdout,
                args.k_eval, seeds, text_mode=args.text_mode,
            )
            name = getattr(embedder, "name", spec)
            print(
                f"{name:<42} {strat:<9} {m['recall']:>9.3f} {m['ndcg']:>8.3f} "
                f"{m['mrr']:>7.3f} {m['trials']:>7d}"
            )
    print()


if __name__ == "__main__":
    main()
