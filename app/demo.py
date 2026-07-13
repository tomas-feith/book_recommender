"""End-to-end demo of the 'Tinder for books' loop on real data.

Runs a scripted session so it works non-interactively:
  1. create a user
  2. seed ~8 liked books by typing (fuzzy) titles
  3. show initial recommendations
  4. run swipe rounds, auto-reacting from a hidden simulated taste
  5. show how recommendations concentrate as the profile sharpens
  6. demonstrate a genre filter

Run:  python -m app.demo         (needs only numpy — embeddings are precomputed)
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

from .recommender import Scored
from .service import BookRecommenderService

# A hidden taste we use to auto-swipe, to prove the profile adapts.
LIKES = {"fantasy", "dystopian", "young-adult", "science-fiction", "dystopia"}
DISLIKES = {"romance", "chick-lit"}

SEED_QUERIES = [
    "the hunger games", "harry potter philosophers stone", "divergent",
    "the maze runner", "ender's game", "the giver", "fahrenheit 451",
    "the hobbit",
]


def simulated_reaction(book: dict) -> str:
    subs = {s.lower() for s in book.get("subjects", [])}
    if subs & LIKES:
        return "like"
    if subs & DISLIKES:
        return "dislike"
    return "skip"


def show(cards, label: str) -> None:
    print(f"\n{label}")
    for c in cards:
        b = c.book
        genres = ", ".join(b.get("subjects", [])[:3])
        driver = "CF" if c.cf_weight >= 0.5 else "content"
        print(f"  {b['title'][:42]:<42} | {b.get('author','')[:20]:<20} "
              f"| {genres:<30} | via {driver} (w_cf={c.cf_weight:.2f})")


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "demo.db"
    svc = BookRecommenderService(db_path=tmp)
    rng = random.Random(7)
    print(f"Catalog loaded: {len(svc.catalog)} books.")

    uid = svc.new_user("demo")

    # --- onboarding: type some titles -----------------------------------
    seed = svc.seed(uid, SEED_QUERIES)
    print("\nSeed titles resolved:")
    for m in seed.resolved:
        print(f"  '{m.title}' by {m.author}  (match {m.score})")
    if seed.unresolved:
        print("  unresolved:", seed.unresolved)

    show(svc.recommendations(uid, n=8), "INITIAL 'For You' (from seed likes):")

    # --- swipe rounds ----------------------------------------------------
    for rnd in range(1, 4):
        cards = svc.next_cards(uid, n=8, rng=rng)
        tally = {"like": 0, "dislike": 0, "skip": 0}
        for c in cards:
            r = simulated_reaction(c.book)
            svc.swipe(uid, c.book["id"], r)
            tally[r] += 1
        print(f"\nSwipe round {rnd}: shown {len(cards)} -> "
              f"{tally['like']} likes, {tally['dislike']} dislikes, {tally['skip']} skips")

    print("\nProfile now:", svc.profile_summary(uid))
    show(svc.recommendations(uid, n=8), "'For You' AFTER swiping (should skew to liked genres):")

    # --- a hard filter ---------------------------------------------------
    show(svc.recommendations(uid, n=5, genres=["fantasy"], year_min=2000),
         "Filtered: genre=fantasy, year>=2000:")

    svc.close()
    os.remove(tmp)


if __name__ == "__main__":
    main()
