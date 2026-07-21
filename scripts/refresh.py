"""Periodic refresh job -- keep the serving catalog fresh over time.

Two things drift after the initial build:

1. **New books** appear. Pass ``--add PATH`` to ingest a JSON list of new book
   records; this delegates to :func:`add_books.add_books` (embeds the new books,
   appends them CF-cold).

2. **Signal accumulates.** The initial CF matrix only knows the original
   goodbooks ratings. Every day the app runs, users leave likes / 'interested' /
   dislikes in ``data/app.db``. This job rebuilds the item-item CF matrix from
   **all** of that signal -- goodbooks ratings PLUS the app's own swipes -- so
   books people actually engage with gain collaborative warmth, and a new book
   that accrues swipes stops being CF-cold.

Swipes become pseudo-ratings on goodbooks' 1-5 scale:

    like -> 5    interested -> 4    dislike -> 2    skip -> ignored

The offline-eval users (``data/real_profiles.json``) are kept OUT of the CF
training set, exactly as ``scripts/build_real_dataset.py`` does, so the eval
harness stays honest after a refresh.

Run (in the uv env):
    uv run --no-sync python scripts/refresh.py                  # rebuild CF from swipes
    uv run --no-sync python scripts/refresh.py --add new.json   # ingest, then rebuild
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # sibling-script imports

from add_books import add_books  # noqa: E402
from build_real_dataset import GB_BASE, fetch, load_csv  # noqa: E402
from fetch_new_books import fetch_new_books  # noqa: E402

DATA = ROOT / "data"

# Swipe reaction -> pseudo-rating on goodbooks' 1-5 scale ('skip' contributes none).
SWIPE_RATING = {"like": 5.0, "interested": 4.0, "dislike": 2.0}


def _catalog_order() -> list[str]:
    from app.store import catalog_records  # base real_books.json + append-only sidecar

    return [b["id"] for b in catalog_records(DATA)]


def _eval_user_ids() -> set:
    """goodbooks user ids reserved for offline eval -- excluded from CF training."""
    path = DATA / "real_profiles.json"
    if not path.exists():
        return set()
    profiles = json.loads(path.read_text(encoding="utf-8"))
    return {p["user"].removeprefix("gr_") for p in profiles}


def _goodbooks_ratings(order: set) -> dict[str, dict[str, int]]:
    """Per-user {book_id: rating} from goodbooks, restricted to catalog books."""
    text = fetch(GB_BASE + "ratings.csv", "gb_ratings.csv")
    by_user: dict[str, dict[str, int]] = defaultdict(dict)
    for row in load_csv(text):
        bid = row["book_id"]
        if bid in order:
            by_user[row["user_id"]][bid] = int(row["rating"])
    return by_user


def _app_ratings(order: set) -> dict[str, dict[str, float]]:
    """Per-user pseudo-ratings from the app's swipe log (app.db)."""
    db = DATA / "app.db"
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT user_id, book_id, reaction FROM swipes").fetchall()
    finally:
        conn.close()
    by_user: dict[str, dict[str, float]] = defaultdict(dict)
    for user_id, book_id, reaction in rows:
        if book_id in order and reaction in SWIPE_RATING:
            by_user[user_id][book_id] = SWIPE_RATING[reaction]
    return by_user


def rebuild_cf(data_dir: Path = DATA, max_items: int | None = None) -> None:
    """Retrain EASE over goodbooks non-eval users + app swipes.

    ``max_items`` defaults to ``cf_build.EASE_MAX_ITEMS`` rather than a local constant:
    it decides the size of a dense inverse (~16·h² bytes at peak), so a stale duplicate
    here is an OOM waiting for the catalog to grow into it.
    """
    from cf_build import EASE_MAX_ITEMS, ease_cf

    from app.store import save_cf

    max_items = EASE_MAX_ITEMS if max_items is None else max_items

    # Guard: this path learns CF from the *goodbooks* ratings dump, whose ids only match
    # a goodbooks-derived catalog. Against a Goodreads ingest (``gr:`` ids) nothing joins,
    # and the rebuild would quietly replace a 5M-interaction CF matrix with one learned
    # from a few hundred app swipes -- a catastrophic downgrade that looks like success.
    # Large catalogs rebuild CF with scripts/rebuild_cf.py instead, off the cached
    # interaction matrix.
    order_probe = _catalog_order()
    if order_probe and not any(b[:1].isdigit() for b in order_probe[:100]):
        raise SystemExit(
            "refresh --rebuild-cf only understands goodbooks ids, but this catalog uses "
            f"ids like {order_probe[0]!r}. Use:\n"
            "  uv run --no-sync python scripts/rebuild_cf.py --data data --method hybrid\n"
            "(or pass --no-cf to skip the CF rebuild)."
        )

    order = _catalog_order()
    order_set = set(order)

    gb = _goodbooks_ratings(order_set)
    eval_uids = _eval_user_ids()
    app = _app_ratings(order_set)

    # Combine: goodbooks non-eval users + all app users. Namespaced keys so a
    # goodbooks id and an app id can never collide.
    combined: dict[str, dict[str, float]] = {}
    for uid, ratings in gb.items():
        if uid not in eval_uids:
            combined[f"gb:{uid}"] = ratings
    for uid, ratings in app.items():
        combined[f"app:{uid}"] = dict(ratings)

    sim, pop = ease_cf(order, combined, max_items=max_items)
    save_cf(data_dir / "real_cf.npz", order, sim, pop)
    n_app = sum(1 for k in combined if k.startswith("app:"))
    print(
        f"Rebuilt EASE-R CF ({sim.shape[0]}x{sim.shape[1]}, {sim.nnz} nnz) from "
        f"{len(combined)} users ({n_app} from app swipes) -> {data_dir / 'real_cf.npz'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh the serving catalog.")
    ap.add_argument("--add", metavar="PATH", help="JSON list of new book records to ingest first.")
    ap.add_argument(
        "--fetch-new",
        type=int,
        metavar="N",
        help="Pull N new books from Open Library and ingest them.",
    )
    ap.add_argument(
        "--no-cf", action="store_true", help="Skip the CF rebuild (only ingest new books)."
    )
    ap.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Dense-EASE budget: solve item-item CF over the this-many most-rated books "
        "(the rest fall to content). Peak memory is ~16*h^2 bytes, so this is what "
        "decides whether the rebuild finishes; defaults to cf_build.EASE_MAX_ITEMS "
        "(20k ~ 6.4 GB). Raise it only on a box with the RAM to back it.",
    )
    args = ap.parse_args()

    if args.add:
        add_books(json.loads(Path(args.add).read_text(encoding="utf-8")))
    if args.fetch_new:
        print(f"Fetching up to {args.fetch_new} new books from Open Library...")
        records = fetch_new_books(want=args.fetch_new)
        print(f"  fetched {len(records)}; ingesting...")
        add_books(records)
    if not args.no_cf:
        rebuild_cf(max_items=args.max_items)


if __name__ == "__main__":
    main()
