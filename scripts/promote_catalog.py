"""Promote a built catalog into the serving directory, migrating user swipes.

Swapping `data/` is not a file copy: `data/app.db` holds real swipe history keyed on
the *old* catalog's book ids, and a straight overwrite orphans every user's taste
profile silently -- they would simply log in to a blank slate. Book ids are per-source
(goodbooks `126` vs Goodreads `gr:5907`), so the swipes have to be re-pointed.

Matching is by normalized title + first author, the same key `hygiene.dedup_key` uses.
Measured on the real data, 251 of 258 swipes (97%) carry over and all three users with
history keep it.

Everything is backed up before anything is written, and the run is a dry run unless
``--apply`` is passed.

    uv run --no-sync python scripts/promote_catalog.py --from data_100k          # preview
    uv run --no-sync python scripts/promote_catalog.py --from data_100k --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hygiene import norm_title  # noqa: E402

# The artifacts that define a catalog. Derived files (catalog.db, emb.f16, ann.idx) are
# deliberately NOT copied -- Catalog.load rebuilds them from these when inputs change,
# and a stale derived file paired with fresh inputs is a silent corruption.
ARTIFACTS = ("real_books.json", "real_embeddings.npz", "real_cf.npz", "real_profiles.json")
DERIVED = ("catalog.db", "emb.f16", "ann.idx")


def _key(book: dict) -> tuple[str, str]:
    return (
        norm_title(book.get("title", "")),
        norm_title((book.get("author", "") or "").split(",")[0]),
    )


def plan_swipe_migration(old_dir: Path, new_dir: Path) -> tuple[dict[str, str], dict]:
    """Return (old_book_id -> new_book_id, stats). Empty mapping if there is no app.db."""
    db = old_dir / "app.db"
    if not db.exists():
        return {}, {"swipes": 0, "note": "no app.db"}

    old_books = json.loads((old_dir / "real_books.json").read_text(encoding="utf-8"))
    new_books = json.loads((new_dir / "real_books.json").read_text(encoding="utf-8"))
    old_by_id = {b["id"]: b for b in old_books}
    new_by_key: dict[tuple[str, str], str] = {}
    for b in new_books:  # first wins: new_books is ordered most-rated first
        new_by_key.setdefault(_key(b), b["id"])

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT user_id, book_id FROM swipes").fetchall()
    conn.close()

    mapping: dict[str, str] = {}
    kept_users: set[str] = set()
    lost = 0
    for user_id, bid in rows:
        book = old_by_id.get(bid)
        target = new_by_key.get(_key(book)) if book else None
        if target is None:
            lost += 1
        else:
            mapping[bid] = target
            kept_users.add(user_id)
    stats = {
        "swipes": len(rows),
        "migrated": len(rows) - lost,
        "dropped": lost,
        "users_with_history": len({u for u, _ in rows}),
        "users_retaining": len(kept_users),
    }
    return mapping, stats


def apply_migration(old_dir: Path, new_dir: Path, mapping: dict[str, str], backup: Path) -> None:
    """Back up, re-point swipes, then swap the catalog artifacts in place."""
    backup.mkdir(parents=True, exist_ok=True)
    for f in (*ARTIFACTS, *DERIVED, "app.db"):
        src = old_dir / f
        if src.exists():
            shutil.copy2(src, backup / f)
    print(f"  backed up {old_dir} -> {backup}")

    db = old_dir / "app.db"
    if db.exists() and mapping:
        conn = sqlite3.connect(db)
        # Swipes whose book has no counterpart are deleted, not left dangling: a
        # dangling id is silently skipped by _split() and would look like signal loss
        # with no explanation.
        conn.execute("CREATE TEMP TABLE idmap (old TEXT PRIMARY KEY, new TEXT)")
        conn.executemany("INSERT INTO idmap VALUES (?,?)", mapping.items())
        conn.execute("DELETE FROM swipes WHERE book_id NOT IN (SELECT old FROM idmap)")
        conn.execute(
            "UPDATE swipes SET book_id = (SELECT new FROM idmap WHERE old = swipes.book_id)"
        )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM swipes").fetchone()[0]
        conn.close()
        print(f"  re-pointed swipes -> {n} rows now reference the new catalog")

    for f in DERIVED:  # stale derived artifacts must not survive the swap
        (old_dir / f).unlink(missing_ok=True)
    for f in ARTIFACTS:
        src = new_dir / f
        if src.exists():
            shutil.copy2(src, old_dir / f)
            print(f"  installed {f} ({src.stat().st_size / 1e6:.0f} MB)")
        else:
            print(f"  WARNING: {new_dir / f} missing, left the old one in place")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", type=Path, required=True, help="catalog dir to promote")
    ap.add_argument("--to", dest="dst", type=Path, default=ROOT / "data", help="serving dir")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    for d in (src, dst):
        if not (d / "real_books.json").exists():
            raise SystemExit(f"{d} does not look like a catalog (no real_books.json)")

    n_old = len(json.loads((dst / "real_books.json").read_text(encoding="utf-8")))
    n_new = len(json.loads((src / "real_books.json").read_text(encoding="utf-8")))
    print(f"promote {src} ({n_new:,} books) -> {dst} ({n_old:,} books)\n")

    mapping, stats = plan_swipe_migration(dst, src)
    print("swipe migration:")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = dst.parent / f"{dst.name}_backup_{stamp}"
    print("\napplying:")
    apply_migration(dst, src, mapping, backup)
    print(f"\nDone. Previous catalog preserved at {backup}")
    print("Derived artifacts were removed; the next Catalog.load rebuilds them.")


if __name__ == "__main__":
    main()
