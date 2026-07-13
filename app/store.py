"""Storage layer for the serving app.

Two concerns, deliberately separated so the vector/CF store can later move to
Postgres+pgvector without touching the recommender or the swipe log:

* ``Catalog`` -- read-only, in-memory: books + their embeddings, the item-item
  CF matrix, popularity, and metadata filters. Everything is aligned to one
  canonical book order so a book's row is the same index everywhere. Requires
  only numpy (no torch) -- embeddings are precomputed by
  ``scripts/build_embeddings.py``.
* ``SwipeStore`` -- read/write user state (users + swipes) in SQLite.

The pgvector swap replaces Catalog's numpy search with SQL; SwipeStore's
interface is already database-shaped.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from eval.data import load_books

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

REACTIONS = ("like", "dislike", "skip", "interested")  # skip == "haven't read"


@dataclass
class Catalog:
    books: List[dict]
    emb: np.ndarray          # (N, D) L2-normalized
    sim: np.ndarray          # (N, N) item-item CF, 0 diagonal
    pop: np.ndarray          # (N,) rating counts (CF-warmth proxy)
    id_to_idx: Dict[str, int]

    @classmethod
    def load(cls, data_dir: Path = DATA) -> "Catalog":
        books = load_books(data_dir / "real_books.json")
        id_to_idx = {b["id"]: i for i, b in enumerate(books)}

        def aligned(npz_path: Path, key: str) -> np.ndarray:
            npz = np.load(npz_path, allow_pickle=True)
            pos = {str(bid): i for i, bid in enumerate(npz["ids"].tolist())}
            perm = np.array([pos[b["id"]] for b in books])
            arr = npz[key][perm]
            if key == "sim":  # sim is 2-D: reorder columns too
                arr = arr[:, perm]
            return arr

        emb = aligned(data_dir / "real_embeddings.npz", "emb").astype(np.float32)
        sim = aligned(data_dir / "real_cf.npz", "sim").astype(np.float32)
        pop = aligned(data_dir / "real_cf.npz", "pop").astype(np.float32)
        return cls(books, emb, sim, pop, id_to_idx)

    def __len__(self) -> int:
        return len(self.books)

    def idx(self, book_id: str) -> int:
        return self.id_to_idx[book_id]

    def indices(self, book_ids: Sequence[str]) -> List[int]:
        return [self.id_to_idx[b] for b in book_ids if b in self.id_to_idx]

    def filter_mask(
        self,
        languages: Optional[Sequence[str]] = None,
        genres: Optional[Sequence[str]] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> np.ndarray:
        """Boolean mask over the catalog for the given hard filters.

        These are structured-metadata filters applied AROUND vector search --
        never baked into the embedding.
        """
        mask = np.ones(len(self), dtype=bool)
        genre_set = {g.lower() for g in genres} if genres else None
        lang_set = {l.lower() for l in languages} if languages else None
        for i, b in enumerate(self.books):
            if lang_set and (b.get("language", "") or "").lower() not in lang_set:
                mask[i] = False
            if genre_set and not (genre_set & {s.lower() for s in b.get("subjects", [])}):
                mask[i] = False
            yr = b.get("year")
            if year_min is not None and (yr is None or yr < year_min):
                mask[i] = False
            if year_max is not None and (yr is None or yr > year_max):
                mask[i] = False
        return mask

    def all_genres(self) -> List[str]:
        seen: Dict[str, int] = {}
        for b in self.books:
            for s in b.get("subjects", []):
                seen[s] = seen.get(s, 0) + 1
        return [g for g, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


class SwipeStore:
    """User + swipe persistence in SQLite."""

    def __init__(self, db_path: Path = DATA / "app.db", check_same_thread: bool = True):
        # Streamlit shares one cached service across per-session threads, so the
        # server app passes check_same_thread=False and relies on the write lock.
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS swipes (
                user_id TEXT NOT NULL,
                book_id TEXT NOT NULL,
                reaction TEXT NOT NULL,
                ts TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, book_id)
            );
            """
        )
        self.conn.commit()

    def create_user(self, name: str = "") -> str:
        uid = uuid.uuid4().hex[:12]
        with self._lock:
            self.conn.execute("INSERT INTO users (id, name) VALUES (?, ?)", (uid, name))
            self.conn.commit()
        return uid

    def user_exists(self, user_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM users WHERE id=?", (user_id,)
        ).fetchone()
        return row is not None

    def set_name(self, user_id: str, name: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE users SET name=? WHERE id=?", (name, user_id))
            self.conn.commit()

    def user_name(self, user_id: str) -> str:
        row = self.conn.execute(
            "SELECT name FROM users WHERE id=?", (user_id,)
        ).fetchone()
        return (row["name"] if row else "") or ""

    def named_users(self) -> List[dict]:
        """Saved profiles (those the user gave a real name), newest first."""
        rows = self.conn.execute(
            "SELECT id, name FROM users "
            "WHERE name != '' AND name != 'web' ORDER BY created_at DESC"
        ).fetchall()
        return [{"id": r["id"], "name": r["name"]} for r in rows]

    def record(self, user_id: str, book_id: str, reaction: str) -> None:
        if reaction not in REACTIONS:
            raise ValueError(f"reaction must be one of {REACTIONS}, got {reaction!r}")
        # Latest swipe wins (re-swiping updates the reaction).
        with self._lock:
            self.conn.execute(
                "INSERT INTO swipes (user_id, book_id, reaction) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, book_id) DO UPDATE SET reaction=excluded.reaction, "
                "ts=datetime('now')",
                (user_id, book_id, reaction),
            )
            self.conn.commit()

    def reactions(self, user_id: str) -> Dict[str, str]:
        rows = self.conn.execute(
            "SELECT book_id, reaction FROM swipes WHERE user_id=?", (user_id,)
        ).fetchall()
        return {r["book_id"]: r["reaction"] for r in rows}

    def seen(self, user_id: str) -> Set[str]:
        return set(self.reactions(user_id).keys())

    def close(self) -> None:
        self.conn.close()
