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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from eval.data import load_books

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

REACTIONS = ("like", "dislike", "skip", "interested")  # skip == "haven't read"


def save_cf(path: Path, ids: Sequence[str], sim: sparse.csr_matrix, pop: np.ndarray) -> None:
    """Persist a sparse top-k CF matrix + popularity, keyed by book id.

    Stored as flat CSR components (data/indices/indptr/shape) so the file needs
    only numpy+scipy to load. Written atomically.
    """
    sim = sim.tocsr()
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        ids=np.array(list(ids), dtype=str),
        pop=pop.astype(np.float32),
        sim_data=sim.data.astype(np.float32),
        sim_indices=sim.indices.astype(np.int32),
        sim_indptr=sim.indptr.astype(np.int64),
        sim_shape=np.array(sim.shape, dtype=np.int64),
    )
    import os

    os.replace(tmp, path)


def load_cf(path: Path):
    """Return (ids: list[str], sim: csr_matrix, pop: np.ndarray)."""
    with np.load(path, allow_pickle=True) as z:
        ids = [str(b) for b in z["ids"].tolist()]
        pop = z["pop"].astype(np.float32)
        sim = sparse.csr_matrix(
            (z["sim_data"], z["sim_indices"], z["sim_indptr"]),
            shape=tuple(z["sim_shape"]),
        )
    return ids, sim, pop


@dataclass
class Catalog:
    books: list[dict]
    emb: np.ndarray  # (N, D) L2-normalized
    sim: sparse.csr_matrix  # (N, N) item-item CF (sparse top-k), 0 diagonal
    pop: np.ndarray  # (N,) rating counts (CF-warmth proxy)
    id_to_idx: dict[str, int]
    # Precomputed columnar filter indices (built once), so filter_mask is vectorized
    # numpy instead of a per-book Python loop on every recommendation.
    _lang: np.ndarray = field(init=False, repr=False)
    _year: np.ndarray = field(init=False, repr=False)
    _genre_mask: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._lang = np.array([(b.get("language") or "").lower() for b in self.books], dtype=object)
        self._year = np.array(
            [b["year"] if b.get("year") is not None else np.nan for b in self.books],
            dtype=np.float64,
        )
        gm: dict[str, np.ndarray] = {}
        for i, b in enumerate(self.books):
            for s in b.get("subjects", []) or []:
                arr = gm.get(s.lower())
                if arr is None:
                    arr = gm[s.lower()] = np.zeros(len(self.books), dtype=bool)
                arr[i] = True
        self._genre_mask = gm

    @classmethod
    def load(cls, data_dir: Path = DATA) -> Catalog:
        books = load_books(data_dir / "real_books.json")
        id_to_idx = {b["id"]: i for i, b in enumerate(books)}

        def perm_for(ids: Sequence[str]) -> np.ndarray:
            pos = {str(b): i for i, b in enumerate(ids)}
            return np.array([pos[b["id"]] for b in books])

        # Embeddings are stored fp16 (half the file + load), but kept fp32 in RAM:
        # numpy has no fp16 GEMV on CPU, so an fp16-resident matrix would upcast on
        # every query (~9x slower). fp16 ranking is accuracy-neutral -- the query-
        # bandwidth win lands with an fp16-native index (FAISS/pgvector) at scale.
        emb_npz = np.load(data_dir / "real_embeddings.npz", allow_pickle=True)
        emb = emb_npz["emb"][perm_for(emb_npz["ids"].tolist())].astype(np.float32)

        cf_ids, cf_sim, cf_pop = load_cf(data_dir / "real_cf.npz")
        p = perm_for(cf_ids)
        sim = cf_sim[p][:, p].tocsr()  # reorder rows AND columns to catalog order
        pop = cf_pop[p]
        return cls(books, emb, sim, pop, id_to_idx)

    def __len__(self) -> int:
        return len(self.books)

    def idx(self, book_id: str) -> int:
        return self.id_to_idx[book_id]

    def indices(self, book_ids: Sequence[str]) -> list[int]:
        return [self.id_to_idx[b] for b in book_ids if b in self.id_to_idx]

    def filter_mask(
        self,
        languages: Sequence[str] | None = None,
        genres: Sequence[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> np.ndarray:
        """Boolean mask over the catalog for the given hard filters.

        These are structured-metadata filters applied AROUND vector search --
        never baked into the embedding. Vectorized via the precomputed columnar
        indices, so the loops here are over the (small) requested filter values,
        never over the catalog.
        """
        mask = np.ones(len(self), dtype=bool)
        if languages:
            mask &= np.isin(self._lang, [lang.lower() for lang in languages])
        if genres:
            gmask = np.zeros(len(self), dtype=bool)
            for g in genres:
                arr = self._genre_mask.get(g.lower())
                if arr is not None:
                    gmask |= arr
            mask &= gmask
        if year_min is not None:  # NaN (missing year) compares False -> excluded, as before
            mask &= self._year >= year_min
        if year_max is not None:
            mask &= self._year <= year_max
        return mask

    def all_genres(self) -> list[str]:
        seen: dict[str, int] = {}
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
        row = self.conn.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone()
        return row is not None

    def set_name(self, user_id: str, name: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE users SET name=? WHERE id=?", (name, user_id))
            self.conn.commit()

    def user_name(self, user_id: str) -> str:
        row = self.conn.execute("SELECT name FROM users WHERE id=?", (user_id,)).fetchone()
        return (row["name"] if row else "") or ""

    def named_users(self) -> list[dict]:
        """Saved profiles (those the user gave a real name), newest first."""
        rows = self.conn.execute(
            "SELECT id, name FROM users WHERE name != '' AND name != 'web' ORDER BY created_at DESC"
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

    def reactions(self, user_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT book_id, reaction FROM swipes WHERE user_id=?", (user_id,)
        ).fetchall()
        return {r["book_id"]: r["reaction"] for r in rows}

    def seen(self, user_id: str) -> set[str]:
        return set(self.reactions(user_id).keys())

    def close(self) -> None:
        self.conn.close()
