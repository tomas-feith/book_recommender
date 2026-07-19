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

import json
import os
import re
import sqlite3
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from eval.data import load_books

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

REACTIONS = ("like", "dislike", "skip", "interested")  # skip == "haven't read"

# Serving metadata store. The catalog no longer lives in RAM as one big list of
# dicts parsed from real_books.json (~800 MB / multi-GB heap at 1M books). Instead:
#   * real_books.json      -- the base catalog, still written by the bulk builders.
#   * real_books_added.jsonl -- append-only sidecar for incrementally-added books,
#     so an add never rewrites the base file.
#   * catalog.db           -- a SQLite serving cache built from (base + sidecar),
#     giving columnar filters + lazy per-row records so descriptions/images stay
#     off the Python heap. Rebuilt only when an input changes (mtime), not per boot.
CATALOG_DB = "catalog.db"
SIDECAR = "real_books_added.jsonl"
# Columns mirrored out of the record for fast columnar reads (the JSON blob keeps
# the full-fidelity record for lazy fetch).
_COLS = ("id", "title", "author", "subjects", "language", "year")


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


def catalog_records(data_dir: Path = DATA) -> list[dict]:
    """The full base+sidecar catalog as records, in canonical order.

    Base ``real_books.json`` first, then any append-only ``real_books_added.jsonl``
    rows. This is the offline "whole catalog" view -- used to (re)build the serving
    DB and by the batch tools (refresh / add_books / build_embeddings) so
    incrementally-added books are not dropped on a rebuild.
    """
    base_path = data_dir / "real_books.json"
    records = load_books(base_path) if base_path.exists() else []
    side = data_dir / SIDECAR
    if side.exists():
        for line in side.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def append_book_to_sidecar(book: dict, data_dir: Path = DATA) -> None:
    """Append one record to the sidecar (durable, O(1) -- never rewrites the base)."""
    with (data_dir / SIDECAR).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(book, ensure_ascii=False) + "\n")


def _db_row(book: dict) -> tuple:
    """(id, title, author, subjects_json, language, year, data_json) for one record."""
    return (
        str(book["id"]),
        book.get("title", ""),
        book.get("author", "") or "",
        json.dumps(book.get("subjects", []) or []),
        (book.get("language") or ""),
        book.get("year"),
        json.dumps(book, ensure_ascii=False),
    )


def _newest_input_mtime(data_dir: Path) -> float:
    paths = [data_dir / "real_books.json", data_dir / SIDECAR, data_dir / "real_embeddings.npz"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def build_catalog_db(
    db_path: Path, records: Sequence[dict], keep_ids: set[str] | None = None
) -> None:
    """(Re)build the serving DB from records, keyed by a contiguous catalog index.

    ``keep_ids`` (when given) filters to books that also have an embedding + CF row,
    so the DB rows stay 1:1 and in-order with the emb/CF matrices -- no subset drift.
    Written to a temp file and swapped in atomically.
    """
    tmp = db_path.with_name(db_path.stem + ".tmp.db")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(str(tmp))
    try:
        conn.execute(
            "CREATE TABLE books (idx INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, "
            "title TEXT, author TEXT, subjects TEXT, language TEXT, year INTEGER, data TEXT)"
        )
        rows = (_db_row(b) for b in records if keep_ids is None or str(b["id"]) in keep_ids)
        conn.executemany(
            "INSERT OR IGNORE INTO books (id, title, author, subjects, language, year, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        # Trigram FTS over title/original-title/author for sublinear onboarding search
        # (replaces the O(N) SequenceMatcher scan). rowid == books.idx, so a match maps
        # straight back to a catalog row.
        conn.execute(
            "CREATE VIRTUAL TABLE books_fts USING fts5("
            "title, orig_title, author, tokenize='trigram')"
        )
        conn.execute(
            "INSERT INTO books_fts(rowid, title, orig_title, author) "
            "SELECT idx, title, COALESCE(json_extract(data, '$.orig_title'), ''), author FROM books"
        )
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, db_path)


def _has_fts(db_path: Path) -> bool:
    """Whether an existing DB carries the FTS table (else it predates it -> rebuild)."""
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='books_fts'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


class BookTable:
    """Lazy, SQLite-backed sequence of book records in catalog order.

    ``table[i]`` fetches row ``i``'s full record on demand (small LRU cache);
    iteration streams all rows; ``append`` INSERTs. Keeps descriptions/images off
    the Python heap -- only the columnar arrays the recommender needs stay resident
    (see ``Catalog``). One shared connection guarded by a lock, matching
    ``SwipeStore``'s threading model.
    """

    def __init__(self, db_path: Path, check_same_thread: bool = True, cache_cap: int = 8192):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
        self._n = self.conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cap = cache_cap
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._n

    def ids(self) -> list[str]:
        """Book ids in catalog (idx) order."""
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT id FROM books ORDER BY idx")]

    def columns(self) -> tuple[np.ndarray, list[list[str]], np.ndarray, np.ndarray]:
        """(authors, subjects, languages, years) as columnar arrays, one DB scan.

        Reads only the small columns -- never the description blob -- so building the
        resident filter/selection arrays at boot is cheap even at 1M rows.
        """
        authors: list[str] = []
        subjects: list[list[str]] = []
        langs: list[str] = []
        years: list[float] = []
        with self._lock:
            cur = self.conn.execute(
                "SELECT author, subjects, language, year FROM books ORDER BY idx"
            )
            for author, subs, lang, year in cur:
                authors.append(author or "")
                subjects.append(json.loads(subs) if subs else [])
                langs.append((lang or "").lower())
                years.append(year if year is not None else np.nan)
        return (
            np.array(authors, dtype=object),
            subjects,
            np.array(langs, dtype=object),
            np.array(years, dtype=np.float64),
        )

    def __getitem__(self, i: int) -> dict:
        i = int(i)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        with self._lock:
            hit = self._cache.get(i)
            if hit is not None:
                self._cache.move_to_end(i)
                return hit
            row = self.conn.execute("SELECT data FROM books WHERE idx=?", (i + 1,)).fetchone()
            rec = json.loads(row[0])
            self._cache[i] = rec
            if len(self._cache) > self._cap:
                self._cache.popitem(last=False)
            return rec

    def __iter__(self) -> Iterator[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT data FROM books ORDER BY idx").fetchall()
        for (data,) in rows:
            yield json.loads(data)

    def append(self, book: dict) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO books (id, title, author, subjects, language, year, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                _db_row(book),
            )
            self.conn.execute(
                "INSERT INTO books_fts(rowid, title, orig_title, author) VALUES (?, ?, ?, ?)",
                (
                    cur.lastrowid,
                    book.get("title", ""),
                    book.get("orig_title", "") or "",
                    book.get("author", "") or "",
                ),
            )
            self.conn.commit()
            i = self._n
            self._n += 1
            return i

    def search_fts(self, query: str, limit: int = 200) -> list[int]:
        """Candidate positions (0-based) trigram-matching ``query`` in title/author.

        Recall-oriented: any query word (>=3 chars, trigram-searchable) OR-matched,
        best bm25 first. The precise ranking is left to the reranker in ``TitleIndex``.
        Short/CJK queries fall back to a bounded substring scan.
        """
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 3]
        with self._lock:
            if words:
                expr = " OR ".join(f'"{w}"' for w in words)
                rows = self.conn.execute(
                    "SELECT rowid FROM books_fts WHERE books_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expr, limit),
                ).fetchall()
            else:
                like = f"%{query.strip()}%"
                rows = self.conn.execute(
                    "SELECT idx FROM books WHERE title LIKE ? OR author LIKE ? LIMIT ?",
                    (like, like, limit),
                ).fetchall()
        return [r[0] - 1 for r in rows]  # rowid/idx is 1-based -> 0-based position

    def close(self) -> None:
        self.conn.close()


def append_to_catalog_files(to_add: list[dict], new_emb: np.ndarray, data_dir: Path = DATA) -> None:
    """Persist already-embedded new books, append-only (no base-file rewrite).

    ``new_emb`` is the (k, D) embedding matrix for ``to_add``. New books get pop=0
    and empty CF rows/cols. Records go to the append-only sidecar; embeddings and
    CF are grown in their npz files. Callers dedup first. Used by
    ``scripts/add_books.py`` (batch) and the live service (on-demand adds) so a book
    survives a restart; the serving DB is (re)built from base+sidecar on next load.
    """
    emb_path, cf_path = data_dir / "real_embeddings.npz", data_dir / "real_cf.npz"
    with np.load(emb_path, allow_pickle=True) as z:  # close handle before writing (Windows lock)
        old_emb, old_ids, model = z["emb"], z["ids"].astype(str), np.array(z["model"])
    old_cf_ids, sim, pop = load_cf(cf_path)

    new_ids = [b["id"] for b in to_add]
    k = len(to_add)
    emb = np.vstack([old_emb, new_emb]).astype(np.float16)  # keep fp16 storage
    emb_ids = np.concatenate([old_ids, np.array(new_ids, dtype=str)])
    grown = sparse.block_diag([sim, sparse.csr_matrix((k, k))], format="csr")
    new_pop = np.concatenate([pop, np.zeros(k, dtype=np.float32)]).astype(np.float32)

    tmp = emb_path.with_name(emb_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, ids=emb_ids, emb=emb, model=model)
    os.replace(tmp, emb_path)
    save_cf(cf_path, list(old_cf_ids) + new_ids, grown, new_pop)
    for b in to_add:
        append_book_to_sidecar(b, data_dir)


@dataclass
class Catalog:
    # ``books`` is a sequence of records -- a plain list in tests, or a lazy
    # SQLite-backed ``BookTable`` in production (``load``), so descriptions/images
    # stay off the Python heap. Only the columnar arrays below are resident.
    books: list[dict] | BookTable
    emb: np.ndarray  # (N, D) L2-normalized
    sim: sparse.csr_matrix  # (N, N) item-item CF (sparse top-k), 0 diagonal
    pop: np.ndarray  # (N,) rating counts (CF-warmth proxy)
    id_to_idx: dict[str, int]
    # Precomputed columnar arrays (built once), so filters + hot selection loops
    # (author cap, genre calibration) are resident numpy/lists, never a per-book
    # fetch on every recommendation.
    _lang: np.ndarray = field(init=False, repr=False)
    _year: np.ndarray = field(init=False, repr=False)
    _authors: np.ndarray = field(init=False, repr=False)
    _subjects: list[list[str]] = field(init=False, repr=False, default_factory=list)
    # Inverted index: subject -> int32 array of the book rows carrying it. A dense
    # {subject: bool[N]} mask costs G*N bytes -- ~200 MB at 22k books, tens-to-hundreds
    # of GB at 1M (G grows with the catalog). The index is O(total tags) instead
    # (~0.4 MB at 22k) and scatters into a mask in O(hits), not O(N)-per-genre.
    _genre_idx: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.books, BookTable):  # one columnar scan, no description blobs
            authors, subjects, langs, years = self.books.columns()
        else:  # in-memory list (tests): derive the same arrays from the records
            authors = np.array([b.get("author", "") or "" for b in self.books], dtype=object)
            subjects = [list(b.get("subjects") or []) for b in self.books]
            langs = np.array([(b.get("language") or "").lower() for b in self.books], dtype=object)
            years = np.array(
                [b["year"] if b.get("year") is not None else np.nan for b in self.books],
                dtype=np.float64,
            )
        self._authors, self._subjects, self._lang, self._year = authors, subjects, langs, years
        rows_by_genre: dict[str, list[int]] = {}
        for i, subs in enumerate(subjects):
            for s in subs:
                rows_by_genre.setdefault(s.lower(), []).append(i)
        self._genre_idx = {s: np.array(rows, dtype=np.int32) for s, rows in rows_by_genre.items()}

    @classmethod
    def load(cls, data_dir: Path = DATA, check_same_thread: bool = True) -> Catalog:
        # Embeddings are stored fp16 (half the file + load), but kept fp32 in RAM:
        # numpy has no fp16 GEMV on CPU, so an fp16-resident matrix would upcast on
        # every query (~9x slower). fp16 ranking is accuracy-neutral -- the query-
        # bandwidth win lands with an fp16-native index (FAISS/pgvector) at scale.
        with np.load(data_dir / "real_embeddings.npz", allow_pickle=True) as emb_npz:
            emb_ids = [str(x) for x in emb_npz["ids"].tolist()]
            cf_ids, cf_sim, cf_pop = load_cf(data_dir / "real_cf.npz")
            keep = set(emb_ids) & set(cf_ids)  # only books with both an embedding and a CF row

            # (Re)build the serving DB from base+sidecar only when an input changed,
            # keyed to `keep` so DB rows stay 1:1 and in-order with emb/CF.
            db_path = data_dir / CATALOG_DB
            if (
                not db_path.exists()
                or db_path.stat().st_mtime < _newest_input_mtime(data_dir)
                or not _has_fts(db_path)  # schema upgrade: older DBs predate the FTS table
            ):
                build_catalog_db(db_path, catalog_records(data_dir), keep_ids=keep)
            table = BookTable(db_path, check_same_thread=check_same_thread)

            order = table.ids()
            id_to_idx = {b: i for i, b in enumerate(order)}
            emb_pos = {b: i for i, b in enumerate(emb_ids)}
            emb = emb_npz["emb"][[emb_pos[b] for b in order]].astype(np.float32)

        cf_pos = {b: i for i, b in enumerate(cf_ids)}
        p = np.array([cf_pos[b] for b in order])
        sim = cf_sim[p][:, p].tocsr()  # reorder rows AND columns to catalog order
        pop = cf_pop[p]
        return cls(table, emb, sim, pop, id_to_idx)

    def __len__(self) -> int:
        return len(self.books)

    def idx(self, book_id: str) -> int:
        return self.id_to_idx[book_id]

    def indices(self, book_ids: Sequence[str]) -> list[int]:
        return [self.id_to_idx[b] for b in book_ids if b in self.id_to_idx]

    # Resident-array accessors for the hot selection loops, so scoring never
    # round-trips a full record out of SQLite.
    def author(self, i: int) -> str:
        return self._authors[i]

    def subjects(self, i: int) -> list[str]:
        return self._subjects[i]

    def languages(self) -> list[str]:
        """Distinct languages present (for the filter UI); missing -> 'en'."""
        return sorted({(lang or "en") for lang in self._lang})

    def years(self) -> list[int]:
        """Sorted publication years present (for the filter UI)."""
        return sorted(int(y) for y in self._year if not np.isnan(y))

    def append(self, book: dict, emb_vec: np.ndarray) -> int:
        """Append one CF-cold book to the in-memory catalog; returns its index.

        The book gets pop=0 and an empty CF row/col (content-ranked until it accrues
        reactions). Filter/selection arrays are grown incrementally (no O(N) rebuild).
        Persist separately with ``append_to_catalog_files`` so a restart keeps it.
        """
        i = len(self.books)
        self.books.append(book)  # BookTable INSERT (live DB) or list.append (tests)
        self.emb = np.vstack([self.emb, np.asarray(emb_vec, dtype=self.emb.dtype)[None, :]])
        self.sim = sparse.block_diag(
            [self.sim, sparse.csr_matrix((1, 1), dtype=self.sim.dtype)], format="csr"
        ).tocsr()
        self.pop = np.append(self.pop, np.float32(0.0))
        self.id_to_idx[book["id"]] = i
        self._authors = np.append(self._authors, book.get("author", "") or "")
        self._subjects.append(list(book.get("subjects") or []))
        self._lang = np.append(self._lang, (book.get("language") or "").lower())
        self._year = np.append(self._year, book["year"] if book.get("year") is not None else np.nan)
        for s in book.get("subjects", []) or []:
            prev = self._genre_idx.get(s.lower())
            self._genre_idx[s.lower()] = (
                np.append(prev, np.int32(i)) if prev is not None else np.array([i], dtype=np.int32)
            )
        return i

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
                rows = self._genre_idx.get(g.lower())
                if rows is not None:
                    gmask[rows] = True
            mask &= gmask
        if year_min is not None:  # NaN (missing year) compares False -> excluded, as before
            mask &= self._year >= year_min
        if year_max is not None:
            mask &= self._year <= year_max
        return mask

    def all_genres(self) -> list[str]:
        """Subjects ranked by frequency (ties by first appearance) -- from the
        resident inverted index, so no full catalog scan."""
        return sorted(self._genre_idx, key=lambda s: -len(self._genre_idx[s]))


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
