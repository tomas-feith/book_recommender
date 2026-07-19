"""Application service: the API the UI (or a CLI) calls.

Ties together the catalog, the adaptive-hybrid recommender, title resolution,
and the swipe log. This is the seam a Streamlit app or an HTTP layer sits on.
"""

from __future__ import annotations

import random
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .external import enrich_description, search_books
from .library import LibraryEntry
from .recommender import Recommender, Scored
from .search import Match, TitleIndex
from .store import DATA, Catalog, SwipeStore, append_to_catalog_files

_AUTHOR_TOK = re.compile(r"[a-z]{2,}")


@dataclass
class SeedResult:
    resolved: list[Match]
    unresolved: list[str]


@dataclass
class MatchedEntry:
    entry: LibraryEntry
    match: Match


@dataclass
class ImportResult:
    matched: list[MatchedEntry]
    unmatched: list[LibraryEntry]

    @property
    def n_matched(self) -> int:
        return len(self.matched)


class BookRecommenderService:
    def __init__(
        self,
        data_dir: Path = DATA,
        db_path: Path | None = None,
        check_same_thread: bool = True,
    ):
        self.data_dir = data_dir
        self.catalog = Catalog.load(data_dir, check_same_thread=check_same_thread)
        self.recommender = Recommender(self.catalog)
        self.titles = TitleIndex(self.catalog)
        self.store = SwipeStore(db_path or data_dir / "app.db", check_same_thread=check_same_thread)
        self._encoder_cache: object | None = None
        self._encoder_loaded = False
        self._catalog_lock = threading.Lock()  # guards on-demand catalog growth

    # ---- users --------------------------------------------------------------

    def new_user(self, name: str = "") -> str:
        return self.store.create_user(name)

    def user_exists(self, user_id: str) -> bool:
        return self.store.user_exists(user_id)

    def name_profile(self, user_id: str, name: str) -> None:
        self.store.set_name(user_id, name)

    def profile_name(self, user_id: str) -> str:
        return self.store.user_name(user_id)

    def list_profiles(self) -> list[dict]:
        return self.store.named_users()

    def liked_titles(self, user_id: str) -> dict[str, str]:
        """Map of book_id -> title for the user's likes (to rebuild onboarding seeds)."""
        out = {}
        for bid, r in self.store.reactions(user_id).items():
            if r == "like" and bid in self.catalog.id_to_idx:
                out[bid] = self.catalog.books[self.catalog.idx(bid)]["title"]
        return out

    def profile_summary(self, user_id: str) -> dict[str, int]:
        counts = {"like": 0, "dislike": 0, "skip": 0, "interested": 0}
        for r in self.store.reactions(user_id).values():
            counts[r] = counts.get(r, 0) + 1
        return counts

    def wishlist(self, user_id: str) -> list[dict]:
        """Books the user marked 'interested' — their saved reading list."""
        reactions = self.store.reactions(user_id)
        return [
            self.catalog.books[self.catalog.idx(bid)]
            for bid, r in reactions.items()
            if r == "interested" and bid in self.catalog.id_to_idx
        ]

    # ---- onboarding ---------------------------------------------------------

    def search_titles(self, query: str, k: int = 5) -> list[Match]:
        return self.titles.search(query, k)

    def semantic_search(self, query: str, k: int = 8) -> list[dict]:
        """Search the catalog by meaning ("books about grief and the sea").

        Encodes the query into the same space as the book embeddings and returns
        the nearest books. Needs the embedding encoder (torch); returns [] if it
        isn't available (e.g. the numpy-only serving image), so callers can fall
        back to title search.
        """
        enc = self._encoder()
        if enc is None or not query.strip():
            return []
        qv = enc.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        order = np.argsort(-(self.catalog.emb @ qv))[:k]
        return [self.catalog.books[int(i)] for i in order]

    def similar_books(self, book_id: str, n: int = 8) -> list[Scored]:
        """'More like this' for a given book (content + CF neighbours)."""
        return self.recommender.similar(book_id, n)

    def external_search(self, query: str, k: int = 5) -> list[dict]:
        """Look a title up on Open Library (for books not in the catalog)."""
        return search_books(query, k=k)

    def add_external_book(self, record: dict, embedding: np.ndarray | None = None) -> str:
        """Ingest an Open Library record into the catalog CF-cold; return its id.

        The book is embedded (co-read encoder), appended to the live in-memory
        catalog + title index, and persisted to disk so it survives a restart. It
        starts with pop=0 (content-ranked) and can immediately be liked. Idempotent.
        """
        from eval.data import book_to_text

        rec = _external_record(record)
        if rec["id"] in self.catalog.id_to_idx:
            return rec["id"]
        rec = enrich_description(rec)  # fetch the description before embedding
        if embedding is None:
            enc = self._encoder()
            if enc is None:
                raise RuntimeError("embedding encoder unavailable")
            embedding = enc.encode([book_to_text(rec)], normalize_embeddings=True)[0].astype(
                np.float32
            )
        with self._catalog_lock:
            if rec["id"] in self.catalog.id_to_idx:  # re-check under the lock
                return rec["id"]
            self.catalog.append(rec, embedding)
            self.titles = TitleIndex(self.catalog)
            append_to_catalog_files([rec], np.asarray(embedding)[None, :], self.data_dir)
        return rec["id"]

    def _encoder(self):
        """Lazily load the embedding encoder (the co-read one if built, else base).

        Cached; returns None if sentence-transformers/torch isn't installed.
        """
        if not self._encoder_loaded:
            self._encoder_loaded = True
            try:
                from sentence_transformers import SentenceTransformer

                coread = self.data_dir / "coread-encoder"
                model = str(coread) if coread.exists() else "BAAI/bge-small-en-v1.5"
                self._encoder_cache = SentenceTransformer(model)
            except Exception:
                self._encoder_cache = None
        return self._encoder_cache

    def seed(self, user_id: str, titles: Sequence[str]) -> SeedResult:
        """Resolve typed titles to catalog books and record them as likes."""
        resolved, unresolved = [], []
        for t in titles:
            m = self.titles.best(t)
            if m:
                self.store.record(user_id, m.book_id, "like")
                resolved.append(m)
            else:
                unresolved.append(t)
        return SeedResult(resolved, unresolved)

    def import_library(
        self,
        user_id: str,
        entries: Sequence[LibraryEntry],
        reaction: str = "like",
        threshold: float = 0.55,
    ) -> ImportResult:
        """Match imported reading-list entries to catalog books and record them.

        Each entry is fuzzy-matched by title; a matching author (any shared name
        token) confirms a weaker title match, which trims false positives on
        common titles. Matched books are recorded with ``reaction`` (default a
        like, so an import seeds the taste profile). Unmatched entries are
        returned so the UI can tell the user what we couldn't find.
        """
        matched: list[MatchedEntry] = []
        unmatched: list[LibraryEntry] = []
        recorded: set[str] = set()
        for entry in entries:
            m = self._resolve_entry(entry, threshold)
            if m is None:
                unmatched.append(entry)
                continue
            if m.book_id not in recorded:
                self.store.record(user_id, m.book_id, reaction)
                recorded.add(m.book_id)
            matched.append(MatchedEntry(entry, m))
        return ImportResult(matched, unmatched)

    def _resolve_entry(self, entry: LibraryEntry, threshold: float) -> Match | None:
        hits = self.titles.search(entry.title, k=5)
        if not hits:
            return None
        if entry.author:
            want = set(_AUTHOR_TOK.findall(entry.author.lower()))
            for h in hits:
                have = set(_AUTHOR_TOK.findall(h.author.lower()))
                if want & have and h.score >= 0.4:  # author confirms a softer title match
                    return h
        return hits[0] if hits[0].score >= threshold else None

    # ---- swipe loop ---------------------------------------------------------

    def next_cards(
        self, user_id: str, n: int = 10, rng: random.Random | None = None, **filters
    ) -> list[Scored]:
        return self.recommender.next_cards(
            self.store.reactions(user_id), _clean_filters(filters), n=n, rng=rng
        )

    def swipe(self, user_id: str, book_id: str, reaction: str) -> None:
        self.store.record(user_id, book_id, reaction)

    def recommendations(self, user_id: str, n: int = 10, **filters) -> list[Scored]:
        return self.recommender.recommend(
            self.store.reactions(user_id), _clean_filters(filters), n=n
        )

    def surprises(self, user_id: str, n: int = 10, **filters) -> list[Scored]:
        return self.recommender.surprise(
            self.store.reactions(user_id), _clean_filters(filters), n=n
        )

    def genres(self) -> list[str]:
        return self.catalog.all_genres()

    def close(self) -> None:
        self.store.close()


def _clean_filters(filters: dict) -> dict:
    """Keep only the filter keys Catalog.filter_mask understands, dropping Nones."""
    allowed = ("languages", "genres", "year_min", "year_max")
    return {k: v for k, v in filters.items() if k in allowed and v is not None}


def _external_record(record: dict) -> dict:
    """Coerce an external (Open Library) record into the catalog schema."""
    return {
        "id": str(record["id"]),
        "title": record.get("title", ""),
        "author": record.get("author", ""),
        "subjects": record.get("subjects", []),
        "language": record.get("language", "en"),
        "year": record.get("year"),
        "image": record.get("image", ""),
        "description": record.get("description", ""),
    }
