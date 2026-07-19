"""Title/author resolution for onboarding.

When a user types the name of a book (or an author) they like, we resolve it to a
specific work in the catalog. This is a search problem, not a recommendation one:
fuzzy matching over titles and authors, biased toward the more canonical (popular)
edition. On the SQLite-backed catalog this is a **trigram FTS retrieve** (sublinear)
followed by an exact fuzzy rerank of the candidates; a plain list catalog (tests)
scans every book with the same scorer.

Foreign-language works keep their original title as a searchable alias, so both
"the three-body problem" and "三体" resolve to the same book.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import NamedTuple

import numpy as np

from .store import BookTable, Catalog

_WORD = re.compile(r"\w+")  # Unicode-aware, so non-Latin titles/aliases (三体) tokenize too


def _norm(s: str) -> str:
    return " ".join(_WORD.findall((s or "").lower()))


class Match(NamedTuple):
    book_id: str
    title: str
    author: str
    score: float


def _score_fields(
    q: str,
    q_tokens: set[str],
    title: str,
    blob: str,
    blob_tokens: set[str],
    author: str,
    author_tokens: set[str],
    pop_rank: float,
) -> float:
    """The fuzzy title/author score for one book (all fields already normalized)."""
    if not blob:
        return 0.0
    ratio = SequenceMatcher(None, q, title).ratio()
    overlap = len(q_tokens & blob_tokens) / len(q_tokens) if q_tokens else 0.0
    substring = 1.0 if q in blob or blob in q else 0.0
    title_score = 0.5 * ratio + 0.35 * overlap + 0.15 * substring
    # The query might be an author name ("brandon sanderson").
    a_overlap = len(q_tokens & author_tokens) / len(q_tokens) if q_tokens else 0.0
    a_sub = 1.0 if q and q in author else 0.0
    author_score = 0.7 * a_overlap + 0.3 * a_sub
    # Title matches beat author matches on a tie; popularity breaks near-ties.
    return max(title_score, 0.85 * author_score) + 0.02 * pop_rank


class TitleIndex:
    """Fuzzy title/author resolution.

    On a SQLite-backed catalog it retrieves candidates via the trigram FTS index
    (sublinear) and reranks them with the exact fuzzy scorer; on an in-memory list
    catalog (tests) it scans and scores every book. Either way the *ranking* is the
    same ``_score_fields``, so results match -- FTS only narrows the field first.
    """

    def __init__(self, catalog: Catalog):
        self.cat = catalog
        # Popularity as a 0..1 rank, a small tiebreak so the canonical edition of a
        # fuzzy match (e.g. Harry Potter #1) surfaces above lesser ones. Cheap even
        # at 1M (a function of pop alone), so it stays resident on both paths.
        pop = np.asarray(catalog.pop, dtype=np.float64)
        self._pop_rank = (np.argsort(np.argsort(pop)) / max(len(pop) - 1, 1)).astype(float)
        self._fts = isinstance(catalog.books, BookTable)
        if not self._fts:  # list catalog: precompute the per-book normalized fields
            books = catalog.books
            self._title = [_norm(b["title"]) for b in books]
            self._blob = [_norm(f"{b['title']} {b.get('orig_title', '')}") for b in books]
            self._blob_tokens = [set(t.split()) for t in self._blob]
            self._author = [_norm(b.get("author", "")) for b in books]
            self._author_tokens = [set(a.split()) for a in self._author]

    def _score_list(self, q: str, q_tokens: set[str], i: int) -> float:
        return _score_fields(
            q,
            q_tokens,
            self._title[i],
            self._blob[i],
            self._blob_tokens[i],
            self._author[i],
            self._author_tokens[i],
            self._pop_rank[i],
        )

    def _score_record(self, q: str, q_tokens: set[str], i: int) -> float:
        b = self.cat.books[i]
        title = _norm(b["title"])
        blob = _norm(f"{b['title']} {b.get('orig_title', '')}")
        author = _norm(b.get("author", ""))
        return _score_fields(
            q,
            q_tokens,
            title,
            blob,
            set(blob.split()),
            author,
            set(author.split()),
            self._pop_rank[i],
        )

    def search(self, query: str, k: int = 5) -> list[Match]:
        q = _norm(query)
        if not q:
            return []
        q_tokens = set(q.split())
        books = self.cat.books
        if isinstance(books, BookTable):  # retrieve a candidate field via FTS, then rank exactly
            cand = books.search_fts(query, limit=max(200, 40 * k))
            scored = [(self._score_record(q, q_tokens, i), i) for i in cand]
        else:
            scored = [(self._score_list(q, q_tokens, i), i) for i in range(len(self.cat.books))]
        scored.sort(reverse=True)
        out = []
        for score, i in scored[:k]:
            b = self.cat.books[i]
            out.append(Match(b["id"], b["title"], b.get("author", ""), round(score, 3)))
        return out

    def best(self, query: str, threshold: float = 0.34) -> Match | None:
        hits = self.search(query, k=1)
        return hits[0] if hits and hits[0].score >= threshold else None
