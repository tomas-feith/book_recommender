"""Title/author resolution for onboarding.

When a user types the name of a book (or an author) they like, we resolve it to a
specific work in the catalog. This is a search problem, not a recommendation one:
fuzzy matching over titles and authors, biased toward the more canonical (popular)
edition. (In production this is Postgres trigram / full-text search; here it's a
lightweight in-memory scorer.)

Foreign-language works keep their original title as a searchable alias, so both
"the three-body problem" and "三体" resolve to the same book.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import NamedTuple

import numpy as np

from .store import Catalog

_WORD = re.compile(r"\w+")  # Unicode-aware, so non-Latin titles/aliases (三体) tokenize too


def _norm(s: str) -> str:
    return " ".join(_WORD.findall((s or "").lower()))


class Match(NamedTuple):
    book_id: str
    title: str
    author: str
    score: float


class TitleIndex:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        books = catalog.books
        # Title match runs against the display title PLUS any original-language
        # alias, so English and original queries both resolve.
        self._title = [_norm(b["title"]) for b in books]
        self._title_blob = [_norm(f"{b['title']} {b.get('orig_title', '')}") for b in books]
        self._title_tokens = [set(t.split()) for t in self._title_blob]
        self._author = [_norm(b.get("author", "")) for b in books]
        self._author_tokens = [set(a.split()) for a in self._author]
        # Popularity as a 0..1 rank, a small tiebreak so the canonical edition of a
        # fuzzy match (e.g. Harry Potter #1) surfaces above lesser ones.
        pop = np.asarray(catalog.pop, dtype=np.float64)
        self._pop_rank = (np.argsort(np.argsort(pop)) / max(len(pop) - 1, 1)).astype(float)

    def _score(self, q: str, q_tokens: set[str], i: int) -> float:
        blob = self._title_blob[i]
        if not blob:
            return 0.0
        ratio = SequenceMatcher(None, q, self._title[i]).ratio()
        overlap = len(q_tokens & self._title_tokens[i]) / len(q_tokens) if q_tokens else 0.0
        substring = 1.0 if q in blob or blob in q else 0.0
        title_score = 0.5 * ratio + 0.35 * overlap + 0.15 * substring
        # The query might be an author name ("brandon sanderson").
        a_overlap = len(q_tokens & self._author_tokens[i]) / len(q_tokens) if q_tokens else 0.0
        a_sub = 1.0 if q and q in self._author[i] else 0.0
        author_score = 0.7 * a_overlap + 0.3 * a_sub
        # Title matches beat author matches on a tie; popularity breaks near-ties.
        return max(title_score, 0.85 * author_score) + 0.02 * self._pop_rank[i]

    def search(self, query: str, k: int = 5) -> list[Match]:
        q = _norm(query)
        if not q:
            return []
        q_tokens = set(q.split())
        scored = [(self._score(q, q_tokens, i), i) for i in range(len(self.cat.books))]
        scored.sort(reverse=True)
        out = []
        for score, i in scored[:k]:
            b = self.cat.books[i]
            out.append(Match(b["id"], b["title"], b.get("author", ""), round(score, 3)))
        return out

    def best(self, query: str, threshold: float = 0.34) -> Match | None:
        hits = self.search(query, k=1)
        return hits[0] if hits and hits[0].score >= threshold else None
