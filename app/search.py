"""Title resolution for onboarding.

When a user types the name of a book they like, we must resolve it to a specific
work in the catalog. This is a search problem, not a recommendation one: fuzzy
title matching with author disambiguation. (In production this is Postgres
trigram / full-text search; here it's a lightweight in-memory scorer.)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, NamedTuple

from .store import Catalog

_WORD = re.compile(r"[a-z0-9]+")


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
        self._norm_titles = [_norm(b["title"]) for b in catalog.books]
        self._title_tokens = [set(t.split()) for t in self._norm_titles]

    def search(self, query: str, k: int = 5) -> List[Match]:
        q = _norm(query)
        if not q:
            return []
        q_tokens = set(q.split())
        results = []
        for i, (nt, toks) in enumerate(zip(self._norm_titles, self._title_tokens)):
            if not nt:
                continue
            ratio = SequenceMatcher(None, q, nt).ratio()
            overlap = len(q_tokens & toks) / len(q_tokens) if q_tokens else 0.0
            substring = 1.0 if q in nt or nt in q else 0.0
            score = 0.5 * ratio + 0.35 * overlap + 0.15 * substring
            results.append((score, i))
        results.sort(reverse=True)
        out = []
        for score, i in results[:k]:
            b = self.cat.books[i]
            out.append(Match(b["id"], b["title"], b.get("author", ""), round(score, 3)))
        return out

    def best(self, query: str, threshold: float = 0.34) -> Match | None:
        hits = self.search(query, k=1)
        return hits[0] if hits and hits[0].score >= threshold else None
