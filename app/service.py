"""Application service: the API the UI (or a CLI) calls.

Ties together the catalog, the adaptive-hybrid recommender, title resolution,
and the swipe log. This is the seam a Streamlit app or an HTTP layer sits on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .recommender import Recommender, Scored
from .search import Match, TitleIndex
from .store import DATA, Catalog, SwipeStore


@dataclass
class SeedResult:
    resolved: List[Match]
    unresolved: List[str]


class BookRecommenderService:
    def __init__(
        self,
        data_dir: Path = DATA,
        db_path: Optional[Path] = None,
        check_same_thread: bool = True,
    ):
        self.catalog = Catalog.load(data_dir)
        self.recommender = Recommender(self.catalog)
        self.titles = TitleIndex(self.catalog)
        self.store = SwipeStore(db_path or data_dir / "app.db", check_same_thread=check_same_thread)

    # ---- users --------------------------------------------------------------

    def new_user(self, name: str = "") -> str:
        return self.store.create_user(name)

    def user_exists(self, user_id: str) -> bool:
        return self.store.user_exists(user_id)

    def name_profile(self, user_id: str, name: str) -> None:
        self.store.set_name(user_id, name)

    def profile_name(self, user_id: str) -> str:
        return self.store.user_name(user_id)

    def list_profiles(self) -> List[dict]:
        return self.store.named_users()

    def liked_titles(self, user_id: str) -> Dict[str, str]:
        """Map of book_id -> title for the user's likes (to rebuild onboarding seeds)."""
        out = {}
        for bid, r in self.store.reactions(user_id).items():
            if r == "like" and bid in self.catalog.id_to_idx:
                out[bid] = self.catalog.books[self.catalog.idx(bid)]["title"]
        return out

    def profile_summary(self, user_id: str) -> Dict[str, int]:
        counts = {"like": 0, "dislike": 0, "skip": 0, "interested": 0}
        for r in self.store.reactions(user_id).values():
            counts[r] = counts.get(r, 0) + 1
        return counts

    def wishlist(self, user_id: str) -> List[dict]:
        """Books the user marked 'interested' — their saved reading list."""
        reactions = self.store.reactions(user_id)
        return [
            self.catalog.books[self.catalog.idx(bid)]
            for bid, r in reactions.items()
            if r == "interested" and bid in self.catalog.id_to_idx
        ]

    # ---- onboarding ---------------------------------------------------------

    def search_titles(self, query: str, k: int = 5) -> List[Match]:
        return self.titles.search(query, k)

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

    # ---- swipe loop ---------------------------------------------------------

    def next_cards(
        self, user_id: str, n: int = 10, rng: Optional[random.Random] = None, **filters
    ) -> List[Scored]:
        return self.recommender.next_cards(
            self.store.reactions(user_id), _clean_filters(filters), n=n, rng=rng
        )

    def swipe(self, user_id: str, book_id: str, reaction: str) -> None:
        self.store.record(user_id, book_id, reaction)

    def recommendations(self, user_id: str, n: int = 10, **filters) -> List[Scored]:
        return self.recommender.recommend(
            self.store.reactions(user_id), _clean_filters(filters), n=n
        )

    def surprises(self, user_id: str, n: int = 10, **filters) -> List[Scored]:
        return self.recommender.surprise(
            self.store.reactions(user_id), _clean_filters(filters), n=n
        )

    def genres(self) -> List[str]:
        return self.catalog.all_genres()

    def close(self) -> None:
        self.store.close()


def _clean_filters(filters: dict) -> dict:
    """Keep only the filter keys Catalog.filter_mask understands, dropping Nones."""
    allowed = ("languages", "genres", "year_min", "year_max")
    return {k: v for k, v in filters.items() if k in allowed and v is not None}
