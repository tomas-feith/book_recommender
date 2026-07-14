"""The adaptive hybrid recommender that powers the swipe loop.

Design follows directly from the offline eval findings:

* Rocchio taste profile from liked (minus disliked) book embeddings.
* Content score (embedding cosine) AND collaborative score (item-item CF).
* **Per-item adaptive blend**: each candidate's weight on CF scales with how many
  ratings it has. A brand-new / obscure book (few ratings) is ranked by content;
  a well-rated book is ranked by CF. This is the direct answer to the cold-start
  result, where a static 50/50 blend was wrong in both regimes.
* Hard metadata filters (language / genre / year) applied AROUND the vector
  search, never inside the embedding.
* Card selection for the swipe loop mixes exploit / explore / diversity so the
  user isn't shown ten near-identical books and we keep learning their taste.

Serving needs only numpy — embeddings are precomputed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .store import Catalog

POP_REF = 500.0      # ratings at which we ~fully trust CF; tuned for the EASE-R
                     # core (which beats content at every pop tier >= 8 ratings),
                     # so the blend leans on CF fast. pop=0 books still get
                     # cf_weight=0 -> pure content, protecting true cold-start.
BETA = 0.5           # dislike weight (Rocchio + CF), < 1 because dislikes are noisier
ALPHA = 0.6          # "interested" weight (Rocchio + CF), < 1: intent is softer than a like
MMR_LAMBDA = 0.5     # diversity vs relevance in card selection
EXPLORE_FRAC = 0.25  # share of swipe cards drawn from beyond the top band


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else x - x.mean()


@dataclass
class Scored:
    book: dict
    score: float
    cf_weight: float   # how much CF drove this pick (0 = pure content)
    novelty: float = 0.0  # 1 - cosine to nearest liked book (set for surprise picks)


class Recommender:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        # Per-book trust in CF, from rating count (log-scaled, capped at 1).
        self.cf_weight = np.clip(
            np.log1p(catalog.pop) / np.log1p(POP_REF), 0.0, 1.0
        ).astype(np.float32)

    # ---- core scoring -------------------------------------------------------

    def _profile(
        self,
        liked: Sequence[int],
        disliked: Sequence[int],
        interested: Sequence[int] = (),
    ) -> Optional[np.ndarray]:
        if not liked and not interested:
            return None
        vec = np.zeros(self.cat.emb.shape[1], dtype=np.float32)
        if liked:
            vec = vec + self.cat.emb[list(liked)].mean(axis=0)
        if interested:
            vec = vec + ALPHA * self.cat.emb[list(interested)].mean(axis=0)
        if disliked:
            vec = vec - BETA * self.cat.emb[list(disliked)].mean(axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def _scores(self, liked, disliked, interested, cand: np.ndarray) -> np.ndarray:
        """Adaptive-hybrid score for each candidate index in ``cand``.

        Note: a *single* Rocchio centroid is used deliberately. Per-cluster
        multi-taste profiles were built and evaluated on the real profiles
        (held-out Recall@10) and consistently underperformed the pooled mean --
        sub-centroids from a handful of likes overfit, and no real user's tastes
        were separable enough to help. See git history for the experiment.
        """
        profile = self._profile(liked, disliked, interested)
        if profile is None:
            # Cold user, no signal yet: fall back to popularity.
            return _standardize(np.log1p(self.cat.pop[cand]))

        content = self.cat.emb[cand] @ profile
        cf = self._cf_sum(cand, liked)
        if interested:
            cf = cf + ALPHA * self._cf_sum(cand, interested)
        if disliked:
            cf = cf - BETA * self._cf_sum(cand, disliked)

        w = self.cf_weight[cand]
        return w * _standardize(cf) + (1.0 - w) * _standardize(content)

    def _cf_sum(self, cand: np.ndarray, idxs) -> np.ndarray:
        """Summed CF similarity from each candidate to the books in ``idxs``.

        ``cat.sim`` is a sparse CSR matrix, so ``.sum(axis=1)`` yields a matrix;
        flatten it back to a 1-D array.
        """
        if len(idxs) == 0:
            return np.zeros(len(cand), dtype=np.float32)
        return np.asarray(self.cat.sim[cand][:, list(idxs)].sum(axis=1)).ravel()

    def _candidate_mask(self, reactions: Dict[str, str], filters: dict) -> np.ndarray:
        mask = self.cat.filter_mask(**filters)
        for bid in reactions:  # exclude everything already swiped
            if bid in self.cat.id_to_idx:
                mask[self.cat.id_to_idx[bid]] = False
        return mask

    def _primary_author(self, book_idx: int) -> str:
        """First credited author, normalized -- so 'Rowling, GrandPré' and plain
        'Rowling' (translator/illustrator noise) dedupe together."""
        return (self.cat.books[book_idx].get("author", "").split(",")[0]).strip().lower()

    def _split(self, reactions: Dict[str, str]):
        liked = [self.cat.idx(b) for b, r in reactions.items()
                 if r == "like" and b in self.cat.id_to_idx]
        disliked = [self.cat.idx(b) for b, r in reactions.items()
                    if r == "dislike" and b in self.cat.id_to_idx]
        interested = [self.cat.idx(b) for b, r in reactions.items()
                      if r == "interested" and b in self.cat.id_to_idx]
        return liked, disliked, interested

    # ---- public API ---------------------------------------------------------

    def recommend(
        self, reactions: Dict[str, str], filters: dict, n: int = 10, per_author: int = 2
    ) -> List[Scored]:
        """Best-guess recommendations (exploit), for a 'For You' list.

        Capped at ``per_author`` books per author so a single series can't take
        over the list.
        """
        liked, disliked, interested = self._split(reactions)
        cand = np.where(self._candidate_mask(reactions, filters))[0]
        if len(cand) == 0:
            return []
        scores = self._scores(liked, disliked, interested, cand)
        order = cand[np.argsort(-scores)]

        picks, author_count = [], {}
        for i in order:
            author = self._primary_author(int(i))
            if author and author_count.get(author, 0) >= per_author:
                continue
            picks.append(int(i))
            author_count[author] = author_count.get(author, 0) + 1
            if len(picks) == n:
                break
        return self._as_scored(picks, liked, disliked, interested)

    def next_cards(
        self,
        reactions: Dict[str, str],
        filters: dict,
        n: int = 10,
        rng: Optional[random.Random] = None,
    ) -> List[Scored]:
        """Cards for the swipe loop: exploit + explore + diversity."""
        rng = rng or random.Random()
        liked, disliked, interested = self._split(reactions)
        cand = np.where(self._candidate_mask(reactions, filters))[0]
        if len(cand) == 0:
            return []

        scores = self._scores(liked, disliked, interested, cand)
        order = cand[np.argsort(-scores)]

        # Positive taste signal = likes plus (softer) declared interest.
        n_pos = len(liked) + len(interested)

        # Early on (little signal) bias toward recognizable, popular books.
        if n_pos < 3:
            order = self._popularity_prior(order)

        n_explore = int(round(n * EXPLORE_FRAC)) if n_pos >= 3 else 0
        n_exploit = n - n_explore

        exploit_pool = order[: max(4 * n_exploit, n_exploit)]
        picks = self._mmr(exploit_pool, liked, disliked, interested, n_exploit, rng)

        if n_explore:
            picks += self._explore(order, set(picks), n_explore, rng)
        return self._as_scored(picks, liked, disliked, interested)

    def surprise(
        self,
        reactions: Dict[str, str],
        filters: dict,
        n: int = 10,
        relevance_quantile: float = 0.75,
        per_author: int = 2,
    ) -> List[Scored]:
        """Serendipity: books UNLIKE your taste that still score strongly.

        Two independent axes make this possible: content (cosine to your taste
        centroid) and CF (readers-like-you). We gate to candidates in the top
        ``1 - relevance_quantile`` of blended score -- so every pick is still a
        confident recommendation -- then rank those by *novelty*: distance to the
        nearest book you've liked. A high-scoring, far-from-taste book is one the
        CF channel is carrying, i.e. "readers like you love it, though it's
        nothing like your usual reads."

        Needs positive signal to define "your taste"; returns [] for cold users.
        """
        liked, disliked, interested = self._split(reactions)
        pos = liked + interested
        if not pos:
            return []
        cand = np.where(self._candidate_mask(reactions, filters))[0]
        if len(cand) == 0:
            return []

        rel = self._scores(liked, disliked, interested, cand)
        # Gate: keep only strongly-recommended candidates.
        keep_mask = rel >= np.quantile(rel, relevance_quantile)
        keep, keep_rel = cand[keep_mask], rel[keep_mask]

        # Novelty = 1 - cosine to the nearest liked/interested book.
        nearest = (self.cat.emb[keep] @ self.cat.emb[pos].T).max(axis=1)
        novelty = 1.0 - nearest

        out, author_count = [], {}
        for j in np.argsort(-novelty):  # most novel first
            i = int(keep[j])
            author = self._primary_author(i)
            if author and author_count.get(author, 0) >= per_author:
                continue
            author_count[author] = author_count.get(author, 0) + 1
            out.append(Scored(
                book=self.cat.books[i], score=float(keep_rel[j]),
                cf_weight=float(self.cf_weight[i]), novelty=float(novelty[j]),
            ))
            if len(out) == n:
                break
        return out

    # ---- selection helpers --------------------------------------------------

    def _popularity_prior(self, order: np.ndarray) -> np.ndarray:
        """Blend score-rank with popularity-rank so early cards are familiar."""
        pop_rank = np.argsort(-self.cat.pop[order])
        combined = 0.5 * np.arange(len(order)) + 0.5 * np.argsort(pop_rank)
        return order[np.argsort(combined)]

    def _mmr(self, pool, liked, disliked, interested, k, rng) -> List[int]:
        """Maximal-marginal-relevance select, with one-book-per-author dedup."""
        pool = list(pool)
        if not pool:
            return []
        rel = self._scores(liked, disliked, interested, np.array(pool))
        rel = {p: rel[i] for i, p in enumerate(pool)}
        selected: List[int] = []
        chosen_authors: set = set()
        while pool and len(selected) < k:
            best, best_val = None, -1e18
            for p in pool:
                author = self._primary_author(p)
                if author and author in chosen_authors:
                    continue
                # Diversity = similarity to the most-similar already-picked book.
                if selected:
                    div = float(np.max(self.cat.emb[p] @ self.cat.emb[selected].T))
                else:
                    div = 0.0
                val = MMR_LAMBDA * rel[p] - (1 - MMR_LAMBDA) * div
                if val > best_val:
                    best, best_val = p, val
            if best is None:  # only same-author books left; relax dedup
                best = max(pool, key=lambda p: rel[p])
            selected.append(best)
            chosen_authors.add(self._primary_author(best))
            pool.remove(best)
        return selected

    def _explore(self, order, taken, k, rng) -> List[int]:
        """Draw from the mid-tail to gain information about uncertain taste."""
        band = [int(i) for i in order[4 * k: 25 * k] if int(i) not in taken]
        rng.shuffle(band)
        return band[:k]

    def _as_scored(self, idxs, liked, disliked, interested) -> List[Scored]:
        idxs = [int(i) for i in idxs]
        if not idxs:
            return []
        scores = self._scores(liked, disliked, interested, np.array(idxs))
        return [
            Scored(book=self.cat.books[i], score=float(s), cf_weight=float(self.cf_weight[i]))
            for i, s in zip(idxs, scores)
        ]
