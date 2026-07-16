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

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .store import Catalog

POP_REF = 500.0  # ratings at which we ~fully trust CF; tuned for the EASE-R
# core (which beats content at every pop tier >= 8 ratings),
# so the blend leans on CF fast. pop=0 books still get
# cf_weight=0 -> pure content, protecting true cold-start.
BETA = 0.5  # dislike weight (Rocchio + CF), < 1 because dislikes are noisier
ALPHA = 0.6  # "interested" weight (Rocchio + CF), < 1: intent is softer than a like
MMR_LAMBDA = 0.5  # diversity vs relevance in MMR selection (1.0 = pure relevance)
EXPLORE_FRAC = 0.25  # share of swipe cards drawn from beyond the top band
REC_POOL_MULT = 20  # "For You": diversify within the top REC_POOL_MULT*n candidates
# Honorifics are comma-split into their own credit ('Martin Luther King, Jr.'), so
# they'd cap every unrelated 'Jr.' together. Mononyms (Plato, CLAMP) are real keys.
_SUFFIXES = frozenset({"jr.", "jr", "sr.", "sr", "ii", "iii", "iv", "ph.d.", "phd", "m.d.", "md"})
CAL_LAMBDA = 0.4  # "For You": genre-calibration strength (0 = off); see finetune notes
CAL_SMOOTH = 0.01  # KL smoothing so an absent list-genre doesn't blow up


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else x - x.mean()


def genre_distribution(
    subject_lists: Sequence[Sequence[str]], weights: Sequence[float] | None = None
) -> dict[str, float]:
    """Normalized genre distribution over a set of books' subject lists.

    Each book splits one unit of mass equally across its own subjects (so a book
    tagged 3 genres contributes 1/3 to each), optionally scaled by ``weights``.
    Books with no subjects contribute nothing. Returns {} if there's no signal.
    """
    counts: dict[str, float] = {}
    for k, subs in enumerate(subject_lists):
        if not subs:
            continue
        w = 1.0 if weights is None else weights[k]
        for s in subs:
            counts[s] = counts.get(s, 0.0) + w / len(subs)
    total = sum(counts.values())
    return {g: v / total for g, v in counts.items()} if total else {}


def kl_calibration(
    target: dict[str, float], q: dict[str, float], alpha: float = CAL_SMOOTH
) -> float:
    """KL(target || smoothed q) -- Steck's list-miscalibration. 0 = perfectly matched.

    ``q`` is smoothed toward ``target`` so a genre the list hasn't covered yet is
    penalized finitely rather than infinitely.
    """
    kl = 0.0
    for g, p in target.items():
        qg = (1 - alpha) * q.get(g, 0.0) + alpha * p
        if p > 0 and qg > 0:
            kl += p * math.log(p / qg)
    return kl


@dataclass
class Scored:
    book: dict
    score: float
    cf_weight: float  # how much CF drove this pick (0 = pure content)
    novelty: float = 0.0  # 1 - cosine to nearest liked book (set for surprise picks)
    explanation: str = ""  # human "why recommended", from the driving signal


class Recommender:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        # Per-book trust in CF, from rating count (log-scaled, capped at 1).
        self.cf_weight = np.clip(np.log1p(catalog.pop) / np.log1p(POP_REF), 0.0, 1.0).astype(
            np.float32
        )

    # ---- core scoring -------------------------------------------------------

    def _profile(
        self,
        liked: Sequence[int],
        disliked: Sequence[int],
        interested: Sequence[int] = (),
    ) -> np.ndarray | None:
        if not liked and not interested:
            return None
        vec: np.ndarray = np.zeros(self.cat.emb.shape[1], dtype=np.float32)
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

    def _candidate_mask(self, reactions: dict[str, str], filters: dict) -> np.ndarray:
        mask = self.cat.filter_mask(**filters)
        for bid in reactions:  # exclude everything already swiped
            if bid in self.cat.id_to_idx:
                mask[self.cat.id_to_idx[bid]] = False
        return mask

    def _author_keys(self, book_idx: int) -> list[str]:
        """EVERY credited name, normalized -- the keys a book is capped under.

        Capping on the first credit only let an author past their own cap under a
        co-credit: 'Richard Bachman, Stephen King' keyed as 'richard bachman' (the
        pseudonym), and 'Stephen Briggs, Terry Pratchett' as 'stephen briggs', so a
        Discworld fan got the cap twice over. Matching on *any* shared credit closes
        that, and still dedupes 'Rowling, GrandPré' against plain 'Rowling'.
        """
        raw = self.cat.books[book_idx].get("author", "") or ""
        return [p for part in raw.split(",") if (p := part.strip().lower()) and p not in _SUFFIXES]

    def _cap_hit(self, keys: list[str], counts: dict[str, int], per_author: int) -> bool:
        return any(counts.get(k, 0) >= per_author for k in keys)

    def _count_author(self, keys: list[str], counts: dict[str, int]) -> None:
        for k in keys:
            counts[k] = counts.get(k, 0) + 1

    def _split(self, reactions: dict[str, str]):
        liked = [
            self.cat.idx(b) for b, r in reactions.items() if r == "like" and b in self.cat.id_to_idx
        ]
        disliked = [
            self.cat.idx(b)
            for b, r in reactions.items()
            if r == "dislike" and b in self.cat.id_to_idx
        ]
        interested = [
            self.cat.idx(b)
            for b, r in reactions.items()
            if r == "interested" and b in self.cat.id_to_idx
        ]
        return liked, disliked, interested

    # ---- public API ---------------------------------------------------------

    def recommend(
        self,
        reactions: dict[str, str],
        filters: dict,
        n: int = 10,
        per_author: int = 2,
        mmr_lambda: float = MMR_LAMBDA,
        cal_lambda: float = CAL_LAMBDA,
    ) -> list[Scored]:
        """Best-guess recommendations for a 'For You' list.

        Retrieves the top ``REC_POOL_MULT * n`` by relevance, then selects the
        final ``n`` with **MMR + genre calibration** so the list is spread out (not
        ten near-identical fantasy novels) *and* mirrors your taste mix (your
        sci-fi and your romance in proportion, not all of the majority genre),
        capped at ``per_author`` books per author. ``mmr_lambda`` trades relevance
        vs redundancy; ``cal_lambda`` sets how hard genres are matched (0 = off).
        """
        liked, disliked, interested = self._split(reactions)
        cand = np.where(self._candidate_mask(reactions, filters))[0]
        if len(cand) == 0:
            return []
        scores = self._scores(liked, disliked, interested, cand)
        order = cand[np.argsort(-scores)]
        pool = order[: max(REC_POOL_MULT * n, n)]
        picks = self._mmr(
            pool,
            liked,
            disliked,
            interested,
            n,
            lam=mmr_lambda,
            per_author=per_author,
            cal_lambda=cal_lambda,
        )
        return self._as_scored(picks, liked, disliked, interested)

    def next_cards(
        self,
        reactions: dict[str, str],
        filters: dict,
        n: int = 10,
        rng: random.Random | None = None,
    ) -> list[Scored]:
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

        n_explore = round(n * EXPLORE_FRAC) if n_pos >= 3 else 0
        n_exploit = n - n_explore

        exploit_pool = order[: max(4 * n_exploit, n_exploit)]
        picks = self._mmr(exploit_pool, liked, disliked, interested, n_exploit, per_author=1)

        if n_explore:
            picks += self._explore(order, set(picks), n_explore, rng)
        return self._as_scored(picks, liked, disliked, interested)

    def surprise(
        self,
        reactions: dict[str, str],
        filters: dict,
        n: int = 10,
        relevance_quantile: float = 0.75,
        per_author: int = 2,
    ) -> list[Scored]:
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

        out: list[Scored] = []
        author_count: dict[str, int] = {}
        for j in np.argsort(-novelty):  # most novel first
            i = int(keep[j])
            keys = self._author_keys(i)
            if self._cap_hit(keys, author_count, per_author):
                continue
            self._count_author(keys, author_count)
            out.append(
                Scored(
                    book=self.cat.books[i],
                    score=float(keep_rel[j]),
                    cf_weight=float(self.cf_weight[i]),
                    novelty=float(novelty[j]),
                )
            )
            if len(out) == n:
                break
        return out

    # ---- selection helpers --------------------------------------------------

    def _popularity_prior(self, order: np.ndarray) -> np.ndarray:
        """Blend score-rank with popularity-rank so early cards are familiar."""
        pop_rank = np.argsort(-self.cat.pop[order])
        combined = 0.5 * np.arange(len(order)) + 0.5 * np.argsort(pop_rank)
        return order[np.argsort(combined)]

    def _book_genre_mass(self, i: int) -> dict[str, float]:
        subs = self.cat.books[i].get("subjects", []) or []
        return {s: 1.0 / len(subs) for s in subs} if subs else {}

    def _genre_target(self, liked, interested) -> dict[str, float]:
        """The user's taste genre distribution (likes full weight, interest ALPHA)."""
        subs = [self.cat.books[i].get("subjects", []) for i in list(liked) + list(interested)]
        weights = [1.0] * len(liked) + [ALPHA] * len(interested)
        return genre_distribution(subs, weights)

    def _mmr(
        self,
        pool,
        liked,
        disliked,
        interested,
        k: int,
        lam: float = MMR_LAMBDA,
        per_author: int = 1,
        cal_lambda: float = 0.0,
    ) -> list[int]:
        """Greedy list selection: relevance − redundancy (MMR) − genre miscalibration.

        Picks the candidate maximizing ``lam * relevance - (1 - lam) *
        max-similarity-to-already-picked - cal_lambda * KL(taste ‖ list genres)``
        (``lam`` toward 0 = more diverse; ``cal_lambda`` > 0 pulls the list's genre
        mix toward the user's, so a multi-taste reader sees each taste in
        proportion). At most ``per_author`` books per author. ``cal_lambda=0``
        recovers plain MMR. Exact set-diversity is NP-hard; this greedy is the
        standard tractable approximation.
        """
        pool = list(pool)
        if not pool:
            return []
        rel_scores = self._scores(liked, disliked, interested, np.array(pool))
        rel = {p: float(rel_scores[i]) for i, p in enumerate(pool)}
        target = self._genre_target(liked, interested) if cal_lambda > 0 else {}
        sel_mass: dict[str, float] = {}  # running (unnormalized) genre mass of `selected`
        sel_total = 0.0
        selected: list[int] = []
        author_count: dict[str, int] = {}
        while pool and len(selected) < k:
            # Redundancy for ALL remaining candidates in one matmul (candidates x picked).
            rel_v = np.fromiter((rel[p] for p in pool), dtype=np.float64, count=len(pool))
            if selected:
                div_v = (self.cat.emb[pool] @ self.cat.emb[selected].T).max(axis=1)
            else:
                div_v = np.zeros(len(pool))
            base = lam * rel_v - (1 - lam) * div_v
            best_idx, best_val = -1, -1e18
            for idx, p in enumerate(pool):
                if self._cap_hit(self._author_keys(p), author_count, per_author):
                    continue
                val = float(base[idx])
                if target:  # genre-calibration penalty for the list-with-p (cheap, per-item)
                    pm = self._book_genre_mass(p)
                    tot = sel_total + sum(pm.values())
                    if tot > 0:
                        q = {
                            g: (sel_mass.get(g, 0.0) + pm.get(g, 0.0)) / tot
                            for g in set(sel_mass) | set(pm)
                        }
                        val -= cal_lambda * kl_calibration(target, q)
                if val > best_val:
                    best_idx, best_val = idx, val
            if best_idx < 0:  # every author cap hit; relax and take the most relevant
                best_idx = int(np.argmax(rel_v))
            best = pool[best_idx]
            selected.append(best)
            self._count_author(self._author_keys(best), author_count)
            for g, m in self._book_genre_mass(best).items():  # fold into running mass
                sel_mass[g] = sel_mass.get(g, 0.0) + m
                sel_total += m
            pool.pop(best_idx)
        return selected

    def _explore(self, order, taken, k, rng) -> list[int]:
        """Draw from the mid-tail to gain information about uncertain taste."""
        band = [int(i) for i in order[4 * k : 25 * k] if int(i) not in taken]
        rng.shuffle(band)
        return band[:k]

    def _as_scored(self, idxs, liked, disliked, interested) -> list[Scored]:
        idxs = [int(i) for i in idxs]
        if not idxs:
            return []
        scores = self._scores(liked, disliked, interested, np.array(idxs))
        pos = list(liked) + list(interested)
        return [
            Scored(
                book=self.cat.books[i],
                score=float(s),
                cf_weight=float(self.cf_weight[i]),
                explanation=self._explain(i, pos),
            )
            for i, s in zip(idxs, scores, strict=True)
        ]

    def _explain(self, i: int, pos: Sequence[int]) -> str:
        """A one-line 'why recommended', attributed to the driving signal.

        CF-driven picks cite the liked book whose readers most overlap; content-
        driven picks cite the most similar liked book. Cold users get a neutral line.
        """
        if not pos:
            return "A popular pick to get you started"
        title = self.cat.books  # local alias
        cf = np.asarray(self.cat.sim.getrow(i)[:, list(pos)].todense()).ravel()
        if self.cf_weight[i] >= 0.5 and cf.size and cf.max() > 0:
            j = pos[int(np.argmax(cf))]
            return f'Readers who liked "{title[j]["title"]}" also enjoyed this'
        content = self.cat.emb[list(pos)] @ self.cat.emb[i]
        j = pos[int(np.argmax(content))]
        return f'Because you liked "{title[j]["title"]}"'

    def similar(self, book_id: str, n: int = 10, per_author: int = 2) -> list[Scored]:
        """Books most like a given one -- 'More like this'.

        Blends content similarity (embedding cosine) and CF ('read together'),
        weighted by the seed book's own CF warmth, excluding the book itself and
        capping per author so a series doesn't fill the shelf.
        """
        if book_id not in self.cat.id_to_idx:
            return []
        i = self.cat.idx(book_id)
        content = self.cat.emb @ self.cat.emb[i]
        cf = np.asarray(self.cat.sim.getrow(i).todense()).ravel()
        w = float(self.cf_weight[i])
        score = w * _standardize(cf) + (1.0 - w) * _standardize(content)
        score[i] = -1e18  # never recommend the book itself
        picks: list[int] = []
        author_count: dict[str, int] = {}
        for j in np.argsort(-score):
            j = int(j)
            keys = self._author_keys(j)
            if self._cap_hit(keys, author_count, per_author):
                continue
            picks.append(j)
            self._count_author(keys, author_count)
            if len(picks) == n:
                break
        seed_title = self.cat.books[i]["title"]
        return [
            Scored(
                book=self.cat.books[j],
                score=float(score[j]),
                cf_weight=float(self.cf_weight[j]),
                explanation=f'Similar to "{seed_title}"',
            )
            for j in picks
        ]
