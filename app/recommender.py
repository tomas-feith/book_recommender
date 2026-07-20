"""The adaptive hybrid recommender that powers the swipe loop.

Design follows directly from the offline eval findings:

* Rocchio taste profile from liked (minus disliked) book embeddings.
* Content score (embedding cosine) AND collaborative score (item-item CF).
* **Per-item adaptive blend**: ``content + cf_weight * cf``, where each candidate's
  weight on CF scales with how many ratings it has. Content is the baseline every
  book is judged on; CF is *evidence on top* for books that have readers. A
  brand-new / obscure book is therefore ranked on content alone rather than being
  compared against a quantity it cannot have. This is the direct answer to the
  cold-start result, where a static 50/50 blend was wrong in both regimes -- and to
  the later finding that a *convex* blend silently made cold books unreachable.
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
# "For You": books outside the top HEAD_FRAC by popularity are the *tail*, and
# TAIL_SLOTS_FRAC of the list is reserved for them. Measured at 100k, head and tail
# compete for the same ten slots: swapping in a stronger head model (EASE over iALS)
# lifted head Recall 0.169 -> 0.237 and pushed tail Recall 0.104 -> 0.050 with no change
# to the tail model at all. So tail exposure cannot be bought by better tail ranking
# alone -- it has to be allocated. 2 of 10 slots by default.
HEAD_FRAC = 0.1
TAIL_SLOTS_FRAC = 0.2


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else x - x.mean()


def _standardize_sparse(x: np.ndarray) -> np.ndarray:
    """Standardize a sparse channel using the spread of its NON-ZERO entries.

    Z-scoring both channels was silently broken for cold-start. The CF channel is a sum
    over a top-k sparse matrix, so ~96% of candidates score exactly 0; that zero mass
    collapses the standard deviation, and the few non-zero entries come out around +57
    sigma while the dense content cosine tops out near +4.5. The blend then added two
    quantities ~13x apart in scale, so *any* CF connection beat *every* pure-content
    book: the measured cold share of the top-10 was 0.0% against a 40% base rate, i.e.
    cold books were structurally unreachable no matter how well they matched.

    Taking mean/std over the entries that actually carry signal fixes the scale while
    keeping CF *magnitude*, which is the part that matters -- EASE weights are
    meaningful, and rank-normalizing (which does fix the scale) flattens "co-read 50x"
    against "co-read once" and cost 32% of warm Recall to buy the same cold gain.

    A second property matters for the additive blend in ``_scores``: a candidate with
    *no* CF evidence maps to ``(0 - mu)/std``, i.e. a negative value. So a warm book
    that nothing co-read is pushed **below** a cold book with equal content fit, which
    is the correct ordering -- absence of evidence for a book that has readers is
    informative, absence for a book that has none is not.
    """
    nz = x[x != 0]
    if nz.size < 2:
        return _standardize(x)
    mu, std = nz.mean(), nz.std()
    return (x - mu) / std if std > 1e-9 else x - mu


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
        # Per-book trust in CF, from rating count (log-scaled, capped at 1)...
        w = np.clip(np.log1p(catalog.pop) / np.log1p(POP_REF), 0.0, 1.0)
        # ...but zeroed wherever the book has no CF row at all. Popularity and CF
        # coverage come apart once the catalog outgrows the EASE budget: EASE solves
        # over the ``max_items`` most-rated books, so at 250k every book is
        # popularity-warm (rank 250,000 still has 201 ratings) while ~230k of them
        # have a structurally empty ``sim`` row. Those would score
        # ``cf_weight ~ 0.85`` against a CF sum of exactly 0, which
        # ``_standardize_sparse`` maps to ~ -2.1 -- penalizing the entire mid-tail
        # ~1.8 z-units *below books with no ratings at all*, on a content channel
        # spanning only about +/-4.5. cf_weight has to mean "we have CF evidence for
        # this book", not "this book is popular".
        has_cf = np.diff(catalog.sim.indptr) > 0
        self.cf_weight = np.where(has_cf, w, 0.0).astype(np.float32)
        # Popularity rank, for the reserved tail slots. Rank rather than a pop threshold
        # so the split means the same thing at 22k and at 1M.
        n_head = max(1, int(len(catalog.pop) * HEAD_FRAC))
        self.is_tail = np.ones(len(catalog.pop), dtype=bool)
        self.is_tail[np.argsort(-catalog.pop, kind="stable")[:n_head]] = False

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

        The blend is **additive**: ``content + cf_weight * cf``, not the convex
        ``w*cf + (1-w)*content`` it used to be. Under the convex form a warm book was
        scored on CF and a cold book on content -- two different quantities compared
        directly, with nothing calibrating them -- and cold books lost every time
        (measured: 0.0% of the top-10 against a 40% base rate). Additively, every book
        shares the content baseline and CF is *evidence on top*, so the comparison is
        always content-vs-content plus a bounded, signed bonus.

        Swept against the served harness on the real catalog at cold fractions
        0.40/0.70/0.90/0.95 (a 250k catalog is ~92% CF-cold, 1M ~98%, so the current
        40% is the wrong point to tune at). Additive+sparse-standardized wins the
        aggregate at *every* fraction -- 0.222/0.136/0.118/0.124 vs the old
        0.206/0.096/0.106/0.118 -- and at f=0.40 it improves warm Recall@10 too
        (0.360 vs 0.348) rather than trading it away. Rank-normalizing both channels
        also fixes the scale but flattens CF magnitude and loses 32% of warm Recall.

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
        return _standardize(content) + w * _standardize_sparse(cf)

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

    def _cf_neighbors(self, liked, interested) -> np.ndarray:
        """Row indices that co-occur (in the CF matrix) with the taste set."""
        rows = list(liked) + list(interested)
        if not rows:
            return np.array([], dtype=np.int64)
        return np.unique(self.cat.sim[rows].indices)

    def _candidates(self, liked, disliked, interested, reactions, filters, want: int) -> np.ndarray:
        """The rows to actually score -- **retrieve-then-rerank**.

        With no ANN index (small catalog / faiss absent) or no taste yet, this is
        the exact filtered set, so behaviour is identical to a full scan. Otherwise
        it's the content-ANN top-K UNION the CF neighbours of the taste set, then
        filtered + de-seen -- a few hundred rows instead of the whole catalog, so the
        blend/MMR downstream stay sublinear. ``want`` sizes the retrieval.

        (``surprise`` deliberately does *not* use this: it needs the whole-catalog
        novelty/score distribution as its gate reference, so it stays a full scan.)
        """
        mask = self._candidate_mask(reactions, filters)
        profile = self._profile(liked, disliked, interested)
        if self.cat.ann is None or profile is None:
            return np.where(mask)[0]
        near = self.cat.ann.search(profile, max(8 * want, 2000))
        cf = self._cf_neighbors(liked, interested)
        cand = np.unique(np.concatenate([near, cf])) if len(cf) else np.unique(near)
        return cand[mask[cand]] if len(cand) else np.array([], dtype=np.int64)

    def _author_keys(self, book_idx: int) -> list[str]:
        """EVERY credited name, normalized -- the keys a book is capped under.

        Capping on the first credit only let an author past their own cap under a
        co-credit: 'Richard Bachman, Stephen King' keyed as 'richard bachman' (the
        pseudonym), and 'Stephen Briggs, Terry Pratchett' as 'stephen briggs', so a
        Discworld fan got the cap twice over. Matching on *any* shared credit closes
        that, and still dedupes 'Rowling, GrandPré' against plain 'Rowling'.
        """
        raw = self.cat.author(book_idx)
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
        tail_frac: float = TAIL_SLOTS_FRAC,
    ) -> list[Scored]:
        """Best-guess recommendations for a 'For You' list.

        Retrieves the top ``REC_POOL_MULT * n`` by relevance, then selects the
        final ``n`` with **MMR + genre calibration** so the list is spread out (not
        ten near-identical fantasy novels) *and* mirrors your taste mix (your
        sci-fi and your romance in proportion, not all of the majority genre),
        capped at ``per_author`` books per author. ``mmr_lambda`` trades relevance
        vs redundancy; ``cal_lambda`` sets how hard genres are matched (0 = off).

        ``tail_frac`` of the list is then **reserved** for books outside the popular
        head, filled by the same MMR from tail candidates only. This is a slot
        allocation rather than a scoring change, because scoring cannot fix it: the
        head and tail compete for the same ``n`` places, so at 100k a stronger head
        model raised head Recall and *halved* tail Recall without the tail model
        changing at all. Set ``tail_frac=0`` to rank purely by score.
        """
        liked, disliked, interested = self._split(reactions)
        cand = self._candidates(
            liked, disliked, interested, reactions, filters, want=REC_POOL_MULT * n
        )
        if len(cand) == 0:
            return []
        scores = self._scores(liked, disliked, interested, cand)
        order = cand[np.argsort(-scores)]

        n_tail = min(round(n * tail_frac), n)
        author_count: dict[str, int] = {}
        picks = self._mmr(
            order[: max(REC_POOL_MULT * n, n)],
            liked,
            disliked,
            interested,
            n - n_tail,
            lam=mmr_lambda,
            per_author=per_author,
            cal_lambda=cal_lambda,
            author_count=author_count,
        )
        if n_tail:
            taken = set(picks)
            tail_order = order[self.is_tail[order]]
            tail_pool = [int(i) for i in tail_order[: REC_POOL_MULT * n] if int(i) not in taken]
            picks += self._mmr(
                tail_pool,
                liked,
                disliked,
                interested,
                n_tail,
                lam=mmr_lambda,
                per_author=per_author,
                cal_lambda=cal_lambda,
                author_count=author_count,
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
        cand = self._candidates(liked, disliked, interested, reactions, filters, want=25 * n)
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
        # Full scan on purpose: surprise gates on the whole-catalog score quantile and
        # ranks by novelty, so a retrieved candidate subset would move the reference.
        # It's the occasional "Surprise me" tab, not the hot swipe path (see §C notes).
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
        subs = self.cat.subjects(i)
        return {s: 1.0 / len(subs) for s in subs} if subs else {}

    def _genre_target(self, liked, interested) -> dict[str, float]:
        """The user's taste genre distribution (likes full weight, interest ALPHA)."""
        subs = [self.cat.subjects(i) for i in list(liked) + list(interested)]
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
        author_count: dict[str, int] | None = None,
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
        # Recomputed on the pool rather than sliced from the caller's scores on purpose:
        # _scores standardizes over whatever set it is given, so scores are SET-RELATIVE.
        # Reusing the candidate-set values here silently changes the relevance/redundancy
        # balance and the resulting list.
        rel_scores = self._scores(liked, disliked, interested, np.array(pool))
        rel = {p: float(rel_scores[i]) for i, p in enumerate(pool)}
        target = self._genre_target(liked, interested) if cal_lambda > 0 else {}
        # Per-book metadata is loop-invariant: author keys and genre mass depend only on
        # the book, but the greedy re-examines every remaining candidate on every one of
        # k passes, so computing them inline meant ~k times more string splits and dict
        # builds than there are books. Hoist them out.
        keys_of = {p: self._author_keys(p) for p in pool}
        mass_of = {p: self._book_genre_mass(p) for p in pool} if cal_lambda > 0 else {}
        sel_mass: dict[str, float] = {}  # running (unnormalized) genre mass of `selected`
        sel_total = 0.0
        selected: list[int] = []
        # Caller may pass a running tally so a second pass (the reserved tail slots)
        # keeps honouring the per-author cap set by the first.
        author_count = {} if author_count is None else author_count
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
                if self._cap_hit(keys_of[p], author_count, per_author):
                    continue
                val = float(base[idx])
                if target:  # genre-calibration penalty for the list-with-p (cheap, per-item)
                    pm = mass_of[p]
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
            self._count_author(keys_of[best], author_count)
            for g, m in mass_of.get(best, {}).items():  # fold into running mass
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
        w = float(self.cf_weight[i])
        score = np.full(len(self.cat), -1e18, dtype=np.float64)
        if self.cat.ann is None:  # exact: score the whole catalog (unchanged)
            content = self.cat.emb @ self.cat.emb[i]
            cf = np.asarray(self.cat.sim.getrow(i).todense()).ravel()
            score = _standardize(content) + w * _standardize_sparse(cf)
            score[i] = -1e18  # never recommend the book itself
            order = np.argsort(-score)
        else:  # retrieve content-ANN + CF neighbours, score just those
            content_cand = self.cat.ann.search(self.cat.emb[i], max(200, 20 * n))
            cf_cand = np.unique(self.cat.sim.getrow(i).indices)
            cand = np.unique(np.concatenate([content_cand, cf_cand]))
            cand = cand[cand != i]
            if len(cand) == 0:
                return []
            content = self.cat.emb[cand] @ self.cat.emb[i]
            cf = np.asarray(self.cat.sim.getrow(i)[:, cand].todense()).ravel()
            score[cand] = _standardize(content) + w * _standardize_sparse(cf)
            order = cand[np.argsort(-score[cand])]
        picks: list[int] = []
        author_count: dict[str, int] = {}
        for jj in order:
            j = int(jj)
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
