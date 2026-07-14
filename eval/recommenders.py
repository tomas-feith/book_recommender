"""Recommender strategies, unified behind a common interface.

Every recommender implements ``score(seed_likes, dislikes, candidates) -> array``
(higher = better) over candidate *positions* in the books list. ``rank`` derives
from it. Sharing one interface lets content-based, collaborative-filtering,
popularity, and hybrid approaches compete on the identical hold-out scoreboard.

* ``PopularityRecommender`` -- the non-personalized floor. If a fancy model
  can't beat "recommend what's popular," it isn't earning its keep.
* ``ItemItemCFRecommender`` -- collaborative filtering: score a candidate by how
  similar it is (in co-rating patterns) to the books the user liked.
* ``EmbeddingRecommender`` -- content-based: the embedding + profile approach.
* ``HybridRecommender`` -- combines standardized CF and content scores. Content
  covers cold-start (new/obscure books CF has no data for); CF captures taste
  correlations content can't see. Production recommenders are usually hybrid.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .data import book_to_text
from .profiles import build_profile


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-9 else x - x.mean()


class EmbeddingRecommender:
    def __init__(self, embedder, strategy: str = "rocchio", text_mode: str = "full"):
        self.embedder = embedder
        self.strategy = strategy
        self.text_mode = text_mode
        self.name = f"content:{getattr(embedder, 'name', '?')}/{strategy}"
        self._cat = None

    def prepare(self, books: List[dict]) -> None:
        texts = [book_to_text(b, mode=self.text_mode) for b in books]
        self._cat = self.embedder.encode(texts)

    def score(self, seed: Sequence[int], dislikes: Sequence[int], cands: Sequence[int]) -> np.ndarray:
        profile = build_profile(
            self._cat[list(seed)],
            self._cat[list(dislikes)] if len(dislikes) else None,
            strategy=self.strategy,
        )
        return self._cat[np.asarray(cands)] @ profile

    def rank(self, seed, dislikes, cands) -> List[int]:
        return _rank_from_score(self.score(seed, dislikes, cands), cands)


class _NpzBacked:
    """Loads the CF/popularity npz and aligns it to the books-list order."""

    def __init__(self, npz_path: Path):
        self._npz = np.load(npz_path, allow_pickle=True)

    def _align(self, books: List[dict]):
        npz_pos = {str(bid): i for i, bid in enumerate(self._npz["ids"].tolist())}
        self._perm = np.array([npz_pos[b["id"]] for b in books])


class ItemItemCFRecommender(_NpzBacked):
    name = "collaborative:item-item"

    def __init__(self, npz_path: Path, beta: float = 0.5):
        super().__init__(npz_path)
        self.beta = beta

    def prepare(self, books: List[dict]) -> None:
        self._align(books)
        # real_cf.npz now stores a sparse top-k matrix (CSR components). Eval is
        # offline and rare, so densify it here -- keeps scoring and cold_start's
        # row/column zeroing working on a plain 2-D array.
        from scipy import sparse
        z = self._npz
        sim = sparse.csr_matrix(
            (z["sim_data"], z["sim_indices"], z["sim_indptr"]),
            shape=tuple(z["sim_shape"]),
        )
        self.sim = np.asarray(sim[self._perm][:, self._perm].todense()).astype(np.float32)

    def score(self, seed, dislikes, cands) -> np.ndarray:
        # Slice the (few) seed/dislike columns once and sum, then gather the
        # candidate rows -- far cheaper than double-fancy-indexing sim[cands][:,seed].
        cands = np.asarray(cands)
        s = self.sim[:, list(seed)].sum(axis=1)
        if len(dislikes):
            s = s - self.beta * self.sim[:, list(dislikes)].sum(axis=1)
        return s[cands]

    def rank(self, seed, dislikes, cands) -> List[int]:
        return _rank_from_score(self.score(seed, dislikes, cands), cands)


class PopularityRecommender(_NpzBacked):
    name = "popularity"

    def prepare(self, books: List[dict]) -> None:
        self._align(books)
        self.pop = self._npz["pop"][self._perm]

    def score(self, seed, dislikes, cands) -> np.ndarray:
        return self.pop[np.asarray(cands)]

    def rank(self, seed, dislikes, cands) -> List[int]:
        return _rank_from_score(self.score(seed, dislikes, cands), cands)


class HybridRecommender:
    def __init__(self, content: EmbeddingRecommender, cf: ItemItemCFRecommender, w_cf: float = 0.5):
        self.content = content
        self.cf = cf
        self.w_cf = w_cf
        self.name = f"hybrid:{w_cf:.0%}cf"

    def prepare(self, books: List[dict]) -> None:
        self.content.prepare(books)
        self.cf.prepare(books)

    def score(self, seed, dislikes, cands) -> np.ndarray:
        c = _standardize(self.content.score(seed, dislikes, cands))
        f = _standardize(self.cf.score(seed, dislikes, cands))
        return self.w_cf * f + (1.0 - self.w_cf) * c

    def rank(self, seed, dislikes, cands) -> List[int]:
        return _rank_from_score(self.score(seed, dislikes, cands), cands)


def _rank_from_score(scores: np.ndarray, cands: Sequence[int]) -> List[int]:
    cands = np.asarray(cands)
    return cands[np.argsort(-scores)].tolist()
