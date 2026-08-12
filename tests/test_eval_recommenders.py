"""Tests for the offline eval recommender strategies.

These four share one interface -- ``score(seed, dislikes, cands) -> array`` over
candidate *positions*, with ``rank`` derived from it -- which is what lets them
compete on the same scoreboard. The module had no coverage at all, so nothing
was holding that contract in place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from eval.recommenders import (
    EmbeddingRecommender,
    HybridRecommender,
    ItemItemCFRecommender,
    PopularityRecommender,
    _rank_from_score,
    _standardize,
)

BOOKS = [
    {"id": "b0", "title": "Dune", "author": "Herbert", "subjects": ["sci-fi"], "description": "a"},
    {
        "id": "b1",
        "title": "Messiah",
        "author": "Herbert",
        "subjects": ["sci-fi"],
        "description": "b",
    },
    {"id": "b2", "title": "Emma", "author": "Austen", "subjects": ["romance"], "description": "c"},
    {
        "id": "b3",
        "title": "Persuasion",
        "author": "Austen",
        "subjects": ["romance"],
        "description": "d",
    },
]

# b0/b1 are co-read; b2/b3 are co-read. Popularity is deliberately the reverse of
# the CF structure so the two strategies cannot accidentally agree.
POP = np.array([1.0, 2.0, 30.0, 40.0], dtype=np.float32)
CF_PAIRS = [(0, 1, 0.9), (2, 3, 0.8)]


class FakeEmbedder:
    """Deterministic 4-d embeddings; no torch, no model download."""

    name = "fake"

    def encode(self, texts: list[str]) -> np.ndarray:
        # One-hot per book, so cosine similarity is 1 with itself and 0 otherwise,
        # except b0/b1 which are given a shared component.
        mat = np.eye(len(texts), 4, dtype=np.float32)
        mat[1] = mat[1] + mat[0]  # b1 leans toward b0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / norms


@pytest.fixture
def npz_path(tmp_path: Path) -> Path:
    """A CF/popularity npz in the sparse-CSR layout the real pipeline writes."""
    dense = np.zeros((4, 4), dtype=np.float32)
    for a, b, w in CF_PAIRS:
        dense[a, b] = dense[b, a] = w
    sim = sparse.csr_matrix(dense)
    path = tmp_path / "real_cf.npz"
    np.savez(
        path,
        ids=np.array([b["id"] for b in BOOKS], dtype=object),
        pop=POP,
        sim_data=sim.data,
        sim_indices=sim.indices,
        sim_indptr=sim.indptr,
        sim_shape=np.array(sim.shape),
    )
    return path


# --- popularity -----------------------------------------------------------


def test_popularity_ranks_by_readership(npz_path: Path) -> None:
    rec = PopularityRecommender(npz_path)
    rec.prepare(BOOKS)
    assert rec.rank(seed=[0], dislikes=[], cands=[0, 1, 2, 3]) == [3, 2, 1, 0]


def test_popularity_ignores_the_user(npz_path: Path) -> None:
    """The non-personalized floor: the same list whatever the user liked."""
    rec = PopularityRecommender(npz_path)
    rec.prepare(BOOKS)
    a = rec.score(seed=[0], dislikes=[], cands=[0, 1, 2, 3])
    b = rec.score(seed=[2], dislikes=[1], cands=[0, 1, 2, 3])
    assert np.array_equal(a, b)


# --- item-item CF ---------------------------------------------------------


def test_cf_scores_co_read_neighbours(npz_path: Path) -> None:
    rec = ItemItemCFRecommender(npz_path)
    rec.prepare(BOOKS)
    scores = rec.score(seed=[0], dislikes=[], cands=[1, 2, 3])
    # b1 is co-read with b0; b2/b3 are not connected to it at all.
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(0.0)


def test_cf_subtracts_dislikes_with_beta(npz_path: Path) -> None:
    rec = ItemItemCFRecommender(npz_path, beta=0.5)
    rec.prepare(BOOKS)
    liked_only = rec.score(seed=[0], dislikes=[], cands=[1])
    with_dislike = rec.score(seed=[0], dislikes=[1], cands=[1])
    # b1's own similarity to itself is 0 in this matrix, so disliking b1 removes
    # 0.5 * sim[1, 1] = 0 -- use b3 (co-read with b2) to get a real subtraction.
    assert with_dislike == pytest.approx(liked_only)
    hit = rec.score(seed=[2], dislikes=[2], cands=[3])
    assert hit[0] == pytest.approx(0.8 - 0.5 * 0.8)


def test_cf_alignment_follows_the_books_order(npz_path: Path) -> None:
    """prepare() permutes the npz into the caller's book order, not the file's."""
    reordered = [BOOKS[2], BOOKS[3], BOOKS[0], BOOKS[1]]
    rec = ItemItemCFRecommender(npz_path)
    rec.prepare(reordered)
    # In the reordered list b0/b1 sit at positions 2/3.
    assert rec.score(seed=[2], dislikes=[], cands=[3])[0] == pytest.approx(0.9)
    assert rec.sim.shape == (4, 4)


def test_alignment_rejects_a_book_missing_from_the_npz(npz_path: Path) -> None:
    rec = PopularityRecommender(npz_path)
    with pytest.raises(KeyError):
        rec.prepare([*BOOKS, {"id": "not-in-npz", "title": "x"}])


# --- content --------------------------------------------------------------


def test_embedding_recommender_prefers_the_similar_book() -> None:
    rec = EmbeddingRecommender(FakeEmbedder(), text_mode="no-subjects")
    rec.prepare(BOOKS)
    scores = rec.score(seed=[0], dislikes=[], cands=[1, 2, 3])
    assert scores[0] > scores[1]  # b1 shares a component with b0
    assert scores[0] > scores[2]


def test_embedding_recommender_requires_prepare() -> None:
    rec = EmbeddingRecommender(FakeEmbedder())
    with pytest.raises(AssertionError, match="prepare"):
        rec.score(seed=[0], dislikes=[], cands=[1])


# --- hybrid ---------------------------------------------------------------


def test_hybrid_weight_interpolates_between_its_parts(npz_path: Path) -> None:
    content = EmbeddingRecommender(FakeEmbedder(), text_mode="no-subjects")
    cf = ItemItemCFRecommender(npz_path)
    seed, dislikes, cands = [0], [], [1, 2, 3]

    pure_content = HybridRecommender(content, cf, w_cf=0.0)
    pure_content.prepare(BOOKS)
    pure_cf = HybridRecommender(content, cf, w_cf=1.0)
    half = HybridRecommender(content, cf, w_cf=0.5)

    c = pure_content.score(seed, dislikes, cands)
    f = pure_cf.score(seed, dislikes, cands)
    h = half.score(seed, dislikes, cands)
    assert np.allclose(h, 0.5 * c + 0.5 * f)


def test_hybrid_name_reports_the_weight(npz_path: Path) -> None:
    rec = HybridRecommender(
        EmbeddingRecommender(FakeEmbedder()), ItemItemCFRecommender(npz_path), 0.25
    )
    assert rec.name == "hybrid:25%cf"


# --- shared helpers -------------------------------------------------------


def test_standardize_gives_zero_mean_unit_std() -> None:
    out = _standardize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert out.mean() == pytest.approx(0.0)
    assert out.std() == pytest.approx(1.0)


def test_standardize_survives_a_constant_channel() -> None:
    """std == 0 must not divide by zero -- it centres and stops."""
    out = _standardize(np.full(5, 7.0))
    assert np.allclose(out, 0.0)


def test_rank_returns_candidate_positions_not_indices() -> None:
    ranked = _rank_from_score(np.array([0.1, 0.9, 0.5]), cands=[10, 20, 30])
    assert ranked == [20, 30, 10]


@pytest.mark.parametrize("w_cf", [0.0, 0.5, 1.0])
def test_every_strategy_satisfies_the_shared_contract(npz_path: Path, w_cf: float) -> None:
    """score() returns one value per candidate and rank() is a permutation of them."""
    content = EmbeddingRecommender(FakeEmbedder(), text_mode="no-subjects")
    cf = ItemItemCFRecommender(npz_path)
    strategies = [
        PopularityRecommender(npz_path),
        ItemItemCFRecommender(npz_path),
        content,
        HybridRecommender(content, cf, w_cf=w_cf),
    ]
    cands = [0, 1, 2, 3]
    for rec in strategies:
        rec.prepare(BOOKS)
        scores = rec.score(seed=[0], dislikes=[2], cands=cands)
        assert len(scores) == len(cands), rec
        assert np.isfinite(scores).all(), rec
        assert sorted(rec.rank(seed=[0], dislikes=[2], cands=cands)) == cands, rec
