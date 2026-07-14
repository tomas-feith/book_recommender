"""Ranking metrics."""

from __future__ import annotations

import math

import numpy as np

from eval.metrics import (
    genre_entropy,
    intra_list_distance,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_counts_hits_over_relevant():
    assert recall_at_k(["a", "b", "c", "d"], {"a", "c"}, k=2) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_ndcg_discounts_by_rank():
    # single relevant item at rank 2 -> 1/log2(3), ideal is 1/log2(2)=1
    got = ndcg_at_k(["x", "a", "y"], {"a"}, k=3)
    assert math.isclose(got, (1 / math.log2(3)))


def test_ndcg_empty_relevant_is_zero():
    assert ndcg_at_k(["a"], set(), k=1) == 0.0


def test_mrr_reciprocal_of_first_hit():
    assert mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert mrr(["a"], {"a"}) == 1.0


def test_mrr_no_hit_is_zero():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ild_zero_for_identical_vectors():
    v = np.tile([1.0, 0.0], (3, 1)).astype(np.float32)
    assert intra_list_distance(v) == 0.0


def test_ild_orthogonal_vectors_distance_one():
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert math.isclose(intra_list_distance(v), 1.0, abs_tol=1e-6)


def test_ild_single_or_empty_is_zero():
    assert intra_list_distance(np.array([[1.0, 0.0]], dtype=np.float32)) == 0.0
    assert intra_list_distance(np.empty((0, 2), dtype=np.float32)) == 0.0


def test_genre_entropy_uniform_beats_concentrated():
    varied = genre_entropy([["fantasy"], ["sci-fi"], ["romance"], ["history"]])
    piled = genre_entropy([["fantasy"], ["fantasy"], ["fantasy"], ["romance"]])
    assert varied > piled
    assert math.isclose(varied, 2.0)  # 4 equally-likely genres -> 2 bits


def test_genre_entropy_empty_is_zero():
    assert genre_entropy([[], []]) == 0.0
