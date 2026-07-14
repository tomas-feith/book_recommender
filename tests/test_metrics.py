"""Ranking metrics."""

from __future__ import annotations

import math

from eval.metrics import mrr, ndcg_at_k, recall_at_k


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
