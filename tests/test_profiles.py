"""Taste-vector construction and candidate ranking."""

from __future__ import annotations

import numpy as np
import pytest

from eval.profiles import build_profile, rank_candidates


def test_mean_profile_is_normalized_centroid():
    liked = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    prof = build_profile(liked, strategy="mean")
    assert np.isclose(np.linalg.norm(prof), 1.0)
    assert np.allclose(prof, [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)


def test_rocchio_subtracts_dislikes():
    liked = np.array([[1.0, 0.0]], dtype=np.float32)
    disliked = np.array([[0.0, 1.0]], dtype=np.float32)
    prof = build_profile(liked, disliked, strategy="rocchio", beta=0.5)
    # before normalization: (1, -0.5); the second component must be negative.
    assert prof[0] > 0 and prof[1] < 0


def test_mean_strategy_ignores_dislikes():
    liked = np.array([[1.0, 0.0]], dtype=np.float32)
    disliked = np.array([[0.0, 1.0]], dtype=np.float32)
    prof = build_profile(liked, disliked, strategy="mean")
    assert np.allclose(prof, [1.0, 0.0])


def test_empty_liked_raises():
    with pytest.raises(ValueError, match="no liked books"):
        build_profile(np.empty((0, 4), dtype=np.float32))


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown profile strategy"):
        build_profile(np.array([[1.0, 0.0]], dtype=np.float32), strategy="bogus")


def test_rank_candidates_orders_by_cosine():
    profile = np.array([1.0, 0.0], dtype=np.float32)
    catalog = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]], dtype=np.float32)
    assert rank_candidates(profile, catalog, [0, 1, 2]) == [1, 2, 0]
