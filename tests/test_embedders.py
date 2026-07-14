"""The dependency-free hashing embedder (the semantic one needs torch, not tested here)."""

from __future__ import annotations

import numpy as np

from eval.embedders import HashingEmbedder, build_embedder


def test_shape_and_normalization():
    emb = HashingEmbedder(dim=64)
    out = emb.encode(["hello world", "a b c d"])
    assert out.shape == (2, 64)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_deterministic():
    a = HashingEmbedder(dim=64).encode(["the quick brown fox"])
    b = HashingEmbedder(dim=64).encode(["the quick brown fox"])
    assert np.array_equal(a, b)


def test_shared_vocabulary_raises_similarity():
    emb = HashingEmbedder(dim=256)
    v = emb.encode(["space wizards fight", "space wizards duel", "quiet garden tea"])
    close = float(v[0] @ v[1])
    far = float(v[0] @ v[2])
    assert close > far


def test_empty_text_is_zero_vector():
    out = HashingEmbedder(dim=32).encode([""])
    assert np.allclose(out, 0.0)


def test_build_embedder_returns_hashing_for_hashing_spec():
    assert isinstance(build_embedder("hashing"), HashingEmbedder)
