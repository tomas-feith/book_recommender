"""Tests for the Google Books enricher (scripts/enrich_google_books.py).

No network, no torch: the embedder is stubbed, so these pin the *plumbing*
around it -- which is where the bug was.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import enrich_google_books  # noqa: E402

BOOK = {"title": "T", "author": "A", "description": "D", "subjects": ["fiction"]}


class _StubEmbedder:
    """Records what model it was handed, returns deterministic unit vectors."""

    seen: ClassVar[list[str]] = []

    def __init__(self, model: str):
        _StubEmbedder.seen.append(model)
        self.model = model

    def encode(self, texts):
        v = np.ones((len(texts), 4), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """A 2-book embeddings file tagged with the co-read *label*, plus the dir it names."""
    emb = np.eye(2, 4, dtype=np.float32)
    np.savez_compressed(
        tmp_path / "real_embeddings.npz",
        ids=np.array(["a", "b"]),
        emb=emb,
        model=np.array("coread-finetuned bge-small"),
    )
    (tmp_path / "coread-encoder").mkdir()  # _resolve_model only resolves if it exists
    _StubEmbedder.seen = []
    monkeypatch.setattr("eval.embedders.SentenceTransformerEmbedder", _StubEmbedder, raising=False)
    return tmp_path


def test_reembed_resolves_the_label_to_the_encoder_dir(catalog):
    """Regression: the stored model is a LABEL ('coread-finetuned bge-small').

    Passed to SentenceTransformer verbatim it is read as a HF repo id and raises
    on the space, so this failed on any catalog built with the co-read encoder --
    i.e. the shipped default.
    """
    enrich_google_books._reembed(catalog, {"a": BOOK})
    assert _StubEmbedder.seen, "embedder was never constructed"
    assert _StubEmbedder.seen[0] == str(catalog / "coread-encoder")


def test_reembed_only_touches_changed_rows(catalog):
    enrich_google_books._reembed(catalog, {"a": BOOK})
    with np.load(catalog / "real_embeddings.npz", allow_pickle=True) as z:
        emb, ids = z["emb"], [str(i) for i in z["ids"]]
    assert ids == ["a", "b"]
    assert np.allclose(emb[1], np.eye(2, 4)[1]), "untouched row was modified"
    assert not np.allclose(emb[0], np.eye(2, 4)[0]), "changed row was not re-embedded"


def test_reembed_preserves_the_stored_label(catalog):
    """add_books guards against mixing spaces by comparing this string, so the
    file must keep the label -- not the resolved path."""
    enrich_google_books._reembed(catalog, {"a": BOOK})
    with np.load(catalog / "real_embeddings.npz", allow_pickle=True) as z:
        assert str(z["model"]) == "coread-finetuned bge-small"


def test_reembed_ignores_unknown_ids(catalog):
    """A book that isn't in the embeddings file must not blow up the zip(strict=True)."""
    enrich_google_books._reembed(catalog, {"nope": BOOK})
    with np.load(catalog / "real_embeddings.npz", allow_pickle=True) as z:
        assert np.allclose(z["emb"], np.eye(2, 4))
