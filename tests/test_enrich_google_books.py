"""Tests for the Google Books enricher (scripts/enrich_google_books.py).

No network, no torch: the embedder is stubbed, so these pin the *plumbing*
around it -- which is where the bug was.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
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


# --- 503 handling / quota accounting -------------------------------------------------
#
# The search endpoint 503s intermittently even when Google is healthy (observed
# ~70% of requests during an outage). These pin that one flaky book neither kills
# the run nor silently poisons the cache, and that --limit bounds *requests*.


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))


@pytest.fixture
def books_dir(tmp_path, monkeypatch):
    """A catalog of 30 description-less books, with a scratch cache file."""
    books = [
        {"id": str(i), "title": f"t{i}", "author": "a", "description": "", "subjects": []}
        for i in range(30)
    ]
    (tmp_path / "real_books.json").write_text(json.dumps(books), encoding="utf-8")
    monkeypatch.setattr(enrich_google_books, "CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(enrich_google_books.time, "sleep", lambda *_: None)
    return tmp_path


def test_query_counts_requests_including_retries(monkeypatch):
    """--limit is a quota guard, so a retried lookup must report all 3 requests."""
    monkeypatch.setattr(enrich_google_books.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        enrich_google_books.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(503)),
    )
    with pytest.raises(enrich_google_books.BackendUnavailable) as e:
        enrich_google_books._query("t", "a", retries=3)
    assert e.value.requests == 3


def test_query_does_not_retry_a_429(monkeypatch):
    """A quota 429 is terminal -- retrying it just burns more quota."""
    monkeypatch.setattr(enrich_google_books.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        enrich_google_books.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(429)),
    )
    with pytest.raises(enrich_google_books.QuotaExceeded) as e:
        enrich_google_books._query("t", "a", retries=3)
    assert e.value.requests == 1


def test_one_flaky_book_does_not_abort_the_run(books_dir, monkeypatch):
    """Regression: BackendUnavailable used to `break`, so the first book whose
    retries all 503'd killed the whole run -- filling 0 of 10k books."""

    def flaky(title, author, api_key=None, retries=3):
        if int(title[1:]) % 2:
            raise enrich_google_books.BackendUnavailable("503", 3)
        return {"description": f"d {title}"}, 1

    monkeypatch.setattr(enrich_google_books, "_query", flaky)
    assert enrich_google_books.enrich(data_dir=books_dir, limit=1000, re_embed=False) == 15


def test_a_503_is_not_cached(books_dir, monkeypatch):
    """Caching the failure as {} would make a later healthy run skip the book forever."""

    def flaky(title, author, api_key=None, retries=3):
        if int(title[1:]) % 2:
            raise enrich_google_books.BackendUnavailable("503", 3)
        return {"description": f"d {title}"}, 1

    monkeypatch.setattr(enrich_google_books, "_query", flaky)
    enrich_google_books.enrich(data_dir=books_dir, limit=1000, re_embed=False)
    cache = json.loads((books_dir / "cache.json").read_text())
    assert [k for k in cache if int(k) % 2] == []


def test_limit_bounds_requests_not_books(books_dir, monkeypatch):
    """Regression: limit counted books, so --limit 1000 could send ~3000 requests
    against a 1000/day quota."""
    monkeypatch.setattr(
        enrich_google_books, "_query", lambda t, a, api_key=None, retries=3: ({"description": t}, 3)
    )
    assert enrich_google_books.enrich(data_dir=books_dir, limit=30, re_embed=False) == 10


def test_gives_up_when_the_endpoint_is_down(books_dir, monkeypatch):
    """Skipping forever would spend the whole budget on a dead endpoint, so bail
    after MAX_CONSECUTIVE_503 books in a row fail."""
    seen = []

    def dead(title, author, api_key=None, retries=3):
        seen.append(title)
        raise enrich_google_books.BackendUnavailable("503", 3)

    monkeypatch.setattr(enrich_google_books, "_query", dead)
    assert enrich_google_books.enrich(data_dir=books_dir, limit=1000, re_embed=False) == 0
    assert len(seen) == enrich_google_books.MAX_CONSECUTIVE_503


def test_a_success_resets_the_consecutive_503_counter(books_dir, monkeypatch):
    """Scattered 503s must not accumulate into a false 'endpoint is down' bail."""
    calls = {"n": 0}

    def mostly_flaky(title, author, api_key=None, retries=3):
        calls["n"] += 1
        if calls["n"] % 5:  # 4 of every 5 fail -- never 10 in a row
            raise enrich_google_books.BackendUnavailable("503", 3)
        return {"description": f"d {title}"}, 1

    monkeypatch.setattr(enrich_google_books, "_query", mostly_flaky)
    assert enrich_google_books.enrich(data_dir=books_dir, limit=1000, re_embed=False) == 6
