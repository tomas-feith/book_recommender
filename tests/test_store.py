"""Catalog filters, CF persistence round-trip, and the SQLite swipe store."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from app.store import Catalog, SwipeStore, load_cf, save_cf


def test_save_load_cf_round_trip(tmp_path):
    ids = ["a", "b", "c"]
    dense = np.array([[0, 0.5, 0.0], [0.5, 0, 0.3], [0.0, 0.3, 0]], dtype=np.float32)
    sim = sparse.csr_matrix(dense)
    pop = np.array([10, 20, 30], dtype=np.float32)

    path = tmp_path / "cf.npz"
    save_cf(path, ids, sim, pop)
    got_ids, got_sim, got_pop = load_cf(path)

    assert got_ids == ids
    assert np.allclose(got_sim.toarray(), dense)
    assert np.allclose(got_pop, pop)


def test_save_cf_is_atomic_no_tmp_left(tmp_path):
    path = tmp_path / "cf.npz"
    save_cf(
        path, ["a"], sparse.csr_matrix(np.zeros((1, 1), np.float32)), np.array([1.0], np.float32)
    )
    assert path.exists()
    assert not (tmp_path / "cf.tmp.npz").exists()


def test_filter_mask_language_genre_year(tiny_catalog: Catalog):
    en = tiny_catalog.filter_mask(languages=["en"])
    assert en.sum() == 5 and not en[5]  # b5 is French

    sf = tiny_catalog.filter_mask(genres=["science fiction"])
    assert sf.tolist() == [True, True, True, False, False, False]

    recent = tiny_catalog.filter_mask(year_min=1900)
    assert not recent[3] and recent[2]  # 1813 out, 1984 in


def test_all_genres_ranked_by_frequency(tiny_catalog: Catalog):
    genres = tiny_catalog.all_genres()
    # "science fiction" (3 books) and "romance" (2) lead "cyberpunk"/"fable" (1).
    assert genres[0] == "science fiction"
    assert genres.index("romance") < genres.index("fable")


def test_indices_skips_unknown_ids(tiny_catalog: Catalog):
    assert tiny_catalog.indices(["b1", "nope", "b3"]) == [1, 3]


def _store(tmp_path) -> SwipeStore:
    return SwipeStore(db_path=tmp_path / "app.db")


def test_swipe_record_and_readback(tmp_path):
    store = _store(tmp_path)
    uid = store.create_user("Ada")
    store.record(uid, "b0", "like")
    store.record(uid, "b1", "interested")
    assert store.reactions(uid) == {"b0": "like", "b1": "interested"}
    assert store.seen(uid) == {"b0", "b1"}
    assert store.user_exists(uid)
    store.close()


def test_reswipe_overwrites(tmp_path):
    store = _store(tmp_path)
    uid = store.create_user()
    store.record(uid, "b0", "like")
    store.record(uid, "b0", "dislike")
    assert store.reactions(uid) == {"b0": "dislike"}
    store.close()


def test_invalid_reaction_raises(tmp_path):
    store = _store(tmp_path)
    uid = store.create_user()
    with pytest.raises(ValueError, match="reaction must be one of"):
        store.record(uid, "b0", "love")
    store.close()


def test_named_users_excludes_anonymous_and_web(tmp_path):
    store = _store(tmp_path)
    store.create_user("Ada")
    store.create_user("")  # anonymous
    store.create_user("web")  # server-session sentinel
    names = {u["name"] for u in store.named_users()}
    assert names == {"Ada"}
    store.close()
