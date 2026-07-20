"""Goodreads ingest: language normalization, dedup keys, streaming selection."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ingest_goodreads_ucsd import (
    _dedup_digest,
    build_user_item,
    choose_users,
    norm_language,
    pick_eval_users,
    select_top_book_ids,
    to_profiles,
)


def test_norm_language_unifies_iso_alphabets():
    # Goodreads mixes 639-1 and 639-2 in one field; the filter needs one alphabet.
    assert norm_language("eng") == "en"
    assert norm_language("spa") == "es"
    assert norm_language("dut") == "nl"
    assert norm_language("nl") == "nl"
    assert norm_language("en-US") == "en"
    assert norm_language("en_GB") == "en"


def test_norm_language_falls_back_to_title_script():
    assert norm_language("", "ノルウェイの森") == "ja"
    assert norm_language("", "Преступление и наказание") == "ru"
    assert norm_language("", "The Great Gatsby") == "en"
    assert norm_language("zzz", "三体") == "zh"  # unknown code, script decides


def test_dedup_digest_collapses_editions_not_works():
    same = _dedup_digest("Gone Girl", "Gillian Flynn") == _dedup_digest(
        "Gone Girl (Movie Tie-In Edition)", "Gillian Flynn, Someone Else"
    )
    assert same
    assert _dedup_digest("Gone Girl", "Gillian Flynn") != _dedup_digest(
        "Sharp Objects", "Gillian Flynn"
    )
    assert _dedup_digest("Ulysses", "James Joyce") != _dedup_digest("Ulysses", "Alfred Tennyson")


def test_dedup_digest_needs_both_title_and_author():
    # Precision-first: a wrong merge silently deletes a real book.
    assert _dedup_digest("Some Title", "") is None
    assert _dedup_digest("", "Some Author") is None


def _write_books(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "books.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_select_top_book_ids_dedups_and_keeps_most_rated(tmp_path):
    authors = {"1": "Gillian Flynn", "2": "Frank Herbert"}
    books = _write_books(
        tmp_path,
        [
            {
                "book_id": "a",
                "title": "Gone Girl",
                "ratings_count": 100,
                "authors": [{"author_id": "1"}],
            },
            # same work, more ratings -> this edition must win
            {
                "book_id": "b",
                "title": "Gone Girl (Tie-In)",
                "ratings_count": 900,
                "authors": [{"author_id": "1"}],
            },
            {
                "book_id": "c",
                "title": "Dune",
                "ratings_count": 500,
                "authors": [{"author_id": "2"}],
            },
            {"book_id": "d", "title": "", "ratings_count": 999, "authors": [{"author_id": "2"}]},
        ],
    )
    got = select_top_book_ids(books, top_n=10, authors=authors)
    assert got == {"b", "c"}  # editions collapsed, untitled dropped


def test_select_top_book_ids_respects_top_n(tmp_path):
    authors = {"1": "A"}
    books = _write_books(
        tmp_path,
        [
            {
                "book_id": str(i),
                "title": f"T{i}",
                "ratings_count": i,
                "authors": [{"author_id": "1"}],
            }
            for i in range(10)
        ],
    )
    got = select_top_book_ids(books, top_n=3, authors=authors)
    assert got == {"9", "8", "7"}  # the most-rated three


def test_choose_users_applies_both_caps():
    counts = {"u1": 100, "u2": 50, "u3": 10, "tiny": 1}
    assert "tiny" not in choose_users(counts)  # below MIN_RATED_PER_USER
    assert set(choose_users(counts, max_users=2)) == {"u1", "u2"}  # most active first
    assert set(choose_users(counts, max_interactions=120)) == {"u1"}  # budget stops it


def _write_interactions(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "inter.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def test_build_user_item_is_binary_and_ignores_unknowns(tmp_path):
    inter = _write_interactions(
        tmp_path,
        [
            {"user_id": "u1", "book_id": "1", "rating": 5},
            {"user_id": "u1", "book_id": "1", "rating": 4},  # duplicate -> still 1.0
            {"user_id": "u1", "book_id": "2", "rating": 3},
            {"user_id": "u2", "book_id": "2", "rating": 0},  # unrated -> skipped
            {"user_id": "u9", "book_id": "1", "rating": 5},  # user not chosen
            {"user_id": "u2", "book_id": "999", "rating": 5},  # book not in catalog
        ],
    )
    col_of = {"gr:1": 0, "gr:2": 1}
    X, pop, _ = build_user_item(inter, col_of, {"u1": 0, "u2": 1}, n_items=2)
    assert X.shape == (2, 2)
    assert set(X.data) == {1.0}  # binarized despite the duplicate
    assert pop.tolist() == [1.0, 1.0]


def test_eval_users_are_held_out_of_cf(tmp_path):
    # A held-out user's ratings must never reach the CF matrix -- otherwise the eval
    # scores a recommender that already saw the answers.
    inter = _write_interactions(
        tmp_path,
        [
            {"user_id": "train", "book_id": "1", "rating": 5},
            {"user_id": "held", "book_id": "1", "rating": 5},
            {"user_id": "held", "book_id": "2", "rating": 1},
        ],
    )
    col_of = {"gr:1": 0, "gr:2": 1}
    X, pop, ev = build_user_item(inter, col_of, {"train": 0}, n_items=2, eval_users={"held"})
    assert X.nnz == 1 and pop.tolist() == [1.0, 0.0]  # only the training user counted
    assert ev == {"held": {"gr:1": 5, "gr:2": 1}}  # ...but their ratings were harvested


def test_choose_users_excludes_eval_users():
    counts = {"u1": 100, "u2": 50, "u3": 10}
    assert set(choose_users(counts, exclude={"u1"})) == {"u2", "u3"}


def test_pick_eval_users_wants_moderate_readers():
    counts = {"omnivore": 5000, "moderate": 20, "barely": 2}
    assert pick_eval_users(counts) == {"moderate"}


def test_pick_eval_users_over_selects_then_caps():
    # Pass 1 can't see like-counts, so the pool must exceed the final profile cap --
    # otherwise the like filter in to_profiles silently shrinks the eval set.
    counts = {f"u{i:04d}": 20 for i in range(5000)}
    assert len(pick_eval_users(counts, pool_mult=1)) == 120
    assert len(pick_eval_users(counts)) == 120 * 6


def test_to_profiles_caps_at_max_users():
    ratings = {f"u{i}": {f"gr:{j}": 5 for j in range(8)} for i in range(500)}
    assert len(to_profiles(ratings)) == 120


def test_to_profiles_splits_likes_and_dislikes():
    ratings = {"u": {f"gr:{i}": 5 for i in range(8)} | {"gr:90": 1, "gr:91": 2}}
    (prof,) = to_profiles(ratings)
    assert prof["user"] == "gr_u"
    assert len(prof["likes"]) == 8
    assert sorted(prof["dislikes"]) == ["gr:90", "gr:91"]
    # too little positive signal to build a profile AND hold books out
    assert to_profiles({"thin": {"gr:1": 5, "gr:2": 5}}) == []
