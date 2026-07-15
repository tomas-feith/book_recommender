"""Tests for the bulk catalog-expansion selector (scripts/fetch_new_books.py).

Selection logic only -- no network. The diversity quotas are the whole point of
the bulk path and each one has been wrong at least once, so they're pinned here.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_new_books import (  # noqa: E402
    HEAD_ONLY_BUCKETS,
    POPULARITY_BUCKETS,
    _bucket_within_year,
    _Selector,
    _to_record,
)

YEARS = range(2015, 2026)


def _doc(key: str, rl: int, author: str = "A", year: int = 2020) -> dict:
    return {
        "key": f"/works/{key}",
        "title": key,
        "author_name": [author],
        "first_publish_year": year,
        "readinglog_count": rl,
        "subject": ["fiction"],
        "cover_i": 1,
    }


def _corpus(per_subject: int = 30) -> dict:
    """year -> subject -> docs, with readinglog descending within each subject."""
    return {
        y: {
            s: [
                # Authors unique per (year, subject, i), or the per-author cap
                # would fire across years and mask what a test is asserting.
                _doc(f"{s[:2]}{y}_{i}", rl=(per_subject - i) * 10, author=f"{s}{y}{i}", year=y)
                for i in range(per_subject)
            ]
            for s in ("fantasy", "history")
        }
        for y in YEARS
    }


def test_selector_spans_all_years():
    """Regression: a loop variable named `buckets` used to clobber the parameter,
    so every year after the first bucketed against a leftover dict."""
    sel = _Selector(_corpus(), set(), set(), max_per_author=4, seed=0, buckets=POPULARITY_BUCKETS)
    picked = sel.take(110, per_subject_cap=50, log=lambda *a: None)
    years = {int(d["title"][2:6]) for d in picked}
    assert years == set(YEARS), f"expected all 11 years, got {sorted(years)}"


def test_head_only_takes_most_read_first():
    """head_only must return the actual top-read books, not a random sample."""
    sel = _Selector(_corpus(), set(), set(), max_per_author=4, seed=0, buckets=HEAD_ONLY_BUCKETS)
    picked = sel.take(22, per_subject_cap=50, log=lambda *a: None)  # 1/year, 2 subjects
    # Every pick should be its (year, subject) cell's most-read book: rl == 300.
    assert picked, "expected picks"
    assert min(d["readinglog_count"] for d in picked) >= 290


def test_year_quota_is_even():
    sel = _Selector(_corpus(), set(), set(), max_per_author=4, seed=0, buckets=POPULARITY_BUCKETS)
    picked = sel.take(110, per_subject_cap=50, log=lambda *a: None)
    per_year = Counter(int(d["title"][2:6]) for d in picked)
    assert max(per_year.values()) - min(per_year.values()) <= 2, per_year


def test_author_cap_enforced():
    corpus = {
        y: {
            "fantasy": [_doc(f"f{y}_{i}", rl=100 - i, author="Prolific", year=y) for i in range(20)]
        }
        for y in YEARS
    }
    sel = _Selector(corpus, set(), set(), max_per_author=3, seed=0, buckets=POPULARITY_BUCKETS)
    picked = sel.take(50, per_subject_cap=50, log=lambda *a: None)
    assert len(picked) <= 3, f"author cap breached: {len(picked)} picks from one author"


def test_known_ids_and_titles_are_skipped():
    corpus = _corpus(per_subject=5)
    known_ids = {"ol:fa2015_0"}
    sel = _Selector(corpus, known_ids, set(), max_per_author=9, seed=0, buckets=POPULARITY_BUCKETS)
    picked = sel.take(110, per_subject_cap=50, log=lambda *a: None)
    assert all("ol:" + d["key"].rsplit("/", 1)[-1] not in known_ids for d in picked)


def test_bucket_within_year_partitions_by_rank():
    docs = [_doc(f"d{i}", rl=i) for i in range(100)]
    got = _bucket_within_year(docs, POPULARITY_BUCKETS)
    assert Counter(got.values())["head"] == 30
    # The head must be the *most-read* 30, not an arbitrary 30.
    heads = {k for k, v in got.items() if v == "head"}
    assert all(int(k.rsplit("d", 1)[-1]) >= 70 for k in heads)


def test_head_only_puts_everything_in_one_band():
    docs = [_doc(f"d{i}", rl=i) for i in range(50)]
    got = _bucket_within_year(docs, HEAD_ONLY_BUCKETS)
    assert set(got.values()) == {"head"}
    assert len(got) == 50


def test_to_record_keeps_hyphenated_subjects():
    """'science-fiction'/'sci-fi' are the catalog's own tags -- an isalpha() filter
    used to drop exactly those and leave a book tagged only with noise."""
    doc = _doc("x", rl=1)
    doc["subject"] = ["science-fiction", "sci-fi", "New York Times bestseller"]
    rec = _to_record(doc, with_description=False)
    assert "science-fiction" in rec["subjects"]
    assert "sci-fi" in rec["subjects"]


def test_to_record_shape():
    rec = _to_record(_doc("x", rl=1, year=2024), with_description=False)
    assert rec["id"] == "ol:x"
    assert rec["year"] == 2024
    assert rec["language"] == "en"
    assert rec["image"].endswith("-M.jpg")
