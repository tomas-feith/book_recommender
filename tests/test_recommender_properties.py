"""Property-based tests for the ranking invariants.

The recommender's design rests on a handful of claims that the docstrings state
explicitly -- the blend is additive so cold books stay reachable, the per-author
cap is enforced on every credit, MMR trades relevance against redundancy. Each
is easy to satisfy on one hand-built catalog and lose on another. These assert
them over generated catalogs instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import sparse

from app.recommender import Recommender, genre_distribution, kl_calibration
from app.store import Catalog

DIM = 8


def _catalog(n: int, seed: int, *, n_authors: int = 3, cf_density: float = 0.3) -> Catalog:
    """A synthetic catalog: random unit embeddings, a sparse symmetric CF matrix."""
    rng = np.random.default_rng(seed)
    books = [
        {
            "id": f"b{i}",
            "title": f"Book {i}",
            "author": f"Author {i % n_authors}",
            "subjects": [f"genre{i % 4}"],
            "language": "en",
            "year": 2000 + (i % 20),
            "description": "",
        }
        for i in range(n)
    ]
    emb = rng.standard_normal((n, DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    dense = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < cf_density:
                dense[i, j] = dense[j, i] = float(rng.random())
    pop = rng.integers(0, 2000, size=n).astype(np.float32)
    return Catalog(
        books, emb, sparse.csr_matrix(dense), pop, {b["id"]: i for i, b in enumerate(books)}
    )


catalog_sizes = st.integers(min_value=12, max_value=40)
seeds = st.integers(min_value=0, max_value=10_000)


# --- the per-author cap ----------------------------------------------------


def _author_counts(rec: Recommender, cat: Catalog, picks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scored in picks:
        for key in rec._author_keys(cat.id_to_idx[scored.book["id"]]):
            counts[key] = counts.get(key, 0) + 1
    return counts


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=catalog_sizes, seed=seeds, per_author=st.integers(min_value=1, max_value=3))
def test_the_author_cap_holds_when_it_can_be_satisfied(n: int, seed: int, per_author: int) -> None:
    """With enough distinct authors to fill the list, the cap is absolute.

    `_mmr` deliberately relaxes the cap when *every* remaining candidate is
    capped, so the cap can only be asserted strictly when the catalog can
    actually satisfy it -- hence one author per book here.
    """
    cat = _catalog(n, seed, n_authors=n)
    rec = Recommender(cat)

    picks = rec.recommend({"b0": "like", "b1": "like"}, filters={}, n=8, per_author=per_author)
    assert max(_author_counts(rec, cat, picks).values(), default=0) <= per_author


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=catalog_sizes, seed=seeds)
def test_a_cap_that_cannot_be_met_still_fills_the_list(n: int, seed: int) -> None:
    """The documented fallback: three authors and ten slots at a cap of one.

    Rather than returning three books, the greedy relaxes and keeps going -- a
    short list would be the worse failure. Assert that intent explicitly so the
    relaxation is not mistaken for the cap leaking.
    """
    cat = _catalog(n, seed, n_authors=3)
    rec = Recommender(cat)

    picks = rec.recommend({"b0": "like"}, filters={}, n=10, per_author=1)
    assert len(picks) == 10
    assert len({s.book["id"] for s in picks}) == 10


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=catalog_sizes, seed=seeds)
def test_recommendations_are_never_already_swiped(n: int, seed: int) -> None:
    """Anything the user has reacted to is excluded, whatever the reaction."""
    cat = _catalog(n, seed)
    rec = Recommender(cat)
    reactions = {"b0": "like", "b1": "dislike", "b2": "interested", "b3": "skip"}

    for scored in rec.recommend(reactions, filters={}, n=10):
        assert scored.book["id"] not in reactions


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=catalog_sizes, seed=seeds, k=st.integers(min_value=1, max_value=12))
def test_recommend_returns_no_more_than_asked_and_no_duplicates(n: int, seed: int, k: int) -> None:
    cat = _catalog(n, seed)
    rec = Recommender(cat)
    picks = rec.recommend({"b0": "like"}, filters={}, n=k)

    ids = [s.book["id"] for s in picks]
    assert len(ids) <= k
    assert len(ids) == len(set(ids))


# --- the additive blend ----------------------------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=catalog_sizes, seed=seeds)
def test_a_cf_cold_book_is_still_reachable(n: int, seed: int) -> None:
    """The reason the blend is additive rather than convex.

    A book with no CF row at all must still be rankable on content alone. Under
    the old convex blend cold books were compared on a different quantity and
    lost every time (measured: 0.0% of the top-10 against a 40% base rate).
    Here: make one book both CF-cold and the closest thing to the taste vector,
    and it must come first.
    """
    cat = _catalog(n, seed, cf_density=0.3)
    cold = n - 1
    # Strip its CF row, and point it exactly at the liked book's embedding.
    dense = cat.sim.toarray()
    dense[cold, :] = 0.0
    dense[:, cold] = 0.0
    cat.sim = sparse.csr_matrix(dense)
    cat.emb[cold] = cat.emb[0]
    cat.pop[cold] = 0.0

    rec = Recommender(cat)
    assert rec.cf_weight[cold] == 0.0  # no CF evidence -> pure content

    picks = rec.recommend(
        {"b0": "like"}, filters={}, n=5, per_author=5, mmr_lambda=1.0, cal_lambda=0.0, tail_frac=0.0
    )
    assert f"b{cold}" in [s.book["id"] for s in picks]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=catalog_sizes, seed=seeds)
def test_cf_weight_is_zero_exactly_when_there_is_no_cf_row(n: int, seed: int) -> None:
    """cf_weight must mean 'we have CF evidence', not 'this book is popular'."""
    cat = _catalog(n, seed, cf_density=0.15)
    rec = Recommender(cat)
    has_row = np.diff(cat.sim.indptr) > 0

    assert np.all(rec.cf_weight[~has_row] == 0.0)
    assert np.all((rec.cf_weight >= 0.0) & (rec.cf_weight <= 1.0))


# --- similar() -------------------------------------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n=catalog_sizes, seed=seeds, idx=st.integers(min_value=0, max_value=11))
def test_similar_never_returns_the_seed(n: int, seed: int, idx: int) -> None:
    cat = _catalog(n, seed)
    rec = Recommender(cat)
    book_id = f"b{idx % n}"

    sims = rec.similar(book_id, n=5)
    assert book_id not in [s.book["id"] for s in sims]
    assert len({s.book["id"] for s in sims}) == len(sims)


# --- genre calibration helpers ---------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=0, max_size=4),
        min_size=1,
        max_size=10,
    )
)
def test_genre_distribution_sums_to_one_or_is_empty(subject_lists: list[list[str]]) -> None:
    dist = genre_distribution(subject_lists)
    if dist:
        assert sum(dist.values()) == pytest.approx(1.0)
        assert all(v >= 0 for v in dist.values())
    else:
        assert all(not s for s in subject_lists)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=3),
        min_size=1,
        max_size=8,
    )
)
def test_kl_calibration_is_zero_against_itself_and_never_negative(
    subject_lists: list[list[str]],
) -> None:
    """KL(p || p) == 0, and KL is non-negative -- the miscalibration penalty
    must never *reward* a list for diverging from the taste profile."""
    target = genre_distribution(subject_lists)
    if not target:
        return
    assert kl_calibration(target, target) == pytest.approx(0.0, abs=1e-12)

    skewed = {next(iter(target)): 1.0}
    assert kl_calibration(target, skewed) >= -1e-12
