"""Reserved tail slots: the knob must actually be wired, not shadowed by a default."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from app.recommender import Recommender


class _Cat:
    """Catalog stand-in where the popular head also *wins on score*.

    Head books share an embedding direction, so a profile built from head likes ranks
    the rest of the head top. Without that, every pick is a tail book anyway and the
    reservation has nothing to change -- the fixture would pass for the wrong reason.
    """

    def __init__(self, n=200, n_head=20, dim=8, seed=0):
        rng = np.random.default_rng(seed)
        e = rng.standard_normal((n, dim)).astype(np.float32)
        e[:n_head] = np.array([1.0] + [0.0] * (dim - 1)) + 0.25 * rng.standard_normal((n_head, dim))
        self.emb = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)
        self.pop = np.zeros(n, dtype=np.float32)
        self.pop[:n_head] = 1000.0
        self.sim = sparse.csr_matrix((n, n), dtype=np.float32)
        self.ann = None
        self.books = [{"id": str(i), "title": f"T{i}", "author": f"A{i}"} for i in range(n)]
        self.id_to_idx = {b["id"]: i for i, b in enumerate(self.books)}

    def __len__(self):
        return len(self.books)

    def idx(self, bid):
        return self.id_to_idx[bid]

    def author(self, i):
        return self.books[i]["author"]

    def subjects(self, i):
        return []

    def filter_mask(self, **_):
        return np.ones(len(self.books), dtype=bool)


def _tail_count(picks, rec):
    return sum(1 for s in picks if rec.is_tail[rec.cat.idx(s.book["id"])])


# Pure relevance, no genre calibration: this is a test of slot allocation, and MMR's
# diversification would otherwise pull in tail books on its own and mask the effect.
_PLAIN = {"mmr_lambda": 1.0, "cal_lambda": 0.0}
_REACTIONS = {"0": "like", "1": "like", "2": "like"}


def test_tail_frac_changes_the_list():
    # Guards the default-argument trap: `tail_frac` defaults are bound at def time, so
    # mutating the module constant does nothing. Only a real parameter works.
    rec = Recommender(_Cat())
    none = rec.recommend(_REACTIONS, {}, n=10, tail_frac=0.0, **_PLAIN)
    some = rec.recommend(_REACTIONS, {}, n=10, tail_frac=0.5, **_PLAIN)
    assert len(none) == len(some) == 10
    # The head dominates on score alone (a few tail books still slip in on a small
    # synthetic), and reserving slots must measurably increase tail presence.
    assert _tail_count(none, rec) <= 4
    assert _tail_count(some, rec) >= 5
    assert _tail_count(some, rec) > _tail_count(none, rec)


def test_tail_slots_reserve_the_requested_share():
    rec = Recommender(_Cat())
    picks = rec.recommend(_REACTIONS, {}, n=10, tail_frac=0.3, **_PLAIN)
    assert len(picks) == 10
    assert _tail_count(picks, rec) >= 3


def test_reserved_slots_do_not_duplicate_picks():
    rec = Recommender(_Cat())
    picks = rec.recommend(_REACTIONS, {}, n=10, tail_frac=0.4, **_PLAIN)
    ids = [s.book["id"] for s in picks]
    assert len(set(ids)) == len(ids)


def test_head_frac_marks_the_popular_tenth_as_head():
    rec = Recommender(_Cat())
    assert rec.is_tail.sum() == 180  # 200 books, top 10% by popularity are head
    assert not rec.is_tail[:20].any()
