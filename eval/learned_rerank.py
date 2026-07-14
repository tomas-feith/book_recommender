"""Does a LEARNED blend beat the hand-tuned adaptive-hybrid formula?

The served score is a hand-tuned per-item blend of content and CF
(``cf_weight * std(cf) + (1-cf_weight) * std(content)``). This trains a small
logistic ranker on real goodbooks interactions (NON-eval users, so the 120 eval
profiles stay held out) over richer per-(user,item) features -- content cosine, CF
strength, cf_weight, popularity, and their interactions -- and checks whether it
ranks held-out likes better than the formula. "Measure, don't guess": if it wins
we wire it in; if not, the formula stays.

Run (needs the real dataset built + cached goodbooks ratings):
    uv run --no-sync python -m eval.learned_rerank
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import numpy as np

from app.recommender import POP_REF
from app.store import Catalog
from eval.data import load_profiles

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE = ROOT / "data", ROOT / ".cache"
K_HOLDOUT, K_EVAL, SEEDS = 3, 10, list(range(5))
N_TRAIN_USERS, N_NEG = 4000, 30


def _std(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x - x.mean()


def features(cat: Catalog, cf_weight, seed_idx, cand_idx) -> np.ndarray:
    """Per-(user, candidate) features -- raw/absolute so train and eval agree."""
    prof = cat.emb[seed_idx].mean(axis=0)
    nrm = np.linalg.norm(prof)
    prof = prof / nrm if nrm else prof
    content = cat.emb[cand_idx] @ prof
    cf = np.asarray(cat.sim[:, seed_idx].sum(axis=1)).ravel()[cand_idx] / max(len(seed_idx), 1)
    w = cf_weight[cand_idx]
    logpop = np.log1p(cat.pop[cand_idx])
    return np.column_stack([content, cf, w, logpop, content * w, cf * w]).astype(np.float64)


def formula_scores(cat: Catalog, cf_weight, seed_idx, cand_idx) -> np.ndarray:
    """The current served blend, for the same candidates (the baseline)."""
    prof = cat.emb[seed_idx].mean(axis=0)
    nrm = np.linalg.norm(prof)
    prof = prof / nrm if nrm else prof
    content = cat.emb[cand_idx] @ prof
    cf = np.asarray(cat.sim[:, seed_idx].sum(axis=1)).ravel()[cand_idx]
    w = cf_weight[cand_idx]
    return w * _std(cf) + (1 - w) * _std(content)


def load_likes_by_user(order_set, eval_uids):
    by_user: dict[str, list[str]] = {}
    reader = csv.DictReader(io.StringIO((CACHE / "gb_ratings.csv").read_text(encoding="utf-8")))
    for r in reader:
        if r["user_id"] in eval_uids or r["book_id"] not in order_set:
            continue
        if int(r["rating"]) >= 4:
            by_user.setdefault(r["user_id"], []).append(r["book_id"])
    return by_user


def train(cat, cf_weight, likes_by_user):
    """Logistic regression on (held-out like = 1) vs (sampled non-like = 0)."""
    idx = cat.id_to_idx
    rng = random.Random(0)
    users = [u for u, ls in likes_by_user.items() if len(ls) > K_HOLDOUT]
    rng.shuffle(users)
    X, y = [], []
    n_books = len(cat.books)
    for u in users[:N_TRAIN_USERS]:
        liked = [idx[b] for b in likes_by_user[u] if b in idx]
        if len(liked) <= K_HOLDOUT:
            continue
        held = rng.sample(liked, 1)
        seed = [i for i in liked if i not in held]
        reacted = set(liked)
        negs: list[int] = []
        while len(negs) < N_NEG:
            c = rng.randrange(n_books)
            if c not in reacted:
                negs.append(c)
        cand = np.array(held + negs)
        X.append(features(cat, cf_weight, seed, cand))
        lab = np.zeros(len(cand))
        lab[0] = 1.0
        y.append(lab)
    Xm = np.vstack(X)
    yv = np.concatenate(y)
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
    Xs = (Xm - mu) / sd
    Xs = np.column_stack([Xs, np.ones(len(Xs))])  # bias
    w = np.zeros(Xs.shape[1])
    lr, lam = 0.5, 1e-4
    for _ in range(300):  # gradient descent
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        w -= lr * (Xs.T @ (p - yv) / len(yv) + lam * w)
    print(f"  trained on {len(users[:N_TRAIN_USERS])} users, {len(yv)} examples", flush=True)
    return w, mu, sd


def recall_at_k(scores, cand, held, k=K_EVAL):
    top = {int(cand[j]) for j in np.argsort(-scores)[:k]}
    return len(held & top) / min(len(held), k)


def main() -> None:
    cat = Catalog.load(DATA)
    cf_weight = np.clip(np.log1p(cat.pop) / np.log1p(POP_REF), 0.0, 1.0).astype(np.float32)
    profiles = load_profiles(DATA / "real_profiles.json")
    eval_uids = {p["user"].removeprefix("gr_") for p in profiles}

    print("Loading non-eval goodbooks likes + training the reranker...", flush=True)
    likes_by_user = load_likes_by_user(set(cat.id_to_idx), eval_uids)
    w, mu, sd = train(cat, cf_weight, likes_by_user)

    n = len(cat.books)
    rc_formula = rc_learned = 0.0
    trials = 0
    for prof in profiles:
        liked_all = [cat.idx(b) for b in prof["likes"] if b in cat.id_to_idx]
        if len(liked_all) <= K_HOLDOUT:
            continue
        for seed in SEEDS:
            rng = random.Random(seed)
            held = set(rng.sample(liked_all, K_HOLDOUT))
            seed_idx = [i for i in liked_all if i not in held]
            cand = np.array([i for i in range(n) if i not in set(seed_idx)])
            f = features(cat, cf_weight, seed_idx, cand)
            learned = ((f - mu) / sd) @ w[:-1] + w[-1]
            baseline = formula_scores(cat, cf_weight, seed_idx, cand)
            rc_learned += recall_at_k(learned, cand, held)
            rc_formula += recall_at_k(baseline, cand, held)
            trials += 1

    print(f"\nHeld-out Recall@10 over {trials} trials:")
    print(f"  adaptive-hybrid formula (served): {rc_formula / trials:.4f}")
    print(f"  learned logistic reranker:        {rc_learned / trials:.4f}")
    delta = (rc_learned - rc_formula) / trials
    verdict = (
        "WINS -> worth wiring in" if delta > 0.002 else "no meaningful gain -> keep the formula"
    )
    print(f"  delta: {delta:+.4f}  ({verdict})")
    print("feature weights:", np.round(w, 3))


if __name__ == "__main__":
    main()
