# Book recommender

Swipe-based book discovery: name a few books you love, then swipe through a
personalized deck — **like / interested / haven't read / pass** — while an
adaptive-hybrid recommender re-ranks after every swipe. The product sits on top of an **offline evaluation
harness** that picked the embedding model and the recommender architecture on
real numbers instead of vibes; that evidence is documented at the bottom.

The serving catalog is **10,000 real books** (all of goodbooks-10k), with real
reader shelf-tags as genres and Open Library descriptions.

## Quick start

The project uses a **uv-managed** environment on Python 3.12 (torch and Streamlit
both have wheels there):

```bash
uv sync                                          # create the venv, install deps
uv run --no-sync python scripts/build_real_dataset.py   # generate data/real_cf.npz (gitignored, ~10MB)
uv run streamlit run streamlit_app.py            # the app at http://localhost:8501
```

The sparse CF matrix (`data/real_cf.npz`) is **gitignored** — it's regenerable,
so build it once with the step above (or `refresh.py`). Serving needs only
**numpy + scipy** (scipy loads the sparse matrix); torch is an *offline-only*
dependency, used to build embeddings, not to serve.

## The app

`streamlit_app.py` is the front end over `app/service.py`:

- **Onboard** — search titles and pick at least three books you love.
- **Discover** — swipe one card at a time: **Like**, **Interested** (soft yes →
  saved to your reading list), **Haven't read** (neutral, just skip), or **Pass**
  (dislike). The taste model updates immediately.
- **For you** — a live grid of best-guess recommendations; **Save** one to your
  reading list or dismiss it as **Not for me** to refine your taste.
- **Surprise me** — wildcards: books *far* from your usual taste that readers
  like you still rate highly (see [Surprise mode](#surprise-mode)).
- **Reading list** — everything you marked **Interested**, in one place.

The sidebar carries your **profile**, language/genre/year **filters**, and a live
taste summary (Liked / Wishlist / Passed / Skipped).

### Durable profiles

Your `user_id` is stored in the URL (`?uid=…`), so a page reload or a server
restart **resumes the same profile** — swipes are persisted in SQLite and are
never lost. Name a profile to save it, and switch between saved profiles from the
sidebar. (New sessions start anonymous until you name them.)

### Reactions and how they score

| Reaction | Meaning | Effect on the model |
|----------|---------|---------------------|
| Like | confident yes | full-weight positive (Rocchio + CF) |
| Interested | soft yes / want to read | positive at weight `α=0.6`, plus reading list |
| Haven't read | can't judge | neutral — excluded from future cards, no signal |
| Pass | dislike | negative at weight `β=0.5` (dislikes are noisier) |

### Theme

A warm literary theme lives in `.streamlit/config.toml` — cream "paper" light
mode and deep "ink" dark mode (both defined, so the in-app toggle works), serif
headings (Fraunces) over an Inter body, pill buttons.

## Architecture

| Module | Role |
|--------|------|
| `app/store.py`       | `Catalog` (books + embeddings + **sparse top-k CF matrix** + popularity + metadata filters, one aligned index) and `SwipeStore` (users/swipes/profiles in SQLite). Split so the store can move to Postgres+pgvector without touching the rest. |
| `app/recommender.py` | Adaptive hybrid: Rocchio profile, content + CF scores, **per-item weight by rating count** (`cf_weight`), MMR diversity, exploit/explore card selection, `surprise()`, author dedup. |
| `app/search.py`      | Fuzzy title resolution for the seed step. |
| `app/service.py`     | `BookRecommenderService`: users/profiles, `seed`, `next_cards`, `swipe`, `recommendations`, `surprises`, `wishlist`, filters. The seam a UI/HTTP layer sits on. |
| `app/demo.py`        | Scripted end-to-end session (seed → recommend → swipe → adapt → filter). |

### The adaptive hybrid

Every candidate is scored on two independent axes and blended **per item**:

- **Content** — cosine of the candidate to your Rocchio taste centroid
  (liked + `α`·interested − `β`·disliked embeddings).
- **CF** — item-item collaborative signal: "readers who liked your books also
  liked this," independent of whether the descriptions resemble each other.

The blend weight is `cf_weight = log(1+pop) / log(1+3000)`, capped at 1: a book
with many ratings is ranked by CF, a thinly-rated / brand-new book by content.
This is the direct answer to the cold-start finding below — a *static* 50/50
blend is wrong in both regimes.

### Surprise mode

`Recommender.surprise()` produces serendipity without abandoning quality. It
**gates** candidates to the top quartile of blended score (so every pick is still
a confident recommendation), then ranks *those* by **novelty** = `1 − cosine to
your nearest liked book`. A book that is both high-scoring and far from your taste
is one the **CF channel** is carrying — nothing like your usual genres, but loved
by readers like you. (It needs some likes to define "your taste," and by
construction it rides CF, so zero-rating cold books can't be surprises.)

## Data pipeline & keeping the catalog fresh

Every book is three aligned artifacts, all keyed by book id:
`data/real_books.json` (metadata), `data/real_embeddings.npz` (content vector),
`data/real_cf.npz` (item-item CF + popularity).

| Script | Purpose |
|--------|---------|
| `scripts/build_real_dataset.py` | Build the whole dataset from goodbooks-10k: top-`N_BOOKS` (=10000) by rating count, reader shelf-tags as genres, Open Library descriptions, 120 focused eval users, and the sparse CF matrix (from ~53k *non-eval* users, so the harness stays honest). |
| `scripts/build_embeddings.py`   | Cache `bge-small-en-v1.5` vectors (10000×384) so serving never loads torch. |
| `scripts/cf_build.py`           | Shared **sparse top-k** CF builder (adjusted-cosine, k=100, block-wise). A dense 10k×10k matrix would be ~370MB; this is ~7MB with no ranking loss. |
| `scripts/fetch_new_books.py`    | Pull genuinely-new books from the **Open Library** search API (by subject, English, recent-year range, ranked by reader count; requires author + cover). Maps to the catalog schema with `ol:`-prefixed ids and dedups against existing ids/titles. |
| `scripts/enrich_google_books.py`| Fill missing descriptions/categories/covers on existing books via the **Google Books API** (~48% of goodbooks entries lack a description), then re-embed only the changed rows. Needs a free `GOOGLE_BOOKS_API_KEY` (the anonymous quota is a shared, usually-exhausted pool). |
| `scripts/fetch_google_books.py` | Add NEW books from the **Google Books API** by subject (`gb:` ids), deduped against the catalog; writes a JSON list for `add_books`/`refresh --add`. The source-side companion to the enricher. Same API key via `.env`. |
| `scripts/add_books.py`          | **Incrementally** append new books to all three artifacts — embeds only the new ones (same model, guarded), grows CF with zero rows so new books start cold (pop=0, content-ranked). Idempotent, atomic. |
| `scripts/refresh.py`            | **Periodic refresh.** Rebuilds CF from *all* accumulated signal — goodbooks ratings **plus the app's own swipe log** (like/interested/dislike → pseudo-ratings 5/4/2) — so engaged books gain collaborative warmth over time and formerly-cold books warm up. `--add PATH` ingests a file first; `--fetch-new N` pulls N new books from Open Library and ingests them in one step. Eval users stay excluded. |
| `scripts/ingest_goodreads_ucsd.py` | **Scale beyond 10k.** Ingest the [UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (~2.3M books, ~876M interactions **with ratings**): streams the gz files, selects top-N by rating count, maps to the schema (`gr:` ids), and builds embeddings + sparse CF from the *real* interactions — so CF stays strong at scale instead of collapsing to cold-start. Download the files once; a single-genre subset is a smaller way to try it. |
| `scripts/ingest_amazon_reviews.py` | **More CF signal.** Ingest [Amazon Reviews 2023 (Books)](https://amazon-reviews-2023.github.io/) — the structural twin of the Goodreads adapter (meta + reviews → sparse CF, `az:` ids). Another huge rated-interactions source. |
| `scripts/ingest_openlibrary_dump.py` | **Breadth (content-only).** Ingest the [Open Library bulk dumps](https://openlibrary.org/developers/dumps) (~30M works, CC0): streams the TSV dump, keeps works with a title + description, resolves authors two-pass, `ol:` ids. No ratings → every book is CF-cold (content-ranked); pair with a ratings source or swipes to grow CF. |

```bash
uv run --no-sync python scripts/build_real_dataset.py   # full rebuild
uv run --no-sync python scripts/build_embeddings.py     # re-embed
uv run --no-sync python scripts/refresh.py --fetch-new 20  # pull new releases + rebuild CF
uv run --no-sync python scripts/refresh.py               # rebuild CF from swipes
```

The design intent: **content carries new books until real usage accrues; the
refresh job then turns that usage into collaborative signal.**

## Why these choices — the evidence

The recommender core was validated *before* the UI, with a held-out ranking
harness (`eval/`): for each synthetic user, hold out `k` liked books, build a
taste profile from the rest, rank the catalog, and measure where the held-out
likes land (Recall@K, NDCG@K, MRR), averaged over users and random splits.

```bash
uv run --no-sync python -m eval.run --strategy both   # mean vs rocchio profile
uv run --no-sync python -m eval.compare_paradigms     # content vs CF vs hybrid
uv run --no-sync python -m eval.cold_start            # the onboarding regime
```

### Paradigm comparison (warm users)

Measured on the **full 10,000-book sparse top-k catalog** (120 users · hold-out=3
· eval@10 · 5 splits/user · 600 trials per recommender):

| recommender             | Recall@10 | NDCG@10 | MRR    | note |
|-------------------------|-----------|---------|--------|------|
| popularity (floor)      | 0.037     | 0.022   | 0.038  | non-personalized baseline |
| content: hashing        | 0.108     | 0.072   | 0.105  | lexical baseline |
| content: bge-small      | 0.114     | 0.083   | 0.125  | best content model; edges hashing |
| collaborative item-item | 0.262     | 0.211   | 0.311  | ~2.3× the best content model |
| **hybrid (adaptive)**   | **0.270** | **0.218** | **0.318** | best overall |

**For warm users, CF wins decisively** — taste correlations live in co-rating
patterns, not description text. At 10k scale the hybrid's edge over pure CF is
thin (0.270 vs 0.262 Recall@10): once popularity is dense, CF carries almost all
the signal and content mainly helps the cold tail. (An earlier *dense*-CF 1,000-
book snapshot showed a wider CF-over-content margin and a naive 50/50 blend that
*diluted* CF — the sparsity of the small catalog exaggerated CF's dominance; the
adaptive per-item weight is what keeps the blend from ever hurting.) Content
remains the *only* thing that works for **cold-start**:

### Cold-start simulation

`eval.cold_start` marks ~40% of the catalog newly-added (zeroed out of CF and
popularity; embeddings untouched) and asks whether each paradigm can surface a
relevant *unrated* book — the onboarding regime. *(Absolute numbers below are
from the earlier dense-CF snapshot; the structural result — hard zeros for CF and
popularity, content unchanged with or without ratings — is what matters and holds
regardless of catalog size.)*

| recommender             | Warm books | Cold books (0 ratings) |
|-------------------------|-----------|------------------------|
| popularity              | ~0.085    | **0.000** |
| collaborative item-item | ~0.297    | **0.000** |
| content: bge-small      | ~0.137    | **~0.142** |
| hybrid 50/50            | ~0.282    | ~0.064 |

CF and popularity **cannot recommend an unrated book at all**; content performs
the same with or without ratings. The two paradigms are complementary, and a
static hybrid is wrong in both regimes — hence the **adaptive per-item weight**.

### The `--text-mode` diagnostic

The sample books carry literal genre words and the synthetic users like strictly
within a genre, which lets a keyword matcher win by matching "fantasy" rather than
understanding anything. `--text-mode no-subjects` strips those words:

```bash
uv run --no-sync python -m eval.run --model hashing \
    --model BAAI/bge-small-en-v1.5 --strategy rocchio --text-mode no-subjects
```

The hashing baseline collapses (~0.73 → ~0.40) while neural models hold (~0.60)
and overtake it. Your real catalog behaves like the `no-subjects` column, so
trust it when picking a model: `bge-small-en-v1.5` > `MiniLM-L6` > lexical, and
Rocchio helps once the keyword crutch is gone.

### Does a *bigger* embedding model help? (measured: no)

`bge-small` (384-dim) is the serving model. We tested whether scaling up the
encoder buys anything, on a fixed shared candidate pool (same pool for every arm,
so cross-model deltas are apples-to-apples):

| model      | dim  | content R@10 | **hybrid R@10** | full-10k embed (6-thread CPU) |
|------------|------|--------------|-----------------|-------------------------------|
| bge-small  | 384  | 0.132        | **0.302**       | ~15 min                       |
| bge-base   | 768  | **0.149**    | 0.297           | ~57 min                       |
| bge-large  | 1024 | 0.140        | 0.302           | ~196 min                      |

**The served hybrid is flat** (0.302 / 0.297 / 0.302 — within noise, barely above
CF-alone at 0.292): the blend is CF-dominated, so a sharper *content* channel gets
washed out. Worse, bigger isn't even monotonic — `bge-large` *regresses* below
`bge-base` on the content arm, so it's strictly dominated (slower **and** less
accurate). Cost scales the wrong way: 4×–13× the offline embed time and 2×–2.7×
the vector storage. `bge-base` does give a real **+13% on the content-only arm**,
which only matters for **cold-start** (where content is the sole signal) — so if
cold-start quality ever becomes the priority, `bge-base` (not `large`) is the one
to re-test on `eval.cold_start`. For the warm hybrid we serve, `bge-small` stays.

### Eval harness layout

| File | Role |
|------|------|
| `data/sample_books.json` / `sample_profiles.json` | 48 books / 8 synthetic users, for the fast keyword-vs-semantic diagnostic. |
| `eval/data.py`      | Loads data; `book_to_text` decides what text represents a book. |
| `eval/embedders.py` | `HashingEmbedder` (numpy) + `SentenceTransformerEmbedder` (optional). |
| `eval/profiles.py`  | `mean` and `rocchio` taste-vector builders. |
| `eval/metrics.py`   | Recall@K, NDCG@K, MRR. |
| `eval/run.py` / `compare_paradigms.py` / `cold_start.py` | The scoreboards. |

## Migrating to Postgres + pgvector

`Catalog` is the only piece that changes: replace the numpy embedding/CF search
with SQL (pgvector `<=>` for content, a stored item-item table for CF) and keep
the `filter_mask` conditions as `WHERE` clauses. `SwipeStore` is already
database-shaped. Nothing in `recommender.py` / `service.py` moves.

## What this deliberately is *not* (yet)

- **Multi-taste profiles.** A single centroid can't perfectly represent someone
  who likes literary fiction *and* hard sci-fi. Per-cluster centroids were built
  and evaluated (held-out Recall@10) and **lost** to the pooled mean — sub-centroids
  from a handful of likes overfit, and no real user's tastes were separable enough
  to help (see git history). A per-user classifier with more signal may still be
  worth trying; the harness will say if it helped.
- **Auth.** Profiles are name-only and URL-resumable; there are no passwords.
- **Scheduling ingestion.** `fetch_new_books.py` + `refresh.py --fetch-new` are the
  live pipeline; running them on a cron/schedule (and expanding beyond Open Library
  to e.g. Google Books) is the remaining operational step.
