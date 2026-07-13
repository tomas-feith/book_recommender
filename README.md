# Book recommender — offline evaluation harness

A scoreboard for the recommender core, built **before** any UI. It answers one
question: *given the books a user likes, does the system rank other books they'd
like near the top?* Use it to pick an embedding model and a profile strategy on
real numbers instead of vibes.

## How it works

For each synthetic user we **hold out** `k` of their liked books, build a taste
profile from the remaining signal (likes + dislikes), rank the whole catalog
(minus books they've already reacted to), and check where the held-out likes
land. Metrics are averaged over users and several random hold-out splits.

- **Recall@K** — fraction of held-out likes that made the top K.
- **NDCG@K** — rewards ranking them higher.
- **MRR** — reciprocal rank of the first hit.

## Run it

```bash
pip install -r requirements.txt        # just numpy
python -m eval.run                      # numpy baseline, rocchio profile
python -m eval.run --strategy both      # compare mean vs rocchio
```

Add real models (needs `pip install sentence-transformers`):

```bash
python -m eval.run \
  --model hashing \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model BAAI/bge-m3 \
  --model Qwen/Qwen3-Embedding-0.6B \
  --strategy both
```

The `hashing` baseline is a dependency-free bag-of-words embedder. It has no
semantic understanding — it's the bar real models must clear.

### The `--text-mode` diagnostic

The sample books carry literal genre words (`subjects`, plus genre-loaded
descriptions), and the synthetic users like books strictly within a genre. That
lets a pure keyword matcher win by matching the word "fantasy" — not by
understanding anything. `--text-mode no-subjects` strips the genre words to
expose this:

```bash
python -m eval.run --model hashing --model BAAI/bge-small-en-v1.5 \
    --strategy rocchio --text-mode no-subjects
```

Observed (Recall@5): the hashing baseline collapses (~0.73 → ~0.40) while the
neural models hold (~0.60) and overtake it. Your **real** catalog behaves like
the `no-subjects` column — descriptions don't announce their genre and readers
cross genres — so trust that column when picking a model. Verdict on this data:
`bge-small-en-v1.5` > `MiniLM-L6` > lexical baseline, and Rocchio helps once the
keyword crutch is gone.

## Layout

| File | Role |
|------|------|
| `data/sample_books.json`    | 48 real books across 6 genres (title, author, description, subjects, language, year). |
| `data/sample_profiles.json` | 8 synthetic users with likes + dislikes, incl. two multi-genre users. |
| `eval/data.py`       | Loads data; `book_to_text` decides what text represents a book. |
| `eval/embedders.py`  | `HashingEmbedder` (numpy) + `SentenceTransformerEmbedder` (optional). |
| `eval/profiles.py`   | `mean` and `rocchio` taste-vector builders. |
| `eval/metrics.py`    | Recall@K, NDCG@K, MRR. |
| `eval/run.py`        | CLI that ties it together and prints the scoreboard. |

## Paradigm comparison on real data

`data/real_books.json` / `real_profiles.json` / `real_cf.npz` are built from
goodbooks-10k (10k books, ~6M Goodreads ratings) by `scripts/build_real_dataset.py`:
top-400 books, real reader shelf-tags as genres, Open Library descriptions, and
120 *focused* readers (moderate, opinionated tastes — omnivores who rate a huge
share of the catalog are excluded; they're CF territory). The CF matrix is
learned from the other ~53k users, so it never sees the eval users.

```bash
python -m eval.compare_paradigms
```

Result (Recall@10, warm users with a rating history):

| recommender            | Recall@10 | note |
|------------------------|-----------|------|
| popularity (floor)     | ~0.085    | non-personalized baseline |
| content: bge-small     | ~0.137    | best content model; beats popularity |
| collaborative item-item| **~0.297**| ~2× the best content model |
| hybrid 50/50           | ~0.282    | naive blend; dilutes dominant CF |

**Architectural takeaway.** For warm users, **collaborative filtering wins
decisively (~2×)** — book taste correlations live in co-rating patterns, not in
description text. But content embeddings still beat the popularity floor, and
they are the *only* thing that works for **cold-start**: brand-new books with
zero ratings, new users, and the long tail — exactly the "Tinder" onboarding
regime before a user has swiped enough for CF to engage. So the production
system is a **hybrid that leans on content when CF data is thin and on CF once
it isn't** (ideally weighting per-item by how many ratings a book has). Note
this eval has only popular/warm books, so it *understates* content's cold-start
value — that's CF's home turf by construction.

## Cold-start simulation

`python -m eval.cold_start` marks ~40% of the catalog as newly-added (zeroes
those books out of the CF matrix and popularity counts; content embeddings are
untouched) and measures whether each paradigm can surface a *relevant unrated*
book — the "Tinder" onboarding regime.

| recommender             | Warm books | Cold books (0 ratings) |
|-------------------------|-----------|------------------------|
| popularity              | ~0.085    | **0.000** |
| collaborative item-item | ~0.297    | **0.000** |
| content: bge-small      | ~0.137    | **~0.142** |
| hybrid 50/50            | ~0.282    | ~0.064 |

CF and popularity **cannot recommend an unrated book at all**. Content performs
identically whether or not a book has ratings. Together with the warm-user
comparison this proves the two paradigms are complementary — and that a *static*
hybrid is wrong in both regimes. The production recommender needs an **adaptive
per-item weight**: content-dominated for books with few ratings, CF-dominated as
ratings accumulate.

## The product (MVP)

The `app/` package is the actual "Tinder for books" service, built on the real
data and the validated adaptive-hybrid design. It needs **only numpy at serve
time** — embeddings are precomputed by `scripts/build_embeddings.py`; torch is an
offline-only dependency.

### Environment (uv)

The project uses a uv-managed venv on Python 3.12 (torch + Streamlit both have
wheels there):

```bash
uv sync                                    # create .venv, install deps
uv run python scripts/build_embeddings.py  # once: cache book vectors
uv run streamlit run streamlit_app.py      # the UI at http://localhost:8501
uv run python -m app.demo                  # or the scripted CLI demo
```

### The UI

`streamlit_app.py` is a "Tinder for books" front end over `service.py`: name a
few books you love, then swipe (like / haven't read / pass) while the adaptive
recommender re-ranks after every swipe. Sidebar has language/genre/year filters
and a live taste summary; a "For you" tab shows current recommendations. Serving
needs only numpy (torch is offline-only, for `build_embeddings.py`).

A warm literary theme lives in `.streamlit/config.toml` — cream "paper" light
mode and deep "ink" dark mode (both defined, so the in-app light/dark toggle
works), serif headings (Fraunces) over an Inter body, and pill buttons.

| Module | Role |
|--------|------|
| `app/store.py`       | `Catalog` (books + embeddings + CF matrix + popularity + metadata filters, one aligned index) and `SwipeStore` (users/swipes in SQLite). Split so the store can move to Postgres+pgvector without touching the rest. |
| `app/recommender.py` | Adaptive hybrid: Rocchio profile, content + CF scores, **per-item weight by rating count** (`cf_weight`), MMR diversity, exploit/explore card selection, primary-author dedup. |
| `app/search.py`      | Fuzzy title resolution for the seed step. |
| `app/service.py`     | `BookRecommenderService`: `new_user`, `search_titles`, `seed`, `next_cards`, `swipe`, `recommendations`, filters. The seam a UI/HTTP layer sits on. |
| `app/demo.py`        | Scripted end-to-end session (seed → recommend → swipe → adapt → filter). |

The demo shows real taste generalization: YA-dystopia seeds lead, after a few
swipes, to adult epic fantasy via CF co-reading patterns. Note every card
currently reports `w_cf≈1.0` because the catalog is 400 *popular* books — the
adaptive weight only visibly shifts toward content for long-tail / unrated books
(proven separately in `eval.cold_start`). Feed a long-tail catalog and the
content path lights up on its own.

### Migrating to Postgres + pgvector
`Catalog` is the only piece that changes: replace the numpy embedding/CF search
with SQL (pgvector `<=>` for content, a stored item-item table for CF) and keep
the `filter_mask` conditions as `WHERE` clauses. `SwipeStore` is already
database-shaped. Nothing in `recommender.py` / `service.py` needs to move.

## What this deliberately is *not* (yet)

- **Real catalog.** Swap `sample_books.json` for the Open Library dump loaded
  into Postgres; `load_books` is the only thing that changes.
- **Real users.** Swap `sample_profiles.json` for Goodreads/StoryGraph ratings
  or your own swipe logs.
- **Multi-taste profiles.** A single centroid can't represent someone who likes
  literary fiction *and* hard sci-fi (see the two cross-genre test users). When
  real models still miss those, upgrade `profiles.py` to per-cluster centroids
  or a small per-user classifier — and this harness tells you if it helped.
- **Hard filters** (language / genre / year). Those are structured-column
  filters applied around the vector search, not part of the embedding.
