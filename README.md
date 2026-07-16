# Book recommender

[![CI](https://github.com/tomas-feith/book_recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/tomas-feith/book_recommender/actions/workflows/ci.yml)

Swipe-based book discovery: name a few books you love, then swipe through a
personalized deck — **like / interested / haven't read / pass** — while an
adaptive-hybrid recommender re-ranks after every swipe. The product sits on top of an **offline evaluation
harness** that picked the embedding model and the recommender architecture on
real numbers instead of vibes; that evidence is documented at the bottom.

The serving catalog is **~22,630 real books**: all of goodbooks-10k (real reader
shelf-tags as genres, Open Library descriptions) plus **~12,600 modern books
(2015-2025)** pulled from Open Library, since goodbooks is frozen at 2017. The
modern half is deliberately spread — an equal quota per publication year, ~40
subjects across fiction and nonfiction, a head/mid/tail popularity mix by
*within-year* percentile, and a per-author cap (see `fetch_new_books.py
--diverse`). It's ingested CF-cold via `scripts/add_books.py`, so new books are
ranked by the co-read content encoder until they accrue reactions. Seed lists
live in `data/` and are replayed with `refresh.py --add` after a rebuild.

## Quick start

The project uses a **uv-managed** environment on Python 3.12 (torch and Streamlit
both have wheels there):

```bash
uv sync                                                 # create the venv, install deps
uv run --no-sync python scripts/build_real_dataset.py   # goodbooks-10k -> books, profiles, CF
uv run --no-sync python scripts/build_embeddings.py     # content vectors (needs torch)
just serve                                              # the app at http://localhost:8501
```

`just serve` falls forward to the next free port if 8501 is already taken, and
prints the URL it settled on.

The serving artifacts in `data/` are **gitignored** — they're build outputs, so a
fresh clone builds them once with the steps above. Only the things that *can't*
be regenerated are committed (the demo fixtures and the Open Library seed lists);
see [Building the data](#building-the-data) for the full pipeline, including the
modern catalog and the co-read encoder.

Serving needs only **numpy + scipy** (scipy loads the sparse matrix); torch is an
*offline-only* dependency, used to build embeddings, not to serve.

## The app

`streamlit_app.py` is the front end over `app/service.py`:

- **Onboard** — search by **title or author** (translated works resolve by their
  English name *or* original, e.g. *The Three-Body Problem* / `三体`), or **search
  by meaning** ("a lonely lighthouse keeper") via the embeddings, or **import your
  reading list** (CSV / TSV / TXT / XLSX, e.g. a Goodreads export). If a book
  **isn't in the catalog**, look it up on **Open Library** and add it on the fly —
  it's ingested CF-cold (content-ranked) so it joins your taste profile immediately.
- **Discover** — swipe one card at a time: **Like**, **Interested** (soft yes →
  saved to your reading list), **Haven't read** (neutral, just skip), or **Pass**
  (dislike). The taste model updates immediately.
- **For you** — a live grid of best-guess recommendations, **diversified with MMR +
  genre calibration** (see [Diversity](#diversity-the-relevancediversity-frontier)),
  each with a **"why recommended"** line and a **"More like this"** popover; **Save**
  to your reading list or dismiss as **Not for me**.
- **Surprise me** — wildcards: books *far* from your usual taste that readers
  like you still rate highly (see [Surprise mode](#surprise-mode)).
- **Reading list** — everything you marked **Interested**: mark a book **read +
  liked**, remove it, get **More like this**, or **export the list as CSV**.

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
| `app/store.py`       | `Catalog` (books + fp16 embeddings + **sparse top-k CF matrix** + popularity + **vectorized metadata filters**, one aligned index) and `SwipeStore` (users/swipes/profiles in SQLite). Split so the store can move to Postgres+pgvector without touching the rest. |
| `app/recommender.py` | Adaptive hybrid: Rocchio profile, content + CF scores, **per-item weight by rating count** (`cf_weight`); list assembly with **MMR + genre calibration**, exploit/explore, `surprise()`, `similar()` ("more like this"), and per-pick explanations. |
| `app/search.py`      | Fuzzy **title + author** resolution for the seed step: English display titles with the original-language name (`三体`) kept as a searchable alias, and a popularity tiebreak so the canonical edition surfaces first. |
| `app/library.py`     | Parse an uploaded reading list (CSV/TSV/TXT/XLSX) into `(title, author)` entries. |
| `app/external.py`    | On-demand Open Library lookup for books not in the catalog (urllib-only, no torch). |
| `app/service.py`     | `BookRecommenderService`: users/profiles, `seed`, `next_cards`, `swipe`, `recommendations`, `surprises`, `wishlist`, `semantic_search`, `similar_books`, `import_library`, filters. The seam a UI/HTTP layer sits on. |
| `app/demo.py`        | Scripted end-to-end session (seed → recommend → swipe → adapt → filter). |

### The adaptive hybrid

Every candidate is scored on two independent axes and blended **per item**:

- **Content** — cosine of the candidate to your Rocchio taste centroid
  (liked + `α`·interested − `β`·disliked embeddings).
- **CF** — item-item collaborative signal (**EASE-R**): "readers who liked your
  books also liked this," independent of whether the descriptions resemble each
  other.

The blend weight is `cf_weight = log(1+pop) / log(1+500)`, capped at 1: a book
with many ratings is ranked by CF, a thinly-rated / brand-new book by content.
The reference (500) is tuned for the EASE-R core — measured to beat content at
*every* popularity tier down to 8 ratings, so the blend leans on CF quickly; a
brand-new book (pop=0) still gets `cf_weight=0`, i.e. pure content. This is the
direct answer to the cold-start finding below — a *static* 50/50 blend is wrong
in both regimes.

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
`data/real_books.json` (metadata), `data/real_embeddings.npz` (content vector,
**stored fp16** — half the file/load), `data/real_cf.npz` (item-item CF + popularity).

> **On fp16 embeddings.** Vectors are stored fp16 (halves the file and load
> bandwidth) but upcast to **fp32 in RAM** for serving: numpy has no fp16 GEMV on
> CPU, so an fp16-resident matrix would upcast on *every* query (~9× slower here,
> measured). fp16 ranking is accuracy-neutral (identical Recall@10). The *query*-
> bandwidth win from fp16 needs an fp16-native index (FAISS / pgvector) — which is
> the same scale-out step as [approximate NN](#migrating-to-postgres--pgvector);
> fp16 storage is the prerequisite.

| Script | Purpose |
|--------|---------|
| `scripts/build_real_dataset.py` | Build the whole dataset from goodbooks-10k: top-`N_BOOKS` (=10000) by rating count, reader shelf-tags as genres, Open Library descriptions, 120 focused eval users, and the sparse CF matrix (from ~53k *non-eval* users, so the harness stays honest). |
| `scripts/build_embeddings.py`   | Cache the content vectors (10000×384) so serving never loads torch. Uses the **co-read fine-tuned** encoder (`data/coread-encoder`) when present, else stock `bge-small`. |
| `scripts/finetune_coread.py`    | **Cold-start fine-tune.** Distill EASE's co-read structure INTO the content encoder: build (anchor, positive) pairs from each book's top EASE neighbors and fine-tune `bge-small` with an in-batch contrastive (InfoNCE) objective. Gives unrated books a collaborative-aware embedding EASE can't (see evidence below). Writes `data/coread-encoder`; then re-run `build_embeddings.py`. |
| `scripts/cf_build.py`           | CF matrix builders, both emitting the same sparse top-k format. Default is **EASE-R** (`ease_cf`: closed-form `B=-P/diag(P)`, `P=(XᵀX+λI)⁻¹`, λ=1000, top-50) — **+35% Recall@10** over the older adjusted-cosine KNN (`sparse_topk_cf`, kept as a fallback). Truncated to ~4MB vs a ~370MB dense matrix, with no ranking loss. |
| `scripts/fetch_new_books.py`    | Pull genuinely-new books from the **Open Library** search API. Maps to the catalog schema with `ol:`-prefixed ids and dedups against existing ids/titles. `--fetch-new N` (via `refresh.py`) tops up by tens. **`--diverse`** is the bulk path (thousands): it partitions the search into (subject × year) cells and selects against explicit quotas — an equal **year** quota (OL holds far more well-read 2015 books than 2025 ones, so pooling years returns a backlist), head/mid/tail **popularity** by *within-year* percentile (readinglog accumulates with age — absolute cutoffs file every recent book as tail), ~40 **subjects** across fiction and nonfiction, and a per-**author** cap. `--head-only` takes the most-read books per cell instead, to top up the marquee titles. |
| `scripts/refresh_subjects.py`   | **Unify the genre vocabulary.** goodbooks tags are Goodreads shelves (`science-fiction`, `young-adult`); Open Library's are library headings (`science fiction`, `juvenile fiction`) — the same genre as a different string, so they never meet. That's not just filtering: the recommender calibrates against the user's genre distribution, so an OL sci-fi book scored zero against a taste vector saying `science-fiction`. Re-fetches raw subjects (batched key queries, ~100 works/request) and normalizes them *toward* the goodbooks vocabulary, then re-embeds only the changed rows. |
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

### Building the data

`data/` is gitignored apart from what cannot be regenerated, so a fresh clone
builds its own artifacts. What's committed, and why:

| Committed | Why it can't be regenerated |
|-----------|------------------------------|
| `data/sample_books.json`, `data/sample_profiles.json` | Hand-curated demo fixtures (48 books / 8 profiles). No generator exists — they're source, and `python -m eval.run` reads them by default. |
| `data/recent_books.json`, `data/expansion_10k.json`, `data/topup_head.json` | **Open Library snapshots.** OL is a live catalog, so re-running the fetcher returns *different* books — never these. Keeping the seed lists is the only way to rebuild the same serving catalog. |

Everything else is a build output: `real_books.json`, `real_profiles.json`,
`real_embeddings.npz`, `real_cf.npz`, `coread-encoder/`, and the runtime
`app.db`. Full rebuild, in order:

```bash
uv sync

# 1. goodbooks-10k -> data/real_books.json + real_profiles.json + real_cf.npz.
#    Downloads are cached, so re-runs are fast.
uv run --no-sync python scripts/build_real_dataset.py

# 2. OPTIONAL but shipped: fine-tune the content encoder on EASE co-read pairs
#    -> data/coread-encoder (~130MB). Needs step 1's CF matrix. Skip it and
#    step 3 falls back to stock bge-small (measurably worse at cold-start).
uv run --no-sync python scripts/finetune_coread.py

# 3. content vectors -> data/real_embeddings.npz (uses the encoder from step 2)
uv run --no-sync python scripts/build_embeddings.py

# 4. the modern catalog: replay the Open Library seeds (goodbooks stops at 2017).
#    --no-cf because these books arrive CF-cold by design; they carry no ratings.
uv run --no-sync python scripts/refresh.py --add data/recent_books.json --no-cf
uv run --no-sync python scripts/refresh.py --add data/expansion_10k.json --no-cf
uv run --no-sync python scripts/refresh.py --add data/topup_head.json --no-cf

# 5. normalize the Open Library books' genre tags onto the goodbooks vocabulary
#    (re-embeds the rows it changes)
uv run --no-sync python scripts/refresh_subjects.py
```

To pull a *fresh* modern batch instead of replaying the seeds — a different set
of books, by design — see `fetch_new_books.py --diverse` above, then ingest the
JSON it writes with `refresh.py --add`.

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
| content: hashing        | 0.107     | 0.074   | 0.111  | lexical baseline |
| content: bge-small      | 0.114     | 0.083   | 0.125  | best content model; edges hashing |
| **collaborative: EASE-R** | **0.351** | 0.279 | 0.380 | closed-form item-item; the CF core |
| hybrid (static 50/50)   | 0.344     | 0.280   | 0.388  | content slightly *dilutes* strong CF |

**For warm users, CF wins decisively** — taste correlations live in co-rating
patterns, not description text. The CF core is **EASE-R** (a closed-form
regularized item-item auto-encoder), which measured **+35% Recall@10 over the
adjusted-cosine KNN it replaced** (0.262 → 0.351; see git history for the
ablation). It's so strong that for *warm* users content slightly dilutes it — but
content is still the **only** signal for **cold-start**, so the served recommender
keeps the adaptive per-item blend (swapping the KNN core for EASE lifted the
served adaptive hybrid 0.275 → 0.333). Content remains the *only* thing that works
when there are no ratings at all:

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
| collaborative (item-item/EASE) | ~0.30 | **0.000** |
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
the vector storage. A bigger *stock* encoder is the wrong lever — the right one
for cold-start turned out to be **fine-tuning** the small model on collaborative
signal (next).

### Collaborative-aware content for cold-start (measured, shipped)

EASE-R is *silent* on an unrated book — an all-zero row, so it can't rank a
brand-new / never-rated book at all. Content is the only signal there. So we
**distilled EASE's co-read structure into the content encoder** (`bge-small`,
in-batch InfoNCE on top-EASE-neighbor pairs; `scripts/finetune_coread.py`), so a
book lands near its would-be co-read neighbors *from text alone*.

The honest test is **leakage-free**: mark ~40% of books cold (held out of *both*
the CF matrix and the training pairs), then rank held-out likes by content only.

| held-out target | base bge-small | co-read fine-tuned | Δ |
|-----------------|----------------|--------------------|---|
| **cold** (model never trained on these) | 0.147 | **0.166** | **+12%** |
| warm (contrast — trained on)            | 0.121 | 0.151 | +25% |

The **+12% on cold books the model never saw** is genuine generalization — a
capability EASE structurally cannot have. (The larger warm gain is the *redundant*
part: books EASE already handles at serving, so it doesn't count.) The win is real
but **narrow**: it only helps the content channel, i.e. onboarding and brand-new
catalog additions; warm users are CF-dominated and unaffected. Serving uses the
fine-tuned encoder's vectors — same 384-dim, same numpy-only serving path.

### Diversity: the relevance↔diversity frontier

You don't want ten near-identical fantasy novels (or all seven Harry Potters), and
if you like fantasy *and* romance you want both — in proportion. The "For You" list
is selected greedily to maximize, per pick,

    λ · relevance − (1 − λ) · max-similarity-to-already-picked − cal · KL(taste ‖ list-genres)

over the top `REC_POOL_MULT·n` candidates, capped per author. The three terms:
**relevance**, an **MMR** redundancy penalty (`mmr_lambda`; exact set-diversity is
NP-hard so greedy is the standard approximation), and **genre calibration** (Steck):
`cal_lambda` pulls the list's genre mix toward the user's taste mix via KL
divergence, so a minority taste isn't drowned out by the majority one. A
single-author saga is collapsed by the author cap; cross-author near-duplicates by
the similarity penalty; taste *coverage* by calibration.

`eval.diversity` sweeps both knobs over the real profiles, measuring Recall@10
against **intra-list distance** (ILD), **genre entropy**, **miscalibration KL**
(list vs. taste genre mix; lower = better), and catalog **coverage**:

| `mmr` | `cal` | Recall@10 | ILD | genre-H | miscalKL | coverage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.0 | 0.0 | 0.341 | 0.500 | 4.09 | 1.550 | 0.052 |
| 0.5 | 0.0 | 0.338 | 0.514 | 4.14 | 1.515 | 0.053 |
| **0.5** | **0.4** | **0.341** | 0.517 | 4.16 | **1.454** | 0.053 |
| 0.5 | 0.8 | 0.338 | 0.523 | 4.19 | 1.388 | 0.054 |

**Both diversity and calibration are ~free here.** Dropping `mmr` to 0.3 costs ~2%
Recall for +7% ILD; adding `cal=0.4` *lowers* miscalibration (1.515 → 1.454) while
nudging Recall *up*. Defaults `MMR_LAMBDA=0.5`, `CAL_LAMBDA=0.4` sit on the cheap
part of both curves; `mmr_lambda` is a natural "focused ↔ eclectic" control.

> Calibration is the answer to the *multi-taste* problem a single Rocchio centroid
> can't solve at scoring time (per-cluster profiles were tried and lost — see below):
> instead of splitting the taste vector, we let scoring stay pooled and fix the
> genre *balance* at list-assembly time. Cheaper, and it measurably works.

### Eval harness layout

| File | Role |
|------|------|
| `data/sample_books.json` / `sample_profiles.json` | 48 books / 8 synthetic users, for the fast keyword-vs-semantic diagnostic. |
| `eval/data.py`      | Loads data; `book_to_text` decides what text represents a book. |
| `eval/embedders.py` | `HashingEmbedder` (numpy) + `SentenceTransformerEmbedder` (optional). |
| `eval/profiles.py`  | `mean` and `rocchio` taste-vector builders. |
| `eval/metrics.py`   | Recall@K, NDCG@K, MRR. |
| `eval/run.py` / `compare_paradigms.py` / `cold_start.py` / `diversity.py` / `learned_rerank.py` | The scoreboards (ranking paradigms, cold-start, diversity frontier, and the learned-reranker check). |

## Development

Lint, type-check, and tests run in CI on every push/PR and as pre-commit hooks.
Everything goes through the uv-locked environment, so the same tool versions run
locally and in CI. Common tasks are wrapped in a [`justfile`](justfile):

```bash
just setup      # venv + deps + dev tools + git hooks (ruff/mypy on commit, pytest on push)
just check      # everything CI runs: lint + format-check + mypy + tests
just cov        # tests with coverage (enforces a 65% floor on the library)
just audit      # pip-audit: known-vuln scan of dependencies
just serve      # run the Streamlit app
just finetune   # fine-tune the co-read encoder + re-embed
```

(Or run the underlying `uv run --no-sync <tool>` commands directly — see the
justfile.) The suite (`tests/`) covers the pure logic — ranking metrics, taste
profiles, the hashing embedder, title search, catalog filters, the CF-matrix
round-trip, both CF builders (KNN + **EASE-R**), the recommender's
scoring/selection contracts, and library-import parsing/matching — all on tiny
synthetic fixtures, so it runs in ~1s with no data files or torch.

**CI / infra** (`.github/`): the `ci.yml` workflow runs ruff (lint + `--check`
format), mypy (scoped to `app/` + `eval/`), and pytest with a coverage floor, plus
an advisory `pip-audit` job; `codeql.yml` runs GitHub's security-and-quality
analysis; Dependabot keeps Python deps, Actions, and the Docker base image current.

### Container

Serving needs only numpy + scipy + streamlit (torch is offline-only), so the
[`Dockerfile`](Dockerfile) is a single slim image with no ML runtime:

```bash
just build-data                 # produce the (gitignored) data artifacts first
docker build -t book-recommender .
docker run --rm -p 8501:8501 book-recommender
```

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
- **A learned reranker.** A logistic ranker trained on real goodbooks interactions
  (over content, CF, cf_weight, popularity, and interactions) was measured against
  the hand-tuned blend (`eval.learned_rerank`) and **matched but did not beat it**
  (0.354 vs 0.353) — it essentially rediscovered the formula. The upside needs
  *real swipe labels* and richer features (recency, skips), so the harness is in
  place but nothing is wired into serving yet.
- **Auth.** Profiles are name-only and URL-resumable; there are no passwords.
- **Scheduling ingestion.** `fetch_new_books.py` + `refresh.py --fetch-new` are the
  live pipeline; running them on a cron/schedule (and expanding beyond Open Library
  to e.g. Google Books) is the remaining operational step.
