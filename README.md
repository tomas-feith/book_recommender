# Book recommender

[![CI](https://github.com/tomas-feith/book_recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/tomas-feith/book_recommender/actions/workflows/ci.yml)

Swipe-based book discovery: name a few books you love, then swipe through a
personalized deck — **like / interested / haven't read / pass** — while an
adaptive-hybrid recommender re-ranks after every swipe. The product sits on top of an
**offline evaluation harness** that picked the embedding model and the recommender
architecture on real numbers instead of vibes; that evidence is documented at the bottom.

The serving catalog is **100,000 real books** ingested from the
[UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)
(`gr:` ids) — the top 100k by rating count, deduped to one canonical edition per work,
with **real rated interactions** behind them. That matters more than the size: 99,996 of
the 100,000 carry a collaborative-filtering row, because CF coverage — not the encoder,
not the ANN — is what measurably decides recommendation quality on the long tail
(see [Scale: what 100k changed](#scale-what-100k-changed)).

Books arrive with 10 canonical genre buckets, ~93% description coverage, and 94% English
(with Arabic, Spanish, German, Japanese and others behind it). Serving is
**retrieve-then-rerank** over a persisted FAISS IVF-PQ index, so a request touches a few
hundred candidates rather than scanning the catalog.

> **Shell.** Command blocks are PowerShell (Windows 11 / PowerShell 5.1). Almost every
> command is identical in bash — the only differences are line continuations
> (PowerShell uses a backtick `` ` ``, bash a backslash `\`), environment variables
> (`$env:NAME = "x"` vs `NAME=x`), and capturing output (`$port = ...` vs `PORT=$(...)`).
> Where a bash form differs it is noted inline.

## Quick start

The project uses a **uv-managed** environment on Python 3.12 (torch and Streamlit both
have wheels there):

```powershell
uv sync                                                 # create the venv, install deps

# Build the serving artifacts (see "Building the data" -- this is the long part)
uv run --no-sync python scripts/build_real_dataset.py   # goodbooks-10k -> books, profiles, CF
uv run --no-sync python scripts/build_embeddings.py     # content vectors (needs torch)

# Run the app, falling forward from 8501 if that port is taken
$port = uv run --no-sync python scripts/freeport.py
uv run --no-sync streamlit run streamlit_app.py --server.port $port
```

Then open `http://localhost:<port>` (8501 unless it was busy).

`scripts/freeport.py` prints the first free port at or after 8501 to stdout, so the two
lines above are the portable equivalent of `just serve`. In bash:

```bash
PORT=$(uv run --no-sync python scripts/freeport.py) && uv run --no-sync streamlit run streamlit_app.py --server.port "$PORT"
```

> **`just` is optional and needs `sh`.** The [`justfile`](justfile) wraps these same
> commands, but `just` shells out to `sh`, which is not on PATH in a stock PowerShell
> install — so `just serve` fails there even with `just` installed. Every recipe has a
> direct `uv run --no-sync ...` equivalent, listed in [Development](#development).

The serving artifacts in `data/` are **gitignored** — they're build outputs, so a fresh
clone builds them once. Only the things that *can't* be regenerated are committed (the
demo fixtures and the Open Library seed lists); see
[Building the data](#building-the-data) for the full pipeline.

Serving needs only **numpy + scipy + faiss** (scipy loads the sparse CF matrix; faiss
powers ANN retrieval, and is import-guarded so its absence just falls back to an exact
numpy scan); torch is an *offline-only* dependency, used to build embeddings, not to serve.

## The app

`streamlit_app.py` is the front end over `app/service.py`. Four tabs: **Discover**,
**For you**, **Surprise me**, **Reading list**.

- **Onboard** — search by **title or author** (a SQLite **trigram FTS5** index, so
  onboarding search is sublinear rather than an O(N) scan over 100k books), or **search
  by meaning** ("a lonely lighthouse keeper") via the embeddings, or **import your
  reading list** (CSV / TSV / TXT / XLSX, e.g. a Goodreads export). If a book **isn't in
  the catalog**, look it up on **Open Library** and add it on the fly — it's ingested
  CF-cold (content-ranked) so it joins your taste profile immediately.
- **Discover** — swipe one card at a time: **Like**, **Interested** (soft yes → saved to
  your reading list), **Haven't read** (neutral, just skip), or **Pass** (dislike). The
  taste model updates immediately.
- **For you** — a live grid of best-guess recommendations, **diversified with MMR +
  genre calibration** and with slots **reserved for the long tail** (see
  [Diversity](#diversity-the-relevancediversity-frontier) and
  [Reserved tail slots](#reserved-tail-slots)), each with a **"why recommended"** line
  and a **"More like this"** popover; **Save** to your reading list or dismiss as
  **Not for me**.
- **Surprise me** — wildcards: books *far* from your usual taste that readers like you
  still rate highly (see [Surprise mode](#surprise-mode)).
- **Reading list** — everything you marked **Interested**: mark a book **read + liked**,
  remove it, get **More like this**, or **export the list as CSV**.

The sidebar carries your **profile**, language/genre/year **filters**, and a live taste
summary (Liked / Wishlist / Passed / Skipped).

### Durable profiles

Your `user_id` is stored in the URL (`?uid=…`), so a page reload or a server restart
**resumes the same profile** — swipes are persisted in SQLite and are never lost. Name a
profile to save it, and switch between saved profiles from the sidebar. (New sessions
start anonymous until you name them.)

### Reactions and how they score

| Reaction | Meaning | Effect on the model |
|----------|---------|---------------------|
| Like | confident yes | full-weight positive (Rocchio + CF) |
| Interested | soft yes / want to read | positive at weight `α=0.6`, plus reading list |
| Haven't read | can't judge | neutral — excluded from future cards, no signal |
| Pass | dislike | negative at weight `β=0.5` (dislikes are noisier) |

### Theme

A warm literary theme lives in `.streamlit/config.toml` — cream "paper" light mode and
deep "ink" dark mode (both defined, so the in-app toggle works), serif headings
(Fraunces) over an Inter body, pill buttons.

## Architecture

| Module | Role |
|--------|------|
| `app/store.py`       | `Catalog` (a lazy **SQLite-backed `BookTable`** for metadata + **trigram FTS** search + fp16 memmapped embeddings + **sparse top-k CF matrix** + popularity + **inverted-index metadata filters**, one aligned index) and `SwipeStore` (users/swipes/profiles in SQLite). Split so the store can move to Postgres+pgvector without touching the rest. |
| `app/ann.py`         | **FAISS IVF-PQ** index over the content embeddings, persisted to `data/ann.idx`. Import-guarded and size-gated (`ANN_MIN = 50_000`), so a small catalog or a faiss-less install falls back to the exact numpy scan. |
| `app/recommender.py` | Adaptive hybrid: Rocchio profile, **additive** content + CF blend gated on CF *evidence*; retrieve-then-rerank; list assembly with **MMR + genre calibration + reserved tail slots**, exploit/explore, `surprise()`, `similar()` ("more like this"), and per-pick explanations. |
| `app/search.py`      | Fuzzy **title + author** resolution for the seed step, reranking the FTS candidate set with a popularity tiebreak so the canonical edition surfaces first. |
| `app/library.py`     | Parse an uploaded reading list (CSV/TSV/TXT/XLSX) into `(title, author)` entries. |
| `app/external.py`    | On-demand Open Library lookup for books not in the catalog (urllib-only, no torch). |
| `app/service.py`     | `BookRecommenderService`: users/profiles, `seed`, `next_cards`, `swipe`, `recommendations`, `surprises`, `wishlist`, `semantic_search`, `similar_books`, `import_library`, filters. The seam a UI/HTTP layer sits on. |
| `app/demo.py`        | Scripted end-to-end session (seed → recommend → swipe → adapt → filter). |

### The adaptive hybrid

Every candidate is scored on two independent axes and blended **per item**:

- **Content** — cosine of the candidate to your Rocchio taste centroid
  (liked + `α`·interested − `β`·disliked embeddings).
- **CF** — item-item collaborative signal: "readers who liked your books also liked
  this," independent of whether the descriptions resemble each other.

The blend is **additive**, not convex:

    score = standardize(content) + cf_weight · standardize_sparse(cf)

Two details carry the result, and both were bugs first:

**1. Additive, not `w·cf + (1−w)·content`.** Under the convex form a warm book was scored
on CF and a cold book on content — two different quantities compared directly with
nothing calibrating them — and cold books lost every time. Measured: **0.0% of the top-10
were cold books against a 40% base rate**, i.e. they were structurally unreachable no
matter how well they matched. Additively, every book shares the content baseline and CF
is *evidence on top*, so the comparison is always content-vs-content plus a bounded,
signed bonus. Swept at cold fractions 0.40/0.70/0.90/0.95, additive wins the aggregate at
every fraction (0.222/0.136/0.118/0.124 vs 0.206/0.096/0.106/0.118) and *improves* warm
Recall too (0.360 vs 0.348) rather than trading it away.

**2. The CF channel is standardized over its non-zero entries.** CF is a sum over a
top-k sparse matrix, so most candidates score exactly 0; that zero mass collapses the
standard deviation and puts the few non-zero entries ~57σ out against a content cosine
that tops out near +4.5. Standardizing on the entries that actually carry signal fixes
the scale while keeping CF *magnitude* — rank-normalizing also fixes the scale but
flattens "co-read 50×" against "co-read once" and costs 32% of warm Recall.

**`cf_weight` tracks CF evidence, not popularity.** It is
`log(1+pop) / log(1+500)` capped at 1 — *but zeroed wherever the book has no CF row at
all*. The two come apart once the catalog outgrows the EASE budget: at 250k every book is
popularity-warm (rank 250,000 still has 201 ratings) while most have a structurally empty
CF row. Those would score `cf_weight ≈ 0.85` against a CF sum of exactly 0, which the
sparse standardization maps to ≈ −2.1 — penalizing the entire mid-tail ~1.8σ *below books
with no ratings at all*, on a content channel spanning only ±4.5.

**Retrieve-then-rerank.** With an ANN index and a taste vector, candidates are the
content-ANN top-K **union** the CF neighbours of the taste set, then filtered and
de-seen — a few hundred rows instead of 100k, so the blend and MMR downstream stay
sublinear. Without faiss, or below `ANN_MIN`, it degrades to the exact filtered scan and
behaviour is identical.

### Reserved tail slots

`REC_POOL_MULT · n` candidates are diversified down to `n`, but a fraction
(`TAIL_SLOTS_FRAC = 0.2`, so 2 of 10) is **reserved** for books outside the popular head
(`HEAD_FRAC = 0.1`, by popularity *rank* so the split means the same thing at 22k and 1M).

This is a slot allocation rather than a scoring change, because scoring cannot fix it:
head and tail compete for the same `n` places, so at 100k swapping in a stronger *head*
model (EASE over iALS) lifted head Recall 0.169 → 0.237 and **halved** tail Recall
0.104 → 0.050 with no change to the tail model at all. Tail exposure has to be allocated,
not earned. Reserving 20% moved tail Recall 0.088 → 0.108. Set `tail_frac=0` to rank
purely by score.

### Surprise mode

`Recommender.surprise()` produces serendipity without abandoning quality. It **gates**
candidates to the top quartile of blended score (so every pick is still a confident
recommendation), then ranks *those* by **novelty** = `1 − cosine to your nearest liked
book`. A book that is both high-scoring and far from your taste is one the **CF channel**
is carrying — nothing like your usual genres, but loved by readers like you. (It needs
some likes to define "your taste," and by construction it rides CF, so zero-rating cold
books can't be surprises.)

**It is sampled, not exhaustive.** ANN retrieval cannot serve this — it finds books
*near* the taste vector and surprise wants far ones — so it used to scan the whole
catalog, reaching 2.3 s at 250k and ~9 s extrapolated to 1M. But the gate is a *quantile*
of the score distribution, which a uniform sample estimates fine. Scoring a
`SURPRISE_SAMPLE = 20_000` sample makes the cost independent of catalog size while
leaving thousands of books in the top-quantile slice to rank by novelty.

## Data pipeline & keeping the catalog fresh

Every book is three aligned artifacts, all keyed by book id: `data/real_books.json`
(metadata), `data/real_embeddings.npz` (content vector, **stored fp16** — half the
file/load), `data/real_cf.npz` (item-item CF + popularity). Three more are **derived** at
load time and rebuilt only when an input's mtime changes: `catalog.db`, `emb.f16`,
`ann.idx`.

> **Serving metadata store.** At serve time the metadata isn't held in RAM as one big
> list of dicts — `Catalog.load` builds a SQLite store (`data/catalog.db`) from
> `real_books.json` plus an append-only `real_books_added.jsonl` sidecar. `Catalog.books`
> is then a **lazy `BookTable`**: full records are fetched per-id on demand while the
> recommender's hot fields (author, subjects, language, year, genre index) stay resident
> as columnar arrays. On-the-fly adds append to the sidecar (never rewriting the base
> file). This keeps ~800 MB of descriptions off the heap at 1M books; see
> [docs/scaling-to-1m.md](docs/scaling-to-1m.md) §A2/§A3.

> **On fp16 embeddings.** Vectors are stored fp16 and materialized into a memmapped
> `emb.f16` in catalog order, so a steady-state boot never loads the full fp32 matrix.
> fp16 ranking is accuracy-neutral (identical Recall@10). ANN retrieval removed the full
> scans that previously forced a resident fp32 matrix.

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ingest_goodreads_ucsd.py` | **The current catalog source.** Ingest the [UCSD Goodreads dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) (~2.3M books, ~876M interactions **with ratings**): streams the gz files, heap-selects top-N by rating count, dedups editions *inside* the streaming selection, normalizes mixed ISO language codes, maps to the schema (`gr:` ids), and builds embeddings + sparse CF from the *real* interactions — so CF stays strong at scale instead of collapsing to cold-start. Encoding is checkpointed into shard pairs, so a restart resumes. |
| `scripts/rebuild_cf.py` | **Rebuild CF without re-ingesting.** The ingest streams 10.7 GB to build the user-item matrix then throws it away, making every CF experiment cost ~50 min of I/O. This caches the matrix beside the catalog (`interactions.npz`) and rebuilds in minutes. `--method hybrid` (default) / `ease` / `ials`. |
| `scripts/promote_catalog.py` | **Swap a built catalog into serving, migrating swipes.** `data/app.db` holds real swipe history keyed on the *old* catalog's ids, and book ids are per-source (goodbooks `126` vs Goodreads `gr:5907`), so a straight file copy silently orphans every profile. Matches on normalized title + first author (97% of real swipes carried over). Backs everything up; dry run unless `--apply`. |
| `scripts/cf_build.py` | CF matrix builders, all emitting the same sparse top-k format — see [CF builders](#cf-builders-ease-ials-and-the-hybrid). |
| `scripts/hygiene.py` | Ingest-time data hygiene: **dedup** near-duplicate works (normalized title+author, keep the most-complete edition; needs *both* fields, so distinct unattributed works are never merged), and **guess language** from the dominant Unicode script. Pure stdlib. |
| `scripts/build_embeddings.py` | Cache the content vectors (N×384) so serving never loads torch. Uses the **co-read fine-tuned** encoder (`data/coread-encoder`) when present, else stock `bge-small`. |
| `scripts/finetune_coread.py` | **Cold-start fine-tune.** Distill EASE's co-read structure INTO the content encoder: build (anchor, positive) pairs from each book's top EASE neighbors and fine-tune `bge-small` with an in-batch contrastive (InfoNCE) objective. Gives unrated books a collaborative-aware embedding EASE can't (see evidence below). Writes `data/coread-encoder`; then re-run `build_embeddings.py`. |
| `scripts/build_real_dataset.py` | The **small/legacy** path: build a 10k dataset from goodbooks-10k — top-`N_BOOKS` by rating count, reader shelf-tags as genres, Open Library descriptions, 120 focused eval users, and the sparse CF matrix (from ~53k *non-eval* users, so the harness stays honest). Fast, self-contained, and what the 10k-era evidence below was measured on. |
| `scripts/refresh.py` | **Periodic refresh** for a *goodbooks-derived* catalog. Rebuilds CF from goodbooks ratings **plus the app's own swipe log** (like/interested/dislike → pseudo-ratings 5/4/2). `--add PATH` ingests a file first; `--fetch-new N` pulls N new books from Open Library. **Guarded:** it refuses to run the CF rebuild against a `gr:` catalog (nothing would join, quietly replacing a 5M-interaction matrix with one learned from a few hundred swipes) and points you at `rebuild_cf.py`. `--add` / `--fetch-new` still work anywhere. |
| `scripts/add_books.py` | **Incrementally** append new books to all three artifacts — embeds only the new ones (same model, guarded), grows CF with zero rows so new books start cold (pop=0, content-ranked). Idempotent, atomic. |
| `scripts/fetch_new_books.py` | Pull genuinely-new books from the **Open Library** search API, `ol:`-prefixed, deduped. `--diverse` is the bulk path: partitions the search into (subject × year) cells against explicit quotas — equal **year** quota, head/mid/tail **popularity** by *within-year* percentile, ~40 **subjects**, and a per-**author** cap. `--head-only` takes the most-read per cell instead. |
| `scripts/refresh_subjects.py` | **Unify the genre vocabulary** between goodbooks shelves (`science-fiction`) and Open Library headings (`science fiction`) — the same genre as a different string, which matters because the recommender calibrates against the user's genre distribution. Only relevant to the goodbooks + OL catalog; the Goodreads source already ships 10 canonical buckets. |
| `scripts/enrich_google_books.py` | Fill missing descriptions/categories/covers via the **Google Books API**, then re-embed only the changed rows. Needs a free `GOOGLE_BOOKS_API_KEY`. One book per request and ~1k/day — prefer `enrich_bulk.py` when a dump is available. |
| `scripts/enrich_bulk.py` | Backfill the same missing descriptions from **bulk dataset dumps** — no API, no quota. Streams the UCSD Goodreads dump and/or the OpenLibrary works dump once each and re-embeds the changed rows. Pass `--goodreads` and/or `--ol-works`. |
| `scripts/fetch_google_books.py` | Add NEW books from the **Google Books API** by subject (`gb:` ids), deduped; writes a JSON list for `add_books`/`refresh --add`. |
| `scripts/ingest_amazon_reviews.py` | **More CF signal.** Ingest [Amazon Reviews 2023 (Books)](https://amazon-reviews-2023.github.io/) — the structural twin of the Goodreads adapter (meta + reviews → sparse CF, `az:` ids). |
| `scripts/ingest_openlibrary_dump.py` | **Breadth (content-only).** Ingest the [Open Library bulk dumps](https://openlibrary.org/developers/dumps) (~30M works, CC0), `ol:` ids. No ratings → every book is CF-cold; pair with a ratings source or swipes to grow CF. |
| `scripts/freeport.py` | Print the first free TCP port at or after 8501 (stdlib only). What the serve commands capture. Checks both that the port **binds** and that nothing is **listening** on it — a bind test alone reports an actively-served port as free, because a cross-process listener holding `SO_REUSEADDR` lets a second bind succeed on Windows. |

### CF builders: EASE, iALS, and the hybrid

All three emit the same sparse top-k `(sim, pop)` format, so `store.save_cf` / `load_cf`
and the recommender's `_cf_sum` don't care which produced the matrix.

| Builder | What it is | Where it wins |
|---|---|---|
| `ease_cf` / `ease_from_X` | **EASE-R**: one closed-form regularized solve, `B = −P/diag(P)` with `P = (XᵀX + λI)⁻¹`, λ=1000, top-50. | The **stronger model**, but its dense inverse is O(H²) memory / O(H³) time, capped at `EASE_MAX_ITEMS = 10_000` by *measured* headroom on this box. Head Recall@10 **0.242** at 100k. |
| `ials_cf` | **Implicit ALS** (Hu/Koren/Volinsky 2008): alternating ridge solves for low-rank user/item factors, converted to the same top-k item-item matrix via cosine over factors. | **Coverage.** O(nnz·k² + N·k³) time and O(N·k) memory, so it reaches the *whole* catalog. Tail Recall@10 **0.104** where EASE gets 0.008. |
| `hybrid_cf` | **EASE rows for the popular head, iALS rows for everything else**, both row-normalized first. | The served default. Rows are the unit of choice because `_cf_sum` scores a candidate from its own row. |

Row normalization is not optional: EASE weights average 0.021 with row sums ~1.06, iALS
cosines average 0.898 with row sums ~44.9 — a 43× difference that would let the iALS block
win every comparison regardless of fit.

`sparse_topk_cf` (adjusted-cosine KNN) is kept as a dependency-light fallback and for
comparison; EASE-R measured **+35% Recall@10** over it on the 10k catalog (0.262 → 0.355).

**iALS `alpha = 40` is tuned; `reg` is not load-bearing.** Swept on the real 100k catalog,
CF-only Recall@10 in a tail-only pool: alpha 1/10/40 → 0.283/0.308/0.327, then falling
away sharply (0.290 at 80, 0.275 at 160, 0.264 at 320). `reg` moves Recall less than
adjacent configs wiggle across 0.1 → 100, so it stays at 10 rather than being chased.

### Building the data

`data/` is gitignored apart from what cannot be regenerated. What's committed, and why:

| Committed | Why it can't be regenerated |
|-----------|------------------------------|
| `data/sample_books.json`, `data/sample_profiles.json` | Hand-curated demo fixtures (48 books / 8 profiles). No generator exists — they're source, and `python -m eval.run` reads them by default. |
| `data/recent_books.json`, `data/expansion_10k.json`, `data/topup_head.json` | **Open Library snapshots.** OL is a live catalog, so re-running the fetcher returns *different* books — never these. Keeping the seed lists is the only way to rebuild the same goodbooks-era catalog. |

Everything else is a build output: `real_books.json`, `real_profiles.json`,
`real_embeddings.npz`, `real_cf.npz`, `coread-encoder/`, the derived `catalog.db` /
`emb.f16` / `ann.idx`, and the runtime `app.db`.

#### Path A — the 100k serving catalog (what ships)

Download the UCSD Goodreads files once (they cache under the gitignored `.cache/goodreads/`;
`goodreads_books.json.gz` is 1.94 GB and `goodreads_interactions_dedup.json.gz` is 10.7 GB).
**This is an overnight job** — encoding runs at ~4.4 books/s on a 4-core CPU, so 100k books
is several hours and the interaction pass is ~50 min of I/O. Build into a *staging*
directory, then promote:

```powershell
# 1. ingest -> data_100k/{real_books.json, real_embeddings.npz, real_cf.npz, real_profiles.json}
uv run --no-sync python -u scripts/ingest_goodreads_ucsd.py `
    --books .cache/goodreads/goodreads_books.json.gz `
    --interactions .cache/goodreads/goodreads_interactions_dedup.json.gz `
    --genres .cache/goodreads/goodreads_book_genres_initial.json.gz `
    --authors .cache/goodreads/goodreads_book_authors.json.gz `
    --top-n 100000 --out data_100k

# 2. rebuild CF with the served hybrid (EASE head + iALS tail); caches the
#    interaction matrix so later sweeps are minutes, not an hour
uv run --no-sync python -u scripts/rebuild_cf.py --data data_100k `
    --interactions .cache/goodreads/goodreads_interactions_dedup.json.gz `
    --method hybrid

# 3. preview the swap (dry run: reports how many swipes carry over), then apply
uv run --no-sync python scripts/promote_catalog.py --from data_100k
uv run --no-sync python scripts/promote_catalog.py --from data_100k --apply
```

`promote_catalog` backs up the previous `data/` to `data_backup_<timestamp>/` and deletes
the derived artifacts, so the next `Catalog.load` rebuilds `catalog.db`, `emb.f16` and
`ann.idx` from the new inputs. A stale derived file paired with fresh inputs is silent
corruption, which is why they are never copied.

A **single-genre subset** (e.g. `goodreads_books_fantasy_paranormal.json.gz` plus its
interactions) is a far smaller, self-contained way to try this path first.

#### Path B — the 10k goodbooks catalog (fast, self-contained)

Minutes rather than hours, no multi-GB downloads. This is what the 10k-scoped evidence
below was measured on, and it's the quickest way to get a working app:

```powershell
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

That yields ~22,630 books: all of goodbooks-10k plus ~12,600 modern books (2015-2025)
from Open Library. To pull a *fresh* modern batch instead of replaying the seeds — a
different set of books, by design — see `fetch_new_books.py --diverse`, then ingest the
JSON it writes with `refresh.py --add`.

#### Ongoing refresh

For a goodbooks-derived catalog, `refresh.py` turns accumulated app usage into
collaborative signal:

```powershell
uv run --no-sync python scripts/refresh.py --fetch-new 20   # pull new releases + rebuild CF
uv run --no-sync python scripts/refresh.py                  # rebuild CF from swipes
```

For the `gr:` catalog, rebuild from the cached interaction matrix instead:

```powershell
uv run --no-sync python scripts/rebuild_cf.py --data data --method hybrid
```

The design intent: **content carries new books until real usage accrues; the refresh job
then turns that usage into collaborative signal.**

## Why these choices — the evidence

The recommender core was validated *before* the UI, with a held-out ranking harness
(`eval/`): for each synthetic user, hold out `k` liked books, build a taste profile from
the rest, rank the catalog, and measure where the held-out likes land (Recall@K, NDCG@K,
MRR), averaged over users and random splits.

```powershell
uv run --no-sync python -m eval.run --strategy both   # mean vs rocchio profile
uv run --no-sync python -m eval.compare_paradigms     # content vs CF vs hybrid
uv run --no-sync python -m eval.cold_start            # the onboarding regime
uv run --no-sync python -m eval.served_eval           # the ACTUAL serving stack
```

> **Scope note.** The four tables immediately below were measured at **10k/22k books** and
> are kept because their *structural* results still hold. The absolute numbers do not
> transfer to 100k — see [Scale: what 100k changed](#scale-what-100k-changed), which
> supersedes them wherever they disagree.

### Paradigm comparison (warm users, 10k catalog)

Measured on the **full 10,000-book sparse top-k catalog** (120 users · hold-out=3 ·
eval@10 · 5 splits/user · 600 trials per recommender):

| recommender             | Recall@10 | NDCG@10 | MRR    | note |
|-------------------------|-----------|---------|--------|------|
| popularity (floor)      | 0.037     | 0.022   | 0.038  | non-personalized baseline |
| content: hashing        | 0.107     | 0.074   | 0.111  | lexical baseline |
| content: bge-small      | 0.114     | 0.083   | 0.125  | best content model; edges hashing |
| **collaborative: EASE-R** | **0.351** | 0.279 | 0.380 | closed-form item-item; the CF core |
| hybrid (static 50/50)   | 0.344     | 0.280   | 0.388  | content slightly *dilutes* strong CF |

**For warm users, CF wins decisively** — taste correlations live in co-rating patterns,
not description text. EASE-R measured **+35% Recall@10 over the adjusted-cosine KNN it
replaced** (0.262 → 0.351). It's so strong that for *warm* users content slightly dilutes
it — but content is still the **only** signal for cold-start, so the served recommender
keeps the adaptive per-item blend (swapping the KNN core for EASE lifted the served
adaptive hybrid 0.275 → 0.333).

### Cold-start simulation

`eval.cold_start` marks ~40% of the catalog newly-added (zeroed out of CF and popularity;
embeddings untouched) and asks whether each paradigm can surface a relevant *unrated*
book — the onboarding regime.

| recommender             | Warm books | Cold books (0 ratings) |
|-------------------------|-----------|------------------------|
| popularity              | ~0.085    | **0.000** |
| collaborative (item-item/EASE) | ~0.30 | **0.000** |
| content: bge-small      | ~0.137    | **~0.142** |
| hybrid 50/50            | ~0.282    | ~0.064 |

CF and popularity **cannot recommend an unrated book at all**; content performs the same
with or without ratings. The two paradigms are complementary, and a static hybrid is wrong
in both regimes — hence the **adaptive per-item weight**.

### The `--text-mode` diagnostic

The sample books carry literal genre words and the synthetic users like strictly within a
genre, which lets a keyword matcher win by matching "fantasy" rather than understanding
anything. `--text-mode no-subjects` strips those words:

```powershell
uv run --no-sync python -m eval.run --model hashing `
    --model BAAI/bge-small-en-v1.5 --strategy rocchio --text-mode no-subjects
```

The hashing baseline collapses (~0.73 → ~0.40) while neural models hold (~0.60) and
overtake it. Your real catalog behaves like the `no-subjects` column, so trust it when
picking a model: `bge-small-en-v1.5` > `MiniLM-L6` > lexical, and Rocchio helps once the
keyword crutch is gone.

### Does a *bigger* embedding model help? (measured: no)

`bge-small` (384-dim) is the serving model. We tested whether scaling up the encoder buys
anything, on a fixed shared candidate pool (same pool for every arm, so cross-model deltas
are apples-to-apples):

| model      | dim  | content R@10 | **hybrid R@10** | full-10k embed (6-thread CPU) |
|------------|------|--------------|-----------------|-------------------------------|
| bge-small  | 384  | 0.132        | **0.302**       | ~15 min                       |
| bge-base   | 768  | **0.149**    | 0.297           | ~57 min                       |
| bge-large  | 1024 | 0.140        | 0.302           | ~196 min                      |

**The served hybrid is flat** (0.302 / 0.297 / 0.302 — within noise, barely above CF-alone
at 0.292): the blend is CF-dominated, so a sharper *content* channel gets washed out.
Worse, bigger isn't even monotonic — `bge-large` *regresses* below `bge-base` on the
content arm, so it's strictly dominated (slower **and** less accurate). Cost scales the
wrong way: 4×–13× the offline embed time and 2×–2.7× the vector storage. A bigger *stock*
encoder is the wrong lever — the right one for cold-start turned out to be **fine-tuning**
the small model on collaborative signal (next).

### Collaborative-aware content for cold-start (measured, shipped)

EASE-R is *silent* on an unrated book — an all-zero row, so it can't rank a brand-new /
never-rated book at all. Content is the only signal there. So we **distilled EASE's
co-read structure into the content encoder** (`bge-small`, in-batch InfoNCE on top-EASE-
neighbor pairs; `scripts/finetune_coread.py`), so a book lands near its would-be co-read
neighbors *from text alone*.

The honest test is **leakage-free**: mark ~40% of books cold (held out of *both* the CF
matrix and the training pairs), then rank held-out likes by content only.

| held-out target | base bge-small | co-read fine-tuned | Δ |
|-----------------|----------------|--------------------|---|
| **cold** (model never trained on these) | 0.147 | **0.166** | **+12%** |
| warm (contrast — trained on)            | 0.121 | 0.151 | +25% |

The **+12% on cold books the model never saw** is genuine generalization — a capability
EASE structurally cannot have. (The larger warm gain is the *redundant* part: books EASE
already handles at serving.) The win is real but **narrow**: it only helps the content
channel, i.e. onboarding and brand-new catalog additions. Serving uses the fine-tuned
encoder's vectors — same 384-dim, same numpy-only serving path.

### Scale: what 100k changed

`eval.served_eval` drives the **actual serving stack** — `Catalog` + `Recommender` + the
FAISS ANN — rather than research recommenders over raw embeddings, so the
retrieve-then-rerank path, the adaptive blend and the ANN approximation are all measured
end to end. `--split natural` splits on whether a book actually *has* a CF row, which is
the division a real catalog has once it outgrows the EASE budget.

At 100k, before iALS (90,000 of 100,000 books CF-less):

| retrieval | warm (10k, CF-backed) | cold (90k, content-only) | all | coverage | ms/call |
|---|---|---|---|---|---|
| exact | 0.242 | 0.008 | 0.180 | 0.013 | 160 |
| faiss | 0.210 | 0.040 | 0.165 | 0.018 | 9.4 |

**The exact scan is no longer viable** — 160 ms/call at 100k, ~1.5 s extrapolated to 1M.
FAISS is 17× faster and is now the serving path, not an option.

**The tail's ~0.008 Recall was crowding, not incapacity.** A discriminator experiment
isolated it: on *identical* books, CF scores 0.201 against content's 0.078 — **CF is 2.6×
content** — while the blend adds only +0.008 over CF alone. And content is not weak on the
tail; it is *better* there (0.097 vs 0.078 on matched 10k pools), plausibly because tail
books are distinctive while the head is full of broadly-popular books that match every
profile mushily. So the tail was losing a competition against a 2.6×-stronger signal in a
pool 10× larger — and it is not an ignorable slice: **27.6% of users' liked books were in
the CF-less tail.**

⇒ **The lever was CF coverage**, not the encoder, not the ANN, not the blend. Hence iALS.

### What moved the tail, and what didn't

The single most useful result from working the 100k catalog end to end is an asymmetry:
**every change that helped was about how signals are combined or allocated; every change
that failed was about making one component stronger** — and several failures were
confirmed real effects in their own intermediate metric, which is exactly why they were
convincing.

| lever | intermediate metric | end-to-end effect |
|---|---|---|
| ✅ Additive blend + sparse standardization | — | cold 0.000 → 0.017, warm *up* too |
| ✅ `cf_weight` gated on having a CF row | — | prevented a −1.8σ penalty on 90% of the catalog |
| ✅ iALS coverage | 10% → 100% of items covered | tail 0.040 → 0.070 |
| ✅ Reserved tail slots | — | tail 0.088 → 0.108 at 20% reservation |
| ✅ iALS `alpha` 10 → 40 | isolated tail R@10 +7.5% | tail +1.4…5.6% (2–3× smaller in situ) |
| ❌ ANN `nprobe` 48 → 128 | retrieved recall@10 0.912 → **0.988** | **+0.001** |
| ❌ ANN retrieval depth 2k → 20k | none — identical | none |
| ❌ iALS factors 64 → 128 | — | +0.004 exact, −0.003 faiss |
| ❌ A bigger/better encoder for the tail | — | content is *better* on tail (0.097 vs 0.078) |
| ❌ Weighted-λ regularization | imbalance 17× → **1.5×** | best 0.3300 vs 0.3317 |
| ❌ Representation-program edits (`eval.repr_sweep`) | — | strip-promo is a dud; genre-upweight is a cold/warm trade |

Three habits this argues for:

1. **Measure the outcome, not the mechanism.** `nprobe` and weighted-λ both fixed genuine,
   measurable defects and changed nothing that matters.
2. **Isolated sweeps overstate.** The `alpha` sweep predicted +7.5%; the served path gave
   +1.4…5.6%. Crowding compresses differences a clean pool exaggerates — budget a 2–3×
   haircut.
3. **Watch for the experiment that didn't run.** A `tail_frac` A/B returned byte-identical
   numbers, which read as "no effect" but meant the parameter was shadowed by a Python
   default bound at import. Identical-to-the-last-digit results across a changed condition
   mean the change did not apply.

### Diversity: the relevance↔diversity frontier

You don't want ten near-identical fantasy novels (or all seven Harry Potters), and if you
like fantasy *and* romance you want both — in proportion. The "For you" list is selected
greedily to maximize, per pick,

    λ · relevance − (1 − λ) · max-similarity-to-already-picked − cal · KL(taste ‖ list-genres)

over the top `REC_POOL_MULT·n` candidates, capped per author. The three terms:
**relevance**, an **MMR** redundancy penalty (`mmr_lambda`; exact set-diversity is NP-hard
so greedy is the standard approximation), and **genre calibration** (Steck): `cal_lambda`
pulls the list's genre mix toward the user's taste mix via KL divergence, so a minority
taste isn't drowned out by the majority one. A single-author saga is collapsed by the
author cap (which keys on *every* credited name, so a pseudonym or co-credit — "Richard
Bachman, Stephen King" — can't slip a second book past it); cross-author near-duplicates
by the similarity penalty; taste *coverage* by calibration.

`eval.diversity` sweeps both knobs over the real profiles, measuring Recall@10 against
**intra-list distance** (ILD), **genre entropy**, **miscalibration KL** (list vs. taste
genre mix; lower = better), and catalog **coverage**:

| `mmr` | `cal` | Recall@10 | ILD | genre-H | miscalKL | coverage |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.0 | 0.0 | 0.341 | 0.500 | 4.09 | 1.550 | 0.052 |
| 0.5 | 0.0 | 0.338 | 0.514 | 4.14 | 1.515 | 0.053 |
| **0.5** | **0.4** | **0.341** | 0.517 | 4.16 | **1.454** | 0.053 |
| 0.5 | 0.8 | 0.338 | 0.523 | 4.19 | 1.388 | 0.054 |

**Both diversity and calibration are ~free here.** Dropping `mmr` to 0.3 costs ~2% Recall
for +7% ILD; adding `cal=0.4` *lowers* miscalibration (1.515 → 1.454) while nudging Recall
*up*. Defaults `MMR_LAMBDA=0.5`, `CAL_LAMBDA=0.4` sit on the cheap part of both curves;
`mmr_lambda` is a natural "focused ↔ eclectic" control.

> Calibration is the answer to the *multi-taste* problem a single Rocchio centroid can't
> solve at scoring time (per-cluster profiles were tried and lost — see below): instead of
> splitting the taste vector, we let scoring stay pooled and fix the genre *balance* at
> list-assembly time. Cheaper, and it measurably works.

### Eval harness layout

| File | Role |
|------|------|
| `data/sample_books.json` / `sample_profiles.json` | 48 books / 8 synthetic users, for the fast keyword-vs-semantic diagnostic. |
| `eval/data.py`      | Loads data; `book_to_text` decides what text represents a book. |
| `eval/embedders.py` | `HashingEmbedder` (numpy) + `SentenceTransformerEmbedder` (optional). |
| `eval/profiles.py`  | `mean` and `rocchio` taste-vector builders. |
| `eval/metrics.py`   | Recall@K, NDCG@K, MRR, intra-list distance, genre entropy. |
| `eval/recommenders.py` | Popularity / item-item CF / embedding / hybrid strategies behind one `score()` interface, so they compete on the identical scoreboard. |
| `eval/representations.py` | `REGISTRY` of book→text *representation programs* — the item-text transform treated as a tunable artifact (the one transferable idea from AutoIndex). |
| `eval/run.py` · `compare_paradigms.py` · `cold_start.py` · `diversity.py` · `learned_rerank.py` · `repr_sweep.py` | The research scoreboards: ranking paradigms, cold-start, the diversity frontier, the learned-reranker check, and the representation sweep. |
| `eval/served_eval.py` | The **served-stack** scoreboard: real `Catalog` + `Recommender` + FAISS, stratified warm/cold, with coverage and per-call latency. This is the one that predicts the real experience at scale. |

## Development

Lint, type-check, and tests run in CI on every push/PR and as pre-commit hooks. Everything
goes through the uv-locked environment, so the same tool versions run locally and in CI.

The [`justfile`](justfile) wraps the common tasks, but **`just` needs `sh` on PATH**, which
a stock PowerShell install doesn't have. The direct equivalents:

| Task | `just` | PowerShell / direct |
|---|---|---|
| venv + deps + git hooks | `just setup` | `uv sync; uv run --no-sync pre-commit install; uv run --no-sync pre-commit install --hook-type pre-push` |
| lint + format check | `just lint` | `uv run --no-sync ruff check .; uv run --no-sync ruff format --check .` |
| auto-fix | `just fmt` | `uv run --no-sync ruff format .; uv run --no-sync ruff check --fix .` |
| type-check | `just typecheck` | `uv run --no-sync mypy` |
| tests | `just test` | `uv run --no-sync python -m pytest` |
| tests + coverage floor | `just cov` | `uv run --no-sync python -m pytest --cov=app --cov=eval --cov-report=term-missing` |
| everything CI runs | `just check` | run the lint, typecheck and test rows above |
| dependency audit | `just audit` | `uv run --no-sync pip-audit` |
| run the app | `just serve` | `$port = uv run --no-sync python scripts/freeport.py; uv run --no-sync streamlit run streamlit_app.py --server.port $port` |
| fine-tune + re-embed | `just finetune` | `uv run --no-sync python scripts/finetune_coread.py --steps 60; uv run --no-sync python scripts/build_embeddings.py` |

Note PowerShell 5.1 has no `&&` — use `;` to chain (or `; if ($?) { ... }` to chain only
on success).

The suite is **180 tests in ~5 s** (currently **72% coverage** against a **65% floor**),
all on tiny synthetic fixtures with no data files and no torch. It covers the pure logic:
ranking metrics, taste profiles, the hashing embedder, title search, catalog filters, the
CF-matrix round-trip, all three CF builders (KNN, EASE-R, iALS), the recommender's
scoring/selection contracts including `cf_weight` gating and the reserved tail slots,
representation programs, ingest hygiene, and library-import parsing/matching.

**CI / infra** (`.github/`): `ci.yml` runs ruff (lint + `--check` format), mypy (scoped to
`app/` + `eval/`), and pytest with the coverage floor, plus an **advisory** `pip-audit` job
(`continue-on-error`, so it reports without blocking); `codeql.yml` runs GitHub's
security-and-quality analysis; Dependabot keeps Python deps, Actions, and the Docker base
image current. Transitive-dependency advisories are pinned out via
`[tool.uv] constraint-dependencies` in `pyproject.toml` rather than by declaring a direct
dependency.

Batch CLIs (`app/demo.py` and the `eval/*` entrypoint scripts) are omitted from coverage —
they're operational glue over heavy, untyped deps, not unit-tested library code.

### Container

Serving needs only numpy + scipy + streamlit + faiss (torch is offline-only), so the
[`Dockerfile`](Dockerfile) is a single slim image with no ML runtime:

```powershell
# produce the (gitignored) data artifacts first -- see "Building the data"
docker build -t book-recommender .
docker run --rm -p 8501:8501 book-recommender
```

The image pins `python:3.12-slim` (not 3.14) because faiss-cpu wheels lag the newest
CPython, and ships a `HEALTHCHECK` against Streamlit's `/_stcore/health`.

## Migrating to Postgres + pgvector

`Catalog` is the only piece that changes: replace the numpy embedding/CF search with SQL
(pgvector `<=>` for content, a stored item-item table for CF) and keep the `filter_mask`
conditions as `WHERE` clauses. `SwipeStore` is already database-shaped. Nothing in
`recommender.py` / `service.py` moves.

For the **1M-book** target specifically — what breaks at that scale and the fix for each
(the dense genre-mask OOM, the dense-EASE retrain wall, no-ANN full scans, and the
data-hygiene issues) — see [docs/scaling-to-1m.md](docs/scaling-to-1m.md). The measured
ceiling on the dev box is **encoding**: bge-small at 4.4 books/s means 1M books is ~63 h
locally, so a large ingest is an overnight-to-multi-day job (or a rented GPU).

## What this deliberately is *not* (yet)

- **Multi-taste profiles.** A single centroid can't perfectly represent someone who likes
  literary fiction *and* hard sci-fi. Per-cluster centroids were built and evaluated
  (held-out Recall@10) and **lost** to the pooled mean — sub-centroids from a handful of
  likes overfit, and no real user's tastes were separable enough to help. Genre
  calibration addresses the same problem at list-assembly time instead, and does work.
- **A learned reranker.** A logistic ranker over content, CF, `cf_weight`, popularity and
  interactions was measured against the hand-tuned blend (`eval.learned_rerank`) and
  **matched but did not beat it** (0.354 vs 0.353) — it essentially rediscovered the
  formula. The upside needs *real swipe labels* and richer features (recency, skips), so
  the harness is in place but nothing is wired into serving.
- **A tuned item-text representation.** `eval.repr_sweep` treats the book→text transform
  as a tunable artifact and measured the grounded edits: stripping marketing preamble is a
  dud, and up-weighting genre words is a cold/warm trade, not a win. Neither shipped.
- **Re-tuned constants at 100k+.** `POP_REF = 500`, EASE `lam=1000`, `MMR_LAMBDA`,
  `CAL_LAMBDA`, `relevance_quantile` and the import `threshold=0.55` were all fit at
  10k/22k. `POP_REF` in particular is now nearly vestigial given `cf_weight` gates on
  having a CF row at all.
- **Catalog data hygiene at the edges.** The 100k ingest dedups editions and normalizes
  languages, but publication years still range from 16 to 2104 — the year filter is honest
  about what's in the data, not about what's plausible. A canonical genre *taxonomy* and
  description-coverage tracking are also open (§F of the scaling doc).
- **Auth.** Profiles are name-only and URL-resumable; there are no passwords.
- **Scheduled ingestion.** The refresh/rebuild scripts are the live pipeline; running them
  on a cron is the remaining operational step.
