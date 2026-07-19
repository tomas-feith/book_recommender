# Scaling the catalog to 1M books

The serving catalog is **~22,630 books** today. This document analyzes what breaks
when it grows to **1,000,000** — and gives a concrete fix for each. It's an audit,
not a plan of record: numbers are measured against the current data where possible,
and code is cited by module + symbol (line numbers drift).

The two axes you'd expect — **data source** and **algorithm (model training)** — are
both real and covered below (§B, §F). But **neither is what breaks first.** The
first wall is the in-memory storage layout (§A); close behind are serving-time
full scans (§C) that have nothing to do with model *training*. Two further axes are
easy to miss: **operations/concurrency** (§D) and **evaluation validity** (§E) — the
constants and the "+35% EASE" win were all tuned at 10k and may not transfer.

## TL;DR — what breaks, in the order it bites

| # | What breaks | Roughly when | Axis | Severity | Status |
|---|-------------|--------------|------|----------|--------|
| 1 | Dense genre-mask matrix (`Catalog._genre_idx`) OOMs | ~150–300k books | Storage/memory (§A1) | Fatal, silent | ✅ Fixed (Phase 0) |
| 2 | EASE-R retrain can't build (dense N×N inverse) | ~30–50k items | Algorithm/training (§B1) | Fatal | ✅ Fixed (Phase 0) |
| 3 | Full-file rewrites on every book add + monolithic JSON at boot | ~100k books | Storage (§A2/§A3) | Fatal for the live-add path | ✅ Fixed (Phase 0) |
| 4 | Linear title search (`SequenceMatcher` over all books) | ~100k books | Inference (§C) | Onboarding unusable | ✅ Fixed (Phase 1, FTS) |
| 5 | Full-scan scoring per request (no ANN) | ~200k–500k books | Inference (§C) | Latency + memory churn | ✅ Fixed (Phase 1, FAISS) |
| 6 | Tuning constants & the "+35%" claim don't transfer | any | Eval validity (§E) | Silent quality loss | Phase 2 |
| 7 | Dedup / language / selection hygiene | any large ingest | Data source (§F) | Quality | Phase 1 |

> **Latent break resolved.** `scripts/refresh.py` (the operational CF retrain) called
> `ease_cf`, which capped out around 30–50k items (§B1) — a live risk on the *current*
> growth path, not just at 1M. Phase 0 removed it: EASE now solves over the warm head
> only, bounding the dense inverse regardless of catalog size.

---

## A. Storage & in-memory layout — breaks first

`Catalog` (`app/store.py`) is "everything resident in RAM, one monolithic file per
artifact." That's the pre-pgvector design the module docstring already flags. Three
concrete cliffs.

### A1. The genre mask is a dense G×N boolean matrix — the silent OOM

> **✅ Fixed (Phase 0).** `Catalog` now holds an inverted index `_genre_idx:
> {subject -> int32 array of book rows}` and `filter_mask` scatters those rows into a
> boolean mask. Validated on the real 22,630-book catalog: filter output is identical
> to the old dense masks across 361 filter combinations, memory dropped **201.6 MB →
> 0.40 MB (505×)**, and `filter_mask` runs in ~1.1 ms. At 1M this is ~18 MB instead of
> ~100 GB. The rest of this section (the original analysis) is kept for context.

`Catalog.__post_init__` previously built `_genre_mask: {subject -> np.zeros(N, bool)}`
— **one full-length array per distinct subject.**

- **Measured today:** 8,909 distinct subjects × 22,630 books = **~200 MB already**,
  just for filters.
- **At 1M:** subject cardinality keeps climbing (10k books → ~8k subjects; 22.6k →
  8.9k). Conservatively 50k–200k distinct subjects at 1M ⇒ **50–200 GB**. This OOMs
  with no error — it just dies.
- It's **rebuilt from scratch on every `Catalog.append()`**, so each on-demand
  "add from Open Library" reconstructs the whole thing.

**Fix.** Replace dense per-genre arrays with an **inverted index**
(`subject -> np.array(int32)` of book indices) or a single sparse boolean CSC matrix
(`books × genres`). `filter_mask` becomes a column/set lookup instead of an
N-length allocation. Memory drops from `G×N` to `O(nnz)` ≈ `4.4 × N × 4 B` ≈
**~18 MB at 1M** instead of 100 GB. Mandatory, and independent of everything else —
do it before any ingest past ~150k.

### A2. Monolithic JSON catalog loaded whole

> **✅ Fixed (Phase 0).** Serving now reads a SQLite store (`data/catalog.db`) built
> from `real_books.json` (+ an append-only sidecar) only when an input changes.
> `Catalog.books` is a lazy `BookTable`: full records (descriptions/images) are
> fetched per-id on demand, while the recommender's hot fields (author, subjects,
> language, year, genre index) are resident columnar arrays built in one scan.
> Validated on the real 22,630-book catalog: byte-identical order/`emb`/`sim`/`pop`,
> identical `filter_mask` (200 combos) and `recommend`/`next_cards`/`surprise`/
> `similar` output vs the old list-based `Catalog`; resident `books` metadata 34 MB →
> ~0 (descriptions off-heap, ~1.5 GB saved at 1M). The original analysis follows.

`real_books.json` is 18 MB at 22.6k → **~800 MB at 1M**, `json.load`-ed entirely
(`eval.data.load_books`, `Catalog.load`) into 1M Python dicts (multi-GB heap beyond
the raw bytes). Startup latency and RSS both balloon.

**Fix.** Move metadata to a real store — SQLite (already a dependency) or Parquet —
and load only the columnar arrays needed for scoring/filtering (title, author, year,
language) at boot, fetching full records lazily by index for the ~dozens of books
actually rendered. This also removes A3's need to serialize the whole file.

### A3. Write amplification — O(N) work per single book add

> **✅ Fixed (Phase 0), metadata side.** `append_to_catalog_files` now appends the
> record to `data/real_books_added.jsonl` (O(1), the base `real_books.json` is never
> rewritten) and `Catalog.append` grows the resident filter arrays incrementally (no
> full genre-mask rebuild). Validated on a real-data copy: a live add is visible in
> the running catalog, leaves the base JSON byte-identical, appends one sidecar line,
> and persists across a reload.
>
> **✅ Also fixed (embeddings + CF).** Adds no longer load/rewrite the ~1.5 GB
> embeddings npz: fp16 rows go to an append-only `real_embeddings_added.f16` sidecar
> (row-aligned with the book sidecar; base npz wins on id so a full re-embed
> supersedes it), and the CF matrix grows with `sparse.resize` (empty rows/cols, no
> `block_diag` copy). Validated on a real-data copy: an add leaves the base npz
> byte-identical, writes one fp16 row, and reload round-trips the vector; CF grows in
> place. *Remaining micro-costs:* the in-memory `Catalog.append` still `vstack`s the
> resident emb (a RAM copy, not disk), the CF npz is still fully re-serialized on add
> (O(nnz) I/O — segment it to fix), and the resident fp32 emb could move to a fp16
> **memmap** now that ANN removed the full scans (RAM win at 1M).

`store.append_to_catalog_files` previously ran on **every** external add:

- `np.vstack([old_emb, new])` copies the entire **~1.5 GB** embedding matrix,
- `sparse.block_diag` rebuilds the whole CF matrix,
- rewrites **all three files**, including the ~800 MB JSON,

and `Catalog.append` additionally rebuilds every genre mask (A1) and re-instantiates
`TitleIndex` (`service.add_external_book`) — all under a global `_catalog_lock`. At
1M, one user clicking "add this book" freezes **every** session for seconds-to-minutes
and does GBs of disk I/O.

**Fix.** Make adds **append-only**. New books land in a small staging table/segment
(one DB insert + one appended embedding row), served from a secondary small index,
and get folded into the main artifacts by a periodic compaction job — never a
synchronous full rewrite on the request path.

### A4. CF reorder at load is a column-gather on a 1M×1M CSR

`Catalog.load` does `sim = cf_sim[p][:, p]`. Fancy-indexing the *columns* of a large
CSR forces an expensive gather / format conversion. Cheap at 22k, painful at 1M.

**Fix.** Persist the CF matrix already in canonical book order so load is a straight
`load_cf` with no permutation.

---

## B. Algorithm scaling — model training

### B1. EASE-R is a hard wall

> **✅ Fixed (Phase 0), via "head-only EASE".** An item with no interactions is
> *decoupled* in the Gram (empty user column ⇒ zero off-diagonals), so its EASE
> row/column are all-zero anyway. `ease_cf` now solves only over the **warm** items
> (`pop > 0`), capped at the `max_items` most-rated (default 30k; `refresh.py
> --max-items` tunes it), and scatters the block back into the full N×N matrix. This
> bounds the dense inverse to H×H regardless of catalog size. Validated on the real
> goodbooks training set: when `warm ≤ max_items` the result is **bit-for-bit
> identical** to the old dense-full EASE (`max|Δ| = 0`, same neighbors, same nnz);
> the full rebuild reproduces the shipped matrix (warm ≈ 10k, nnz ≈ 500k). Beyond the
> budget the dropped tail falls to content — the point to add MF/iALS (below). The
> original analysis follows.

`cf_build.ease_cf` (before the fix):

```python
G = np.asarray((X.T @ X).todense(), dtype=np.float64)   # n×n DENSE
P = np.linalg.inv(G)                                     # O(n³)
```

At n=1M the dense Gram is **1M×1M×8 B = 8 TB**, and the inverse is ~10¹⁸ flops.
EASE's closed form caps out around **30–50k items** on a large box — which is why it
shipped at 10k. `scripts/refresh.py` and `build_real_dataset.build_cf` both call
`ease_cf`, so the retrain job that's meant to run continuously **cannot execute at
scale.**

**Fix paths** (choose per how much CF the tail actually needs — see B3):

- **Implicit-feedback MF (iALS / LightFM).** `O(nnz·k)` per iteration, scales to
  millions of items, and yields item factors you can ANN-index for "read-together."
  The standard 1M-scale answer; composes with the existing content channel.
- **Sparse / approximate EASE.** Solve `B` column-by-column via conjugate gradient
  against the sparse Gram (never densify or invert), or block-diagonal EASE over
  popularity/genre partitions. Keeps the serving format; buys ~100–200k items.
- **Head/tail split.** Exact EASE on the top ~30k warm items (where its +35% was
  actually measured), MF or pure content for the long tail. Cheapest way to preserve
  the win where it exists.

### B2. The KNN block builder scales better but still isn't 1M-ready

`cf_build.sparse_topk_cf` (used by `ingest_goodreads_ucsd.py`) densifies
`Rn[block] @ RnT` → **~2 GB per 500-row block × ~2000 blocks** at 1M, and item-item
similarity is fundamentally `O(n²·density)`. It won't instantly OOM like EASE, but
it's days, not minutes. Note the **inconsistency**: the UCSD ingest uses KNN while
`refresh`/`build_real_dataset` use EASE — pick one scalable path.

**Fix.** Mine neighbors via ANN over MF item factors (FAISS) instead of exhaustive
blocks; or run the block matmul on GPU; or use B1's factors directly.

### B3. At 1M, CF barely applies to most of the catalog

The OL bulk ingest sets **zero interactions** for every book
(`ingest_openlibrary_dump.ingest`: empty `sim`, `pop=0`, `language="en"`). Even
UCSD's 876M interactions concentrate on the head. So most of a 1M catalog is CF-cold
⇒ `cf_weight → 0` ⇒ **pure content.** The +35% Recall was measured on 10k *warm,
popular* books; it does **not** describe the tail. **Reset the expectation:** at 1M
the product is content-first, and the marginal engineering dollar should go to
content retrieval quality (§C) and data hygiene (§F), not to squeezing CF.

---

## C. Serving / inference latency — no ANN

> **✅ Fixed (Phase 1).** Two pieces:
> - **Title search → trigram FTS** (`build_catalog_db` builds an FTS5 index; `TitleIndex`
>   retrieves candidates then reranks with the exact fuzzy scorer). Validated: 100%
>   top-1 agreement vs the old scan, 1191 ms → 33 ms at 22.6k (~flat at 1M).
> - **Recommender → FAISS retrieve-then-rerank** (`app/ann.py`; content-ANN ∪ CF-neighbours
>   candidate generation in `Recommender._candidates`, then the existing blend/MMR on the
>   small set). Import-guarded with an exact fallback below `ANN_MIN` / without faiss, so
>   small catalogs and the numpy-only path are unchanged. Validated on real profiles:
>   recommend 95% / similar 98% / next_cards 81% agreement vs exact; FAISS retrieval
>   34× faster than a full scan at 1M (3.3 ms vs 111 ms). **`surprise` stays a full scan**
>   on purpose — it gates on the whole-catalog score/novelty distribution, which a
>   retrieved subset would distort; it's the occasional tab, not the hot path.
>
> The original analysis follows.

Every ranking path scans the full catalog:

- **`Recommender._scores`** computes `emb[cand] @ profile` with `cand ≈ whole
  catalog`, then `argsort` over all of it — on *every* `recommend` / `next_cards` /
  `surprise`. `emb[cand]` is a **~1.5 GB fancy-index copy** each call; the argsort is
  `O(N log N)`. Hundreds of ms + large transient allocation per request, GIL-serialized
  across users.
- **`Recommender.similar`** does a full `emb @ emb[i]` + full argsort on every
  "More like this."
- **`Recommender.surprise`** takes a `np.quantile` over all candidates plus
  `emb[keep] @ emb[pos].T` with `keep` = the top quartile of the catalog (~250k rows).
- **`service.semantic_search`** does a full matmul + full argsort.
- **`TitleIndex.search`** is a Python loop calling `SequenceMatcher.ratio()` for
  *every book* — at 1M, seconds-to-tens-of-seconds per query, and `import_library`
  runs it *per entry*. Onboarding becomes unusable. (The `search.py` docstring already
  promises "in production this is Postgres trigram" — that promise now has to be cashed.)

**Fix — retrieve-then-rerank:**

1. **ANN vector index** (FAISS HNSW/IVF, or pgvector) for content retrieval: top-few-
   hundred candidates in ~1 ms instead of scanning 1M.
2. Apply filters, MMR, calibration, author-cap, and the CF blend **on that small
   candidate set** — all the clever selection logic is fine at a few hundred items;
   it's the retrieval that must stop being `O(N)`.
3. `np.argpartition`, not `argsort`, anywhere a full scan remains.
4. **Title/author search → Postgres trigram/FTS** (or a small in-process n-gram/BM25
   index). Kill the per-book `SequenceMatcher`.
5. **Filters → pre-filtered ANN** (metadata predicates pushed into the index) rather
   than building full-length boolean masks after scoring.

---

## D. Concurrency & operations

- One `@st.cache_resource` singleton service; scoring is CPU/GIL-bound, so as
  per-request cost rises with N, concurrent users **serialize** and throughput
  collapses.
- The global `_catalog_lock` wraps the multi-second full-file writes of A3 → one add
  stalls everyone.
- `streamlit_app.filter_options` scans all books for langs/years/genres at startup;
  `Catalog.all_genres` sorts the full (100k+) genre list every call it's used.

**Fix.** Separate a **stateless read-serving tier** (ANN index + metadata store,
horizontally scalable) from the **write path** (swipes + staged adds in a real DB).
Make catalog growth asynchronous/batched, not synchronous-per-click. Precompute the
genre/language/year facets once into small artifacts.

---

## E. Evaluation validity & tuning transfer

Every constant and conclusion was fit on ≤10k warm, popular, mostly-English books:

- `POP_REF = 500` assumes books reach ~500 ratings. At 1M almost none do ⇒
  `cf_weight ≈ 0` for the tail **by construction**, regardless of CF quality.
- `lam=1000` (EASE), `MMR_LAMBDA`, `CAL_LAMBDA`, `relevance_quantile`, the import
  `threshold=0.55` — all 10k-tuned.
- The **"+35% EASE," the "bge-small is enough" ablation, and the co-read fine-tune
  win** were all measured at 10k/22k. At 1M, bge-small's 384-dim space must separate
  ~45× more books — the ablation may flip.
- The eval harness (`eval/`) loads full data; at 1M it needs **stratified sampling**
  to even run, and needs **cold-tail-stratified metrics** (Recall on cold books,
  catalog coverage, dedup rate) — head Recall@10 will look fine while the tail rots.

**Fix.** Stand up a representative 1M-scale (or stratified-sample) eval *before*
trusting any of these numbers; re-tune the constants; re-validate the encoder choice;
add tail/coverage metrics. Treat the README's evidence tables as **10k-scoped** until
re-measured.

---

## F. Data source & hygiene

- **Duplicates.** The OL dump is full of near-duplicate works/editions/translations.
  At 1M, result lists fill with the same book repeated; MMR and the author-cap only
  dedup *within a returned list*, not the catalog. **Fix:** cluster/dedup at ingest by
  normalized title+author / work-key / ISBN before anything is embedded.
- **Language is hardcoded `"en"`** for all OL-dump books
  (`ingest_openlibrary_dump.to_record`) → the language filter is meaningless across a
  1M OL-sourced catalog. **Fix:** carry/detect real language.
- **Selection quality.** `ingest_openlibrary_dump.select_works` keeps the **first**
  top-n works with a title+description in *dump order* — an arbitrary slice, not the
  1M *best* books. Contrast the UCSD ingest, which correctly heap-selects by
  `ratings_count`. **Fix:** rank the OL selection (edition count, cover, subject
  richness) or prefer a ratings-based source.
- **Subject-vocabulary explosion** (8.9k → 100k+) pollutes the genre UI, the KL
  calibration, and A1's memory. **Fix:** map raw shelves to a **canonical genre
  taxonomy** rather than storing raw strings (`refresh_subjects.py` already does a
  narrow version of this — generalize it).
- **Description coverage** decides tail quality (content carries the cold tail). The
  dump-join `enrich_bulk` is the right offline approach, but books that stay
  description-less embed on title+author alone (weak vectors). Track coverage as a
  first-class metric.

*(One thing that holds up: `sim_indices` as int32 is fine — 50M nnz at k=50 is well
under the 2.1B limit. No overflow risk there.)*

---

## Suggested sequencing

**Phase 0 — unblock the build (before any large ingest):**

1. ✅ **Done.** Genre mask → inverted index (A1). Validated: 505× smaller, filter
   output identical on the real catalog.
2. ✅ **Done.** CF training off dense EASE → head-only EASE (B1). Validated:
   bit-for-bit identical to old EASE below the budget; dense inverse now bounded.
3. ✅ **Done.** Metadata → SQLite serving store with lazy records + columnar boot
   (A2); append-only adds via sidecar (A3, metadata side). Validated: identical
   serving output vs the old list `Catalog`, full records off-heap, base JSON never
   rewritten on add. *Remaining:* append-only **embeddings/CF** npz (still O(N) per
   add) and folding the swipe log — follow-on to this slice.

**Phase 1 — make serving sublinear:**

4. ANN vector index + retrieve-then-rerank (C1–C3).
5. Title/author search → trigram/FTS (C4).
6. Dedup + real language + ranked selection at ingest (F).

**Phase 2 — prove it's actually good:**

7. Stratified 1M eval with cold-tail/coverage metrics; re-tune constants; re-validate
   the encoder (E).
8. Split read-serving from the write path (D).

**First three concrete moves:** the genre-mask inverted index (a low-risk drop-in
`Catalog` change), an ANN retrieve-then-rerank wrapper around `Recommender._scores`
(the biggest latency lever), and moving CF to iALS (which also feeds the ANN
"read-together" index). Those three convert every *fatal* row above into a
*manageable* one and lift the ceiling from ~100–200k to 1M+.

## Related

- [Migrating to Postgres + pgvector](../README.md#migrating-to-postgres--pgvector) —
  the storage swap that underwrites §A and §C.
- `scripts/ingest_goodreads_ucsd.py`, `ingest_amazon_reviews.py`,
  `ingest_openlibrary_dump.py` — the breadth/interaction sources this analysis assumes.
