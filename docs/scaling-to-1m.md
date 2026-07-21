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
| 6 | Tuning constants & the "+35%" claim don't transfer | any | Eval validity (§E) | Silent quality loss | 🚧 Phase 2 started (`eval/served_eval`) |
| 6b | Convex CF/content blend made cold books unreachable | any cold-heavy catalog | Ranking (§E1) | Fatal for the tail | ✅ Fixed (additive blend) |
| 6c | CF reached 10% of the catalog (dense-EASE budget) | >10k items | Algorithm (§B1) | Tail unreachable | ✅ Fixed (EASE/iALS hybrid) |
| 6d | Head and tail compete for the same N slots | any large catalog | Ranking (§E3) | Tail crowded out | ✅ Mitigated (reserved slots) |
| 7 | Dedup / language / selection hygiene | any large ingest | Data source (§F) | Quality | ✅ Fixed (dedup/lang/selection); subjects/coverage remain |
| 8 | Ingest adapter holds raw records + interactions in RAM | ~100k books / ~50M interactions | Data source (§G1) | Fatal (OOM at ingest) | ✅ Fixed (streamed) |
| 9 | Encoding throughput — 4.4 books/s on this CPU | any large ingest | Ingest compute (§G2) | ~63 h for 1M locally | ⚠️ Hardware; checkpointed, GPU for 1M |

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
> emb once per add (a RAM copy, not disk), and the CF npz is still fully re-serialized
> on add (O(nnz) I/O — segment it to fix).

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

> **✅ Resolved (2026-07-21) — `cf_build.hybrid_cf`.** EASE rows for the popular head,
> **iALS** rows for everything else. iALS is O(nnz·k² + N·k³) time and O(N·k) memory, so
> it never forms an item×item matrix and covers the whole catalog: at 100k that is
> 99,996 items against EASE's 10,000. Neither builder wins alone (EASE head/tail
> 0.242/0.008, iALS 0.169/0.104) — EASE is the better model where its solve fits, iALS
> is the only one that reaches the rest. Blocks are row-normalized before merging
> (EASE weights average 0.021, iALS cosines 0.898 — a 43× gap). Tuned to `alpha=40`
> (§E3). The dense-EASE budget itself is now measured, not assumed: 10k, because a
> nominally 10 GB box shows only ~3–5 GB available and h=20000 paged before its inverse
> even began.

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
> - **Embeddings → fp16 memmap + IVF-PQ index** (§A2 RAM). `Catalog.emb` is now a fp16
>   memmap (paged, not a resident fp32 matrix — safe because ANN removed the full
>   scans), and the FAISS index is **IVF-PQ** (compressed) and **persisted** so a boot
>   loads it instead of retraining. Validated: memmap == fp32 recommender output
>   (rec/similar 100%), IVF-PQ vs exact rec 95.6% / similar 98.3% (exact rerank on the
>   memmap vectors keeps quality), index **114 B/vec** (13.5× smaller → ~114 MB at 1M
>   vs 1.5 GB), steady-state boot 0.5 s (no npz parse).
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

> **🚧 Phase 2 started — `eval/served_eval.py`.** A stratified eval that drives the
> *served* stack (`Catalog` + `Recommender` + FAISS), not the research recommenders:
> it blinds a fraction of books cold (out of CF + popularity, embeddings intact), holds
> out liked books, and reports Recall@K split **warm/cold**, catalog **coverage**, and
> per-call **latency**, exact vs ANN. First run on the real 22.6k catalog (40% cold)
> already surfaced two things the head-Recall number hides:
> - **cold Recall@10 ≈ 0** (vs warm 0.33): in a mixed catalog the CF-warm head dominates
>   the ranking and crowds cold, content-only books out of the top-10 — so brand-new /
>   tail books barely surface. This is the "+35% CF is head-only" warning made concrete,
>   and the thing to fix (a cold-start exploration/boost) and then measure here.
> - **coverage ≈ 2%**: recommendations concentrate on a small popular head.
> - ANN keeps warm Recall (0.33→0.33) and is ~5× faster even at 22.6k.
>
> Still to do: re-tune the constants below against this harness, run it on a real
> (or stratified-sample) large catalog, and re-validate the encoder choice. The
> original analysis follows.

### E1. Cold Recall ≈ 0 was a scale bug in the blend, not a missing feature

> **✅ Fixed — additive blend + sparse standardization** (`app/recommender.py`).

The diagnosis, measured rather than assumed: `_scores` computed
`w*_standardize(cf) + (1-w)*_standardize(content)`, but the CF channel is a sum over a
top-k sparse matrix — **~96% of candidates score exactly 0**. That zero mass collapses
the standard deviation, so the surviving few come out around **+57σ** while the dense
content cosine tops out near **+4.5σ**. The blend added two quantities ~13× apart in
scale, so *any* book with a single CF connection beat *every* pure-content book: cold
share of the top-10 was **0.0%** against a 40% base rate, and the first cold book
landed at rank ~140. The per-item `cf_weight` was working correctly; it was being
overwhelmed by an artifact.

Two changes, both selected by sweeping against the harness:

1. **`_standardize_sparse`** — mean/std over the non-zero entries. Fixes the scale
   while keeping CF *magnitude*. (Rank-normalizing both channels also fixes the scale,
   but flattens "co-read 50×" against "co-read once" and cost **32% of warm Recall**.)
2. **Additive blend** — `content + w*cf` rather than `w*cf + (1-w)*content`. Convexly,
   a warm book was scored on CF and a cold book on content: two different quantities
   compared directly. Additively every book shares the content baseline and CF is
   evidence *on top*. **This mattered more than the normalizer.** It also handles a
   warm book that nothing co-read: its CF term goes negative, so it correctly falls
   below a cold book of equal content fit.

**Swept across cold fraction**, because 40% is the current catalog but 250k is ~92%
CF-cold and 1M ~98% (EASE trains the warm head only), and aggregate ≈ `(1-f)·warm +
f·cold` — so tuning at f=0.4 optimizes for a catalog we are about to stop having:

| scheme | f=0.40 | f=0.70 | f=0.90 | f=0.95 |
|---|---|---|---|---|
| convex z (old) | 0.206 | 0.096 | 0.106 | 0.118 |
| convex rank | 0.189 | 0.130 | 0.111 | 0.119 |
| convex z-nonzero | 0.208 | 0.118 | 0.112 | 0.119 |
| **additive z-nonzero** | **0.222** | **0.136** | **0.118** | **0.124** |
| additive rank | 0.104 | 0.078 | 0.031 | 0.018 |

Additive wins at *every* fraction. End to end at f=0.40, exact retrieval: warm
0.348 → **0.360**, cold 0.000 → **0.017**, all 0.206 → **0.222**, coverage 0.024 →
0.026; via FAISS cold 0.017 → **0.075**, coverage 0.025 → **0.034**. Warm *improves*
rather than being traded away, so this is not a cold-vs-warm tradeoff.

*(`additive rank` collapsing to cold ≡ 0.000 at every f confirms the mechanism from
the other side: a bounded non-negative `rank(cf) ∈ [0,1]` bonus means any CF
connection always wins, which is the original bug in miniature.)*

**Still open:** coverage is still ~3%, so the long tail remains largely unreachable —
a genuine exploration slot is a separate lever from this fix.

### E2. Measured on a real 100k catalog — CF coverage is the binding constraint

The 100k Goodreads ingest (§G) gave the first measurement outside the 22.6k regime
everything was tuned in. `served_eval --split natural` splits on whether a book
actually *has* a CF row, which is the division a real catalog has once it outgrows the
EASE budget — 90,000 of 100,000 books here:

| retrieval | warm (10k, CF-backed) | cold (90k, content-only) | all | coverage | ms/call |
|---|---|---|---|---|---|
| exact | 0.242 | 0.008 | 0.180 | 0.013 | 160 |
| faiss | 0.210 | 0.040 | 0.165 | 0.018 | 9.4 |

Two things this settles.

**The exact scan is no longer viable**: 160 ms/call at 100k, ~1.5 s extrapolated to 1M.
FAISS is 17× faster and is now the serving path, not an option.

**The faiss/exact gap is selection bias, not retrieval loss.** A sweep against exact
content ranking on this catalog measured the retrieved set's recall@10 at 0.735
(nprobe=16), 0.912 (48, the old constant) and 0.988 (128) — and retrieval *depth* made
no difference at all (identical at 2000/5000/20000, since IVF-PQ only returns what is
inside the probed cells). Raising nprobe to `nlist//10` moved end-to-end Recall by
**0.001**. The ANN was never the bottleneck; faiss simply trades head for tail.

**A discriminator experiment then isolated the real cause.** Two controlled comparisons:

| A. same books (CF-backed head, 10k pool) | Recall@10 | | B. content-only, matched 10k pools | Recall@10 |
|---|---|---|---|---|
| content only | 0.078 | | head targets | 0.078 |
| CF only | **0.201** | | tail targets | **0.097** |
| blend | 0.209 | | | |

Content ranking is **not** weak on the tail — it is *better* there (0.097 vs 0.078),
plausibly because tail books are distinctive while the head is full of broadly-popular
books that match every profile mushily. So a better encoder is not the lever. What is
true is that **CF is 2.6× content on identical books**, and the blend adds only +0.008
over CF alone: where CF exists it dominates, and it reaches 10% of the catalog.

So the tail's ~0.008 Recall is **crowding, not incapacity** — it loses a competition
against a 2.6×-stronger signal in a pool 10× larger. And it is not an ignorable slice:
**27.6% of users' liked books are in the CF-less tail.**

⇒ **The lever is CF coverage** (§B1's iALS/MF), not the encoder, not the ANN, not the
blend. A dense inverse is capped at ~10k items by measured memory; iALS is O(N·k) and
covers the whole catalog.

*(Secondary: `POP_REF` is now nearly vestigial — with `cf_weight` gated on having a CF
row, the log-scaling barely matters. And the content channel's small marginal
contribution suggests the blend can be simplified once CF coverage is broad.)*

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

### E3. What moved the tail, and what didn't — the one transferable result

Working the 100k catalog end to end produced a sharp asymmetry, and it is more useful
than any single number below. **Every change that helped was about how signals are
combined or allocated. Every change that failed was about making one component
stronger** — and several of the failures were confirmed real effects in their own
intermediate metric, which is exactly why they were convincing.

| lever | intermediate metric | end-to-end effect |
|---|---|---|
| ✅ Additive blend + sparse standardization (§E1) | — | cold 0.000 → 0.017, warm *up* too |
| ✅ `cf_weight` gated on having a CF row | — | prevented a −1.8σ penalty on 90% of the catalog |
| ✅ iALS coverage (§B1) | 10% → 100% of items covered | tail 0.040 → 0.070 |
| ✅ Reserved tail slots | — | tail 0.088 → 0.108 at 20% reservation |
| ✅ iALS `alpha` 10 → 40 | isolated tail R@10 +7.5% | tail +1.4…5.6% (2–3× smaller in situ) |
| ❌ ANN `nprobe` 48 → 128 | retrieved recall@10 0.912 → **0.988** | **+0.001** |
| ❌ ANN retrieval depth 2k → 20k | none — identical | none |
| ❌ iALS factors 64 → 128 | — | +0.004 exact, −0.003 faiss |
| ❌ Better encoder for the tail | — | content is *better* on tail (0.097 vs 0.078) |
| ❌ Weighted-λ regularization | imbalance 17× → **1.5×** | best 0.3300 vs 0.3317 |

Three habits this argues for:

1. **Measure the outcome, not the mechanism.** `nprobe` and weighted-λ both fixed
   genuine, measurable defects and changed nothing that matters.
2. **Isolated sweeps overstate.** The `alpha` sweep measured CF-only ranking in a
   tail-only pool and predicted +7.5%; the served path gave +1.4…5.6%. Crowding
   compresses differences that a clean pool exaggerates — budget a 2–3× haircut.
3. **Watch for the experiment that didn't run.** A `tail_frac` A/B returned
   byte-identical numbers, which read as "no effect" but meant the parameter was
   shadowed by a Python default bound at import. Identical-to-the-last-digit results
   across a changed condition mean the change did not apply.

**Still unresolved:** iALS `reg` is inert from 0.1 to 100, `reg=100` *may* be slightly
better (0.3317 vs 0.3250), but the seed was never varied and adjacent configs wiggle by
up to 0.015 — so that is noise-vs-signal, and settling it needs repeated seeds rather
than a wider grid.

---

## F. Data source & hygiene

> **✅ Partly fixed (Phase 1) — `scripts/hygiene.py`.** Three ingest-time fixes:
> **dedup** (`dedup_records`: group by normalized title+author, keep the most-complete
> edition; precision-first — needs both title AND author, so distinct unattributed
> works are never merged), **language** (`guess_language`: Unicode-script guess from
> the *title*, replacing the OL dump's hardcoded `"en"`), and **selection** (OL
> `select_works` now keeps the top-N by a quality score in a heap, not dump order).
> Wired into `ingest_openlibrary_dump` and `add_books`. Validated on the real catalog:
> 28 author-confirmed dups removed (Harry Potter #5 etc.), 150 non-Latin titles get a
> real language (was all `"en"`), zero false positives. **Still open:** a canonical
> genre *taxonomy* (subject vocab explosion) and description-coverage tracking.

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

## G. The ingest itself — where the data actually comes from

**Source decision: UCSD Goodreads** (`scripts/ingest_goodreads_ucsd.py`).
It is the only candidate that supplies breadth *and* interactions — ~2.3M books with
rated interactions. Open Library has ~30M works but **no interactions**, so a catalog
built from it alone is 100% CF-cold, which §E just measured as Recall@10 ≈ 0 on cold
targets. Use OL to widen *after* a ratings-bearing spine exists.

Files (verified live at `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/`,
cached under the gitignored `.cache/goodreads/`):

| file | size | role |
|---|---|---|
| `goodreads_books.json.gz` | 1.94 GB | metadata; heap-select top-N by `ratings_count` |
| `goodreads_interactions_dedup.json.gz` | 10.7 GB | the CF signal |
| `goodreads_book_genres_initial.json.gz` | 0.02 GB | better genres than `popular_shelves` |
| `goodreads_book_authors.json.gz` | 0.02 GB | author names (829,529) |

**The corpus is smaller than the headline number.** Measured over the full dump:
2,360,648 titled works collapse to **1,769,710 distinct works** — 590,938 (25.0%)
are duplicate *editions* of a work already present, each carrying its own ratings.
So the usable ceiling is ~1.77M, not 2.3M. At 250k we take the top 14% by ratings;
a 1M target would take **57% of every distinct work in the dataset**, reaching deep
into books with near-zero ratings and often no description — which is a content-quality
question (§F description coverage), not just a CF-coldness one. Worth re-checking
before treating 1M as a fixed goal.

### G1. The adapter was written for 25k and had three linear-in-N blowups

> **✅ Fixed.** All three, validated on the real dump and the real goodbooks ratings.

The selection heap held **whole raw records** (Goodreads ships ~100 `popular_shelves`
entries per book, so ~6 GB at 1M) → now a two-pass select: ids only in the heap, then
re-stream and convert just the winners. `build_interactions` built
`{user: {book: rating}}` at ~150 B/interaction — hundreds of GB on the full file →
now counted per user in pass 1, then streamed into `array("i")` coordinate buffers
(4 raw bytes each) under both a user cap and an interaction budget, handed to the new
`cf_build.ease_from_X`. And `json.dumps(books)` materialized the whole catalog as one
string → now written incrementally from shards.

*Validated:* the streamed matrix is identical to the dict-built one (X equal, pop
equal) and EASE through both entry points matches bit-for-bit (`max|Δ| = 0.000e+00`)
on the real goodbooks ratings (5,976,479 ratings / 53,424 users).

### G1b. The Goodreads adapter never ran the hygiene pass

> **✅ Fixed.** `scripts/hygiene.py` was wired into `ingest_openlibrary_dump` and
> `add_books` in Phase 1 — but **not** into the Goodreads adapter, which is the one we
> actually ingest from.

Two consequences, both measured on the real dump:

- **Duplicate editions.** 25% corpus-wide, and 3.4% of the *top* 2000 by ratings
  ("Gone Girl" ×4, "Divergent" ×3) — concentrated in exactly the popular head that CF
  surfaces, so result lists would repeat titles. Dedup now happens **inside** the
  streaming selection (`select_top_book_ids`), not via `hygiene.dedup_records`, which
  needs the whole corpus resident — the thing §G1 exists to avoid. Keeping the
  most-rated edition per (normalized title, first author) matches
  `hygiene._completeness` (ratings first) and, for a ratings source, is both the
  canonical edition and the one carrying the CF signal. After: 0 dup groups remain.
- **Mixed ISO alphabets.** Goodreads puts 639-1 (`nl`) and 639-2 (`dut`) in the same
  field, so the language filter silently missed books — filtering `es` matched nothing
  because Spanish books were tagged `spa`. `norm_language` unifies to 639-1 and falls
  back to the title's Unicode script when the field is blank.

*(The subject-vocabulary explosion §F warns about does **not** occur on this source:
the UCSD genres file gives exactly 10 canonical buckets, not raw shelves.)*

### G2. Encoding is the real ceiling, and it is hardware

Measured on the dev machine (AMD Family 23 mobile APU, 4 cores, torch+MKL at
**43.8 GFLOPS** fp32) with bge-small at 139 avg tokens:

| config | throughput | 250k | 1M |
|---|---|---|---|
| `max_seq_length=512` (as served) | 4.4 books/s | ~16 h | ~63 h |
| `max_seq_length=128` | 7.3 books/s | ~9.5 h | ~38 h |

That is not a misconfiguration — bge-small is ~9 GFLOP/text at that length, so 4.4/s
is exactly what 44 GFLOPS predicts. `OMP_NUM_THREADS=8` does not help (torch pins to
the 4 physical cores). **Consequences:**

- A large ingest is an *overnight-to-multi-day* job, and it is re-paid on every
  encoder change — which makes the "bge-small is enough" finding (§E) load-bearing in
  a way it wasn't at 22k.
- Encoding is therefore **checkpointed** into `rec_*.jsonl` + `emb_*.npy` shard pairs;
  a restart reloads finished shards and encodes only the missing tail. The `.npy` is
  written last, so a half-written chunk is never mistaken for done.
- For the full 1M, embed on a rented GPU rather than locally — a T4/A10 does it in
  well under an hour. The shard layout is already the portable unit of work for that.

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

**Phase 2 — prove it's actually good:** ✅ *largely done at 100k (2026-07-20/21)*

7. ✅ Stratified eval on a real 100k catalog (`served_eval --split poprank|natural`,
   `--assemble`). Constants re-tuned where it paid (`alpha`), left alone where it did
   not (`reg`, `nprobe`, factors). Encoder **re-validated and exonerated**: content
   ranking is *better* on the tail than the head, so the encoder was never the
   constraint — see §E3.
8. Split read-serving from the write path (D). *Still open.*

**Phase 3 — what actually blocks 1M now:**

9. **Encoding throughput.** Everything downstream scales; the encoder does not. At the
   measured ~4.9 books/s a 1M ingest is ~57 h locally, so full scale is a GPU job. The
   shard layout (§G2) is already the portable unit of work for that.
10. **`surprise()` is still a full scan** — 2.3 s at 250k, ~9 s at 1M. Its whole-catalog
    quantile gate is intrinsic to its semantics, so it needs a sampled quantile rather
    than a retrieval fix.
11. **Coverage is ~2%** even after the tail work. Reserved slots move it to ~2.4%; the
    long tail is reachable but still largely unreached.

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
