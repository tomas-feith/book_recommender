"""Ingest the UCSD Goodreads dataset -> our catalog artifacts.

goodbooks-10k is a frozen 10k snapshot. The UCSD Goodreads dataset
(https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) is its natural
superset: ~2.3M books and ~876M interactions **with ratings**, so scaling to it
keeps the collaborative signal strong instead of dumping everything into
cold-start. This adapter maps that source into the exact three artifacts the
serving app already loads.

It streams the files (never loads them whole), selects the top-N books by
rating count, and builds:

* ``data/real_books.json``      -- metadata, ids prefixed ``gr:`` (no goodbooks clash)
* ``data/real_embeddings.npz``  -- bge-small vectors (needs the torch env)
* ``data/real_cf.npz``          -- sparse top-k CF from the real interactions

Inputs (download once from the UCSD page; gzipped JSON-lines / CSV):
    --books        goodreads_books.json.gz              (required)
    --interactions goodreads_interactions_dedup.json.gz (required; string ids)
    --genres       goodreads_book_genres_initial.json.gz (optional, better genres)
    --authors      goodreads_book_authors.json.gz       (optional, for author names)

A single-genre subset (e.g. goodreads_books_fantasy_paranormal.json.gz + its
interactions) is a far smaller, self-contained way to try this first.

Run (in the uv/torch env):
    uv run --no-sync python scripts/ingest_goodreads_ucsd.py \
        --books goodreads_books.json.gz \
        --interactions goodreads_interactions_dedup.json.gz \
        --genres goodreads_book_genres_initial.json.gz \
        --authors goodreads_book_authors.json.gz \
        --top-n 25000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import sys
import time
from array import array
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hygiene import guess_language, norm_title  # noqa: E402

from eval.data import book_to_text  # noqa: E402

DATA = ROOT / "data"

# Goodreads mixes ISO 639-1 ("nl") and 639-2 ("dut") in the same field, so the
# language filter silently misses books unless we normalize to one alphabet.
_ISO2 = dict(
    pair.split(":")
    for pair in (
        "eng:en",
        "spa:es",
        "fre:fr",
        "fra:fr",
        "ger:de",
        "deu:de",
        "ita:it",
        "por:pt",
        "dut:nl",
        "nld:nl",
        "rus:ru",
        "jpn:ja",
        "chi:zh",
        "zho:zh",
        "kor:ko",
        "ara:ar",
        "heb:he",
        "gre:el",
        "ell:el",
        "pol:pl",
        "swe:sv",
        "dan:da",
        "nor:no",
        "fin:fi",
        "tur:tr",
        "cze:cs",
        "ces:cs",
        "hun:hu",
        "ind:id",
        "vie:vi",
        "tha:th",
        "hin:hi",
        "ukr:uk",
        "ron:ro",
        "rum:ro",
        "cat:ca",
        "lat:la",
        "per:fa",
        "fas:fa",
    )
)


# The UCSD genres file uses exactly ten canonical buckets, but covers only ~83% of
# books; the rest fall back to `popular_shelves`, which are mostly shelf *states*
# ("to-read" is on 94% of books, "currently-reading" on 52%). Emitting those as genres
# would make "to-read" the largest genre in the catalog and poison the genre
# calibration, so the fallback is mapped onto the same ten buckets and anything
# unrecognized is dropped -- one vocabulary for the whole catalog, never a shelf state.
_G_FIC, _G_NON = "fiction", "non-fiction"
_G_FAN, _G_MYS = "fantasy, paranormal", "mystery, thriller, crime"
_G_HIS, _G_COM = "history, historical fiction, biography", "comics, graphic"
_SHELF_TO_GENRE = {
    "fiction": _G_FIC,
    "literary-fiction": _G_FIC,
    "classics": _G_FIC,
    "novels": _G_FIC,
    "science-fiction": _G_FIC,
    "sci-fi": _G_FIC,
    "scifi": _G_FIC,
    "dystopia": _G_FIC,
    "non-fiction": _G_NON,
    "nonfiction": _G_NON,
    "self-help": _G_NON,
    "science": _G_NON,
    "business": _G_NON,
    "philosophy": _G_NON,
    "psychology": _G_NON,
    "religion": _G_NON,
    "travel": _G_NON,
    "cookbooks": _G_NON,
    "essays": _G_NON,
    "fantasy": _G_FAN,
    "paranormal": _G_FAN,
    "urban-fantasy": _G_FAN,
    "supernatural": _G_FAN,
    "horror": _G_FAN,
    "vampires": _G_FAN,
    "magic": _G_FAN,
    "mystery": _G_MYS,
    "thriller": _G_MYS,
    "crime": _G_MYS,
    "suspense": _G_MYS,
    "detective": _G_MYS,
    "mystery-thriller": _G_MYS,
    "history": _G_HIS,
    "biography": _G_HIS,
    "memoir": _G_HIS,
    "historical": _G_HIS,
    "historical-fiction": _G_HIS,
    "autobiography": _G_HIS,
    "war": _G_HIS,
    "comics": _G_COM,
    "graphic-novels": _G_COM,
    "graphic-novel": _G_COM,
    "manga": _G_COM,
    "comic-books": _G_COM,
    "bd": _G_COM,
    "romance": "romance",
    "contemporary-romance": "romance",
    "erotica": "romance",
    "young-adult": "young-adult",
    "ya": "young-adult",
    "teen": "young-adult",
    "children": "children",
    "childrens": "children",
    "kids": "children",
    "picture-books": "children",
    "middle-grade": "children",
    "poetry": "poetry",
    "poems": "poetry",
}


def shelves_to_genres(shelves: list[dict], limit: int = 5) -> list[str]:
    """Map Goodreads ``popular_shelves`` onto the canonical buckets, most-shelved first.

    Unrecognized shelves are dropped rather than passed through, so a book with only
    ``to-read``/``owned`` ends up with no genre -- which is honest. An empty subject
    list simply excludes it from genre filters and contributes nothing to the taste
    distribution; a wrong one would actively mislead both.
    """
    out: list[str] = []
    for s in sorted(shelves or [], key=lambda s: -_to_int(s.get("count"))):
        g = _SHELF_TO_GENRE.get(str(s.get("name", "")).strip().lower())
        if g and g not in out:
            out.append(g)
            if len(out) == limit:
                break
    return out


def norm_language(code: str, title: str = "") -> str:
    """Normalize Goodreads' ``language_code`` to a 2-letter ISO 639-1 code.

    Handles regional tags ("en-US"), 639-2 codes ("spa"), and blanks -- an empty
    code falls back to the script of the title, which is how a non-Latin book
    avoids being stamped English.
    """
    c = (code or "").strip().lower().replace("_", "-")
    if not c:
        return guess_language(title)
    base = c.split("-")[0]
    if len(base) == 2:
        return base
    return _ISO2.get(base, guess_language(title))


# Swipe/rating scale is 1-5; UCSD ratings are already 0-5 (0 == no rating).
MIN_RATED_PER_USER = 3  # users with too little signal add noise to CF
MAX_USERS = 200_000  # cap CF training users to bound memory
MAX_INTERACTIONS = 120_000_000  # ~1 GB of int32 CSR coordinates; the real RAM ceiling
EMB_CHUNK = 20_000  # books encoded (and written) per chunk, so texts never all resident
# EASE inverts an h*h float64 Gram and numpy returns the inverse as a second array,
# so peak is ~16*h^2 bytes: 20k -> 6.4 GB, cf_build's own 30k default -> 14.4 GB.
EASE_MAX_ITEMS = 20_000
# Eval-profile selection, mirroring build_real_dataset.py: focused, moderate readers,
# not omnivores who rated a huge share of the catalog.
PROF_MIN_RATED, PROF_MAX_RATED = 10, 40
PROF_MIN_LIKES, PROF_MAX_LIKES = 6, 25
PROF_MAX_USERS = 120
PROF_POOL_MULT = 6  # over-select candidates; the like-count filter runs in pass 2


def stream_jsonl_gz(path: Path) -> Iterable[dict]:
    """Yield one dict per line from a gzipped JSON-lines file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- metadata ----------------------------------------------------------------


def load_authors(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    return {str(a["author_id"]): a.get("name", "") for a in stream_jsonl_gz(path)}


def load_genres(path: Path | None, keep: set | None = None) -> dict[str, list[str]]:
    """book_id -> ranked genre list (UCSD genres file is {book_id, genres:{g:count}}).

    The genres file covers ALL ~2.3M books; pass ``keep`` (a set of raw book_ids)
    to hold only the selected subset in memory.
    """
    if not path:
        return {}
    out: dict[str, list[str]] = {}
    for row in stream_jsonl_gz(path):
        bid = str(row["book_id"])
        if keep is not None and bid not in keep:
            continue
        genres = row.get("genres") or {}
        out[bid] = [g for g, _ in sorted(genres.items(), key=lambda kv: -kv[1])][:5]
    return out


def _to_int(x, default=0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _dedup_digest(title: str, author: str) -> bytes | None:
    """Stable 8-byte key for (normalized title, first author), or None if unusable.

    Hashed rather than stored: the full string tuple over 2.4M books is ~600 MB of
    Python objects, and the digest is reproducible across runs (unlike ``hash()``,
    which is salted per process). 64 bits over a few million keys collides with
    probability ~1e-7, and a collision costs one dropped edition.
    """
    t, a = norm_title(title), norm_title((author or "").split(",")[0])
    if not (t and a):  # precision-first: can't confirm a dup without both
        return None
    return hashlib.blake2b(f"{t}\x00{a}".encode(), digest_size=8).digest()


def select_top_book_ids(books_path: Path, top_n: int, authors: dict[str, str]) -> set[str]:
    """Stream the books file; return ids of the ``top_n`` most-rated **distinct** works.

    **Ids only, deliberately.** Holding the raw records in the heap costs several KB
    each (Goodreads ships ``popular_shelves`` with ~100 entries per book), so at
    ``top_n`` = 1M that heap alone is multiple GB. Ids are ~100 bytes; the caller
    re-streams the file and converts only the winners to slim records.

    Dedup happens **here**, not after selection. Goodreads lists a work once per
    edition, each with its own ratings -- ~3% of the head, concentrated in exactly the
    popular titles CF surfaces -- and ``hygiene.dedup_records`` needs the whole corpus
    resident, which is what this pass exists to avoid. Keeping the most-rated edition
    per key matches ``hygiene._completeness`` (ratings first) and is the right pick for
    a ratings source: it is both the canonical edition and the one carrying CF signal.
    """
    best: dict[bytes, tuple[int, str]] = {}  # dedup digest -> (ratings_count, id)
    loose: list[tuple[int, str]] = []  # no confirmable key -> never merged
    n_seen = 0
    for i, raw in enumerate(stream_jsonl_gz(books_path)):
        title = raw.get("title")
        if not title:
            continue
        n_seen += 1
        rc = _to_int(raw.get("ratings_count"))
        bid = str(raw.get("book_id", i))
        names = [authors.get(str(a.get("author_id")), "") for a in raw.get("authors", [])]
        d = _dedup_digest(title, next((n for n in names if n), ""))
        if d is None:
            loose.append((rc, bid))
        elif rc > best.get(d, (-1, ""))[0]:
            best[d] = (rc, bid)
    cand = list(best.values()) + loose
    print(
        f"  {n_seen} titled works -> {len(cand)} distinct "
        f"({n_seen - len(cand)} duplicate editions dropped, {len(loose)} unattributed)"
    )
    return {bid for _, bid in heapq.nlargest(top_n, cand)}


def iter_records(
    books_path: Path, keep_ids: set[str], authors: dict[str, str], genres: dict[str, list[str]]
):
    """Second pass: yield slim catalog records for the selected books, in file order."""
    for i, raw in enumerate(stream_jsonl_gz(books_path)):
        if not raw.get("title"):
            continue
        if str(raw.get("book_id", i)) in keep_ids:
            yield to_record(raw, authors, genres)


def to_record(raw: dict, authors: dict[str, str], genres: dict[str, list[str]]) -> dict:
    bid = str(raw["book_id"])
    author_names = [authors.get(str(a.get("author_id")), "") for a in raw.get("authors", [])]
    author_names = [a for a in author_names if a][:2]
    subs = genres.get(bid) or shelves_to_genres(raw.get("popular_shelves", []))
    year = _to_int(raw.get("publication_year")) or None
    return {
        "id": "gr:" + bid,
        "title": raw.get("title", ""),
        "author": ", ".join(author_names),
        "subjects": subs,
        "language": norm_language(raw.get("language_code", ""), raw.get("title", "")),
        "year": year,
        "image": raw.get("image_url", ""),
        "description": (raw.get("description") or "").strip()[:800],
    }


# ---- interactions ------------------------------------------------------------


def build_interactions(path: Path, keep: set) -> dict[str, dict[str, float]]:
    """Stream interactions, keeping rated ones for selected books.

    Returns {user_id: {gr_book_id: rating}}. Users with < MIN_RATED_PER_USER
    ratings are dropped; the set is capped at MAX_USERS.

    Only viable for small catalogs -- a Python dict-of-dicts costs ~150 bytes per
    interaction, so the full Goodreads file (hundreds of millions of rated
    interactions) needs :func:`build_user_item` instead.
    """
    by_user: dict[str, dict[str, float]] = defaultdict(dict)
    for row in stream_jsonl_gz(path):
        bid = "gr:" + str(row.get("book_id"))
        rating = _to_int(row.get("rating"))
        if rating >= 1 and bid in keep:
            by_user[str(row["user_id"])][bid] = float(rating)
    filtered = {u: r for u, r in by_user.items() if len(r) >= MIN_RATED_PER_USER}
    if len(filtered) > MAX_USERS:
        # deterministic cap: users with the most ratings (richest CF signal)
        top = sorted(filtered.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:MAX_USERS]
        filtered = dict(top)
    return filtered


def count_ratings_per_user(path: Path, keep: set[str]) -> dict[str, int]:
    """Pass 1 over interactions: how many kept books each user rated.

    Bounded by the number of *users*, not interactions -- the one aggregate we can
    afford to hold before we know which users are worth keeping.
    """
    counts: dict[str, int] = defaultdict(int)
    for row in stream_jsonl_gz(path):
        if _to_int(row.get("rating")) >= 1 and ("gr:" + str(row.get("book_id"))) in keep:
            counts[str(row["user_id"])] += 1
    return counts


def choose_users(
    counts: dict[str, int],
    max_users: int = MAX_USERS,
    max_interactions: int = MAX_INTERACTIONS,
    exclude: set[str] | None = None,
) -> dict[str, int]:
    """Pick the CF training users -> {user_id: row}, under both caps.

    Most-active users first (densest Gram per row kept), dropping anyone below
    ``MIN_RATED_PER_USER``, and stopping once the interaction budget is spent so the
    CSR stays within RAM. Ties break on user id so the build is reproducible.
    ``exclude`` holds out the evaluation-profile users so CF never trains on them.
    """
    skip = exclude or set()
    eligible = sorted(
        ((u, c) for u, c in counts.items() if c >= MIN_RATED_PER_USER and u not in skip),
        key=lambda kv: (-kv[1], kv[0]),
    )
    chosen: dict[str, int] = {}
    total = 0
    for u, c in eligible:
        if len(chosen) >= max_users or total + c > max_interactions:
            break
        chosen[u] = len(chosen)
        total += c
    return chosen


def pick_eval_users(counts: dict[str, int], pool_mult: int = PROF_POOL_MULT) -> set[str]:
    """Users held out to become evaluation profiles -- never used for CF training.

    Mirrors ``build_real_dataset.build_profiles``: moderate, focused readers. Selected
    from the pass-1 counts so the choice is made before any CF matrix is built, which
    is what keeps the split honest -- ``choose_users`` then excludes them, so EASE
    never sees a held-out user's ratings.

    Deliberately over-selects. Pass 1 only knows how many books a user rated, not how
    many they *liked*, and ``to_profiles`` then drops anyone under ``PROF_MIN_LIKES``.
    Picking exactly ``PROF_MAX_USERS`` here would silently yield however many survive
    that filter -- possibly half -- and a thin profile set makes Recall noisy. So take
    a pool and cap after filtering. Holding out a few hundred extra users costs the CF
    matrix nothing against ~200k training users.
    """
    eligible = sorted(u for u, c in counts.items() if PROF_MIN_RATED <= c <= PROF_MAX_RATED)
    return set(eligible[: PROF_MAX_USERS * pool_mult])


def build_user_item(
    path: Path,
    col_of: dict[str, int],
    user_row: dict[str, int],
    n_items: int,
    eval_users: set[str] | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[str, dict[str, int]]]:
    """Pass 2: stream interactions straight into a binary users×items CSR.

    Rows/cols accumulate in ``array('i')`` (4 raw bytes each, no Python int objects),
    which is what keeps a 100M-interaction build inside a few hundred MB instead of
    the tens of GB the dict-of-dicts form would need.

    Also harvests the raw ratings of ``eval_users`` on the way past -- they are excluded
    from ``user_row``, so this is the only chance to see them without a third pass over
    a 10.7 GB file. Bounded by design (~120 users x <=40 ratings), and rating *values*
    are kept here because profiles need like/dislike, unlike the binarized CF matrix.
    """
    evals = eval_users or set()
    eval_ratings: dict[str, dict[str, int]] = defaultdict(dict)
    rows, cols = array("i"), array("i")
    for row in stream_jsonl_gz(path):
        rating = _to_int(row.get("rating"))
        if rating < 1:
            continue
        uid = str(row["user_id"])
        bid = "gr:" + str(row.get("book_id"))
        if uid in evals:
            if bid in col_of:
                eval_ratings[uid][bid] = rating
            continue  # held out: never enters the CF matrix
        ui = user_row.get(uid)
        if ui is None:
            continue
        j = col_of.get(bid)
        if j is not None:
            rows.append(ui)
            cols.append(j)
    r = np.frombuffer(rows, dtype=np.int32)
    c = np.frombuffer(cols, dtype=np.int32)
    X = sparse.csr_matrix(
        (np.ones(len(r), dtype=np.float32), (r, c)),
        shape=(max(len(user_row), 1), n_items),
        dtype=np.float32,
    )
    X.sum_duplicates()
    X.data[:] = 1.0  # binarize: EASE uses implicit co-occurrence, not rating value
    pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    return X, pop, dict(eval_ratings)


def to_profiles(eval_ratings: dict[str, dict[str, int]]) -> list[dict]:
    """Held-out ratings -> eval profiles, same shape as ``data/real_profiles.json``.

    Likes are >= 4 and dislikes <= 2, matching ``build_real_dataset``. Users whose
    positive signal is too thin (or implausibly broad) are dropped, and the result is
    sorted to prefer users who also gave dislikes -- richer signal for the Rocchio arm.
    """
    profiles = []
    for uid, ratings in eval_ratings.items():
        likes = [b for b, s in ratings.items() if s >= 4]
        dislikes = [b for b, s in ratings.items() if s <= 2]
        if PROF_MIN_LIKES <= len(likes) <= PROF_MAX_LIKES:
            profiles.append({"user": f"gr_{uid}", "likes": likes, "dislikes": dislikes})
    profiles.sort(key=lambda p: (len(p["dislikes"]) > 0, p["user"]), reverse=True)
    return profiles[:PROF_MAX_USERS]


# ---- driver ------------------------------------------------------------------


def write_books_and_embed(
    out_path: Path, records: Iterator[dict], encoder, chunk: int = EMB_CHUNK, work_dir=None
) -> tuple[list[str], np.ndarray]:
    """Encode records chunk by chunk into shards, then assemble ``real_books.json``.

    Two things that are fine at 25k and fatal at 1M happen here. ``json.dumps(books)``
    builds the whole serialized document as one string before writing (GBs), and
    ``encode([...all texts...])`` holds every text plus the fp32 result. So we work in
    chunks and keep only fp16 vectors.

    Chunks are **checkpointed** to ``work_dir`` (default: beside the output) as a
    ``.jsonl`` + ``.npy`` pair. Encoding dominates the run -- hours per 100k books on
    CPU -- so a crash, a reboot, or a closed lid must not cost the whole job: on
    restart, finished shards are reloaded and only the missing tail is encoded.
    Re-streaming the source dump to get there costs minutes, which is noise.

    Returns (ids in file order, embeddings).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir else out_path.parent / "_shards"
    work.mkdir(parents=True, exist_ok=True)

    shards: list[int] = []
    n_desc = 0
    pending: list[dict] = []
    i = 0
    for rec in records:
        n_desc += 1 if rec["description"] else 0
        pending.append(rec)
        if len(pending) >= chunk:
            _shard(work, i, pending, encoder)
            shards.append(i)
            i += 1
            pending = []
    if pending:
        _shard(work, i, pending, encoder)
        shards.append(i)

    # Assemble: stream the record shards into the JSON array, stack the vectors.
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("[\n")
        first = True
        for s in shards:
            blocks.append(np.load(work / f"emb_{s:05d}.npy"))
            for line in (work / f"rec_{s:05d}.jsonl").read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                fh.write(("" if first else ",\n") + "  " + line)
                first = False
                ids.append(json.loads(line)["id"])
        fh.write("\n]\n")

    emb = np.vstack(blocks) if blocks else np.zeros((0, 384), dtype=np.float16)
    print(f"  {len(ids)} books written ({n_desc} with descriptions), emb {emb.shape}")
    return ids, emb


def _shard(work: Path, i: int, recs: list[dict], encoder) -> None:
    """Encode + persist one chunk, unless its shard pair is already on disk."""
    rec_p, emb_p = work / f"rec_{i:05d}.jsonl", work / f"emb_{i:05d}.npy"
    if rec_p.exists() and emb_p.exists():
        print(f"    chunk {i}: resumed from checkpoint", flush=True)
        return
    t = time.perf_counter()
    vecs = np.asarray(encoder.encode([book_to_text(b) for b in recs]), dtype=np.float16)
    rec_p.write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in recs) + "\n", encoding="utf-8"
    )
    np.save(emb_p, vecs)  # written last: its presence is what marks the chunk done
    print(f"    chunk {i}: {len(recs)} encoded in {time.perf_counter() - t:.0f}s", flush=True)


def ingest(
    books_path,
    interactions_path,
    genres_path,
    authors_path,
    top_n,
    data_dir=DATA,
    max_items=EASE_MAX_ITEMS,
):
    from cf_build import ease_from_X

    from app.store import save_cf
    from eval.embedders import SentenceTransformerEmbedder

    authors = load_authors(authors_path)  # needed by the dedup key, so load first
    print(f"Pass 1/2 over books: selecting top {top_n} distinct works by rating count...")
    keep_raw = select_top_book_ids(books_path, top_n, authors)
    print(f"  {len(keep_raw)} selected")
    genres = load_genres(genres_path, keep=keep_raw)

    # Same encoder choice as build_embeddings.py: the co-read fine-tuned bge-small
    # when it's built, else stock. Using a different one here would make the new
    # catalog incomparable to the existing eval baselines.
    coread = ROOT / "data" / "coread-encoder"
    model = str(coread) if coread.exists() else "BAAI/bge-small-en-v1.5"
    label = "coread-finetuned bge-small" if coread.exists() else model
    print(f"Pass 2/2 over books: writing records + embedding with {label}...")
    order, emb = write_books_and_embed(
        data_dir / "real_books.json",
        iter_records(books_path, keep_raw, authors, genres),
        SentenceTransformerEmbedder(model),
    )
    del authors, genres, keep_raw
    col_of = {bid: i for i, bid in enumerate(order)}
    keep = set(order)

    print("Pass 1/2 over interactions: counting per user...")
    counts = count_ratings_per_user(interactions_path, keep)
    eval_users = pick_eval_users(counts)
    user_row = choose_users(counts, exclude=eval_users)
    print(f"  {len(counts)} users seen -> {len(user_row)} for CF, {len(eval_users)} held out")
    del counts

    print("Pass 2/2 over interactions: building the user-item matrix...")
    X, pop, eval_ratings = build_user_item(
        interactions_path, col_of, user_row, len(order), eval_users
    )
    print(f"  X = {X.shape[0]}x{X.shape[1]}, {X.nnz} interactions")

    profiles = to_profiles(eval_ratings)
    (data_dir / "real_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    n_dis = sum(1 for p in profiles if p["dislikes"])
    avg = sum(len(p["likes"]) for p in profiles) / max(len(profiles), 1)
    print(f"  {len(profiles)} eval profiles ({n_dis} with dislikes, avg {avg:.1f} likes)")

    # EASE peaks at ~16*h^2 bytes: the h*h float64 Gram plus the inverse numpy
    # returns as a second array. 30k -> 14.4 GB, which OOMs a 10 GB box; 20k -> 6.4 GB.
    n_warm = int((pop > 0).sum())
    h = min(n_warm, max_items)
    print(f"Solving EASE over the warm head: {n_warm} warm, cap {max_items} -> h={h}")
    print(f"  peak ~{16 * h * h / 1e9:.1f} GB for the Gram + inverse")
    sim, pop = ease_from_X(X, pop, max_items=max_items)

    np.savez_compressed(
        data_dir / "real_embeddings.npz",
        ids=np.array(order, dtype=str),
        emb=emb,
        model=np.array(label),
    )
    save_cf(data_dir / "real_cf.npz", order, sim, pop)
    print(
        f"Done. Wrote {len(order)} books, sparse CF ({sim.shape[0]}x{sim.shape[1]}, "
        f"{sim.nnz} nnz) from {len(user_row)} users -> {data_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the UCSD Goodreads dataset.")
    ap.add_argument("--books", type=Path, required=True)
    ap.add_argument("--interactions", type=Path, required=True)
    ap.add_argument("--genres", type=Path)
    ap.add_argument("--authors", type=Path)
    ap.add_argument("--top-n", type=int, default=25000)
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA,
        help="output dir (default: data/). Point a large ingest somewhere else -- it "
        "overwrites real_books/embeddings/cf, i.e. the catalog the evals baseline against.",
    )
    ap.add_argument(
        "--max-items",
        type=int,
        default=EASE_MAX_ITEMS,
        help=f"warm items EASE solves over (default {EASE_MAX_ITEMS}). Peak RAM is "
        "~16*h^2 bytes, so raising this is what OOMs the build.",
    )
    args = ap.parse_args()
    ingest(
        args.books,
        args.interactions,
        args.genres,
        args.authors,
        args.top_n,
        args.out,
        args.max_items,
    )


if __name__ == "__main__":
    main()
