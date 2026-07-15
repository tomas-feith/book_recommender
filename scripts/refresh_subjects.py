"""Refresh ``ol:`` books' subjects from Open Library, in the catalog's vocabulary.

Two problems this fixes for books sourced from Open Library:

1. **Tags were being thrown away.** ``fetch_new_books._to_record`` used to keep a
   subject only if ``s.replace(" ", "").isalpha()``, which silently dropped every
   hyphenated tag -- so 'science-fiction' and 'sci-fi' went in the bin and a book
   could end up tagged only 'new york times bestseller'.

2. **The two halves of the catalog speak different dialects.** goodbooks tags are
   Goodreads shelves ('science-fiction', 'young-adult', 'non-fiction'); Open
   Library's are library headings ('science fiction', 'juvenile fiction'). Same
   genre, different string, so they never meet. That matters beyond filtering:
   ``Recommender`` calibrates results against the user's genre distribution
   (``_genre_target`` / ``_book_genre_mass``), so an OL sci-fi book earns zero
   calibration credit from a user whose taste vector says 'science-fiction'.

So we re-fetch raw subjects and normalize them *toward the goodbooks vocabulary*:
lowercase, spaces to hyphens, and prefer tags goodbooks already uses. Both halves
then share one genre space.

Cheap: Open Library's search API accepts batched key queries
(``key:/works/A OR key:/works/B ...``), 100 works per request, so the whole
catalog costs ~100 requests rather than one per book. The slow part is
re-embedding the changed rows, since ``book_to_text`` includes subjects.

Run (in the uv/torch env):
    uv run --no-sync python scripts/refresh_subjects.py            # all ol: books
    uv run --no-sync python scripts/refresh_subjects.py --limit 500 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from add_books import _resolve_model  # noqa: E402

from eval.data import book_to_text  # noqa: E402

DATA = ROOT / "data"
UA = {"User-Agent": "book-rec/0.1 (catalog refresh)"}
SEARCH = "https://openlibrary.org/search.json"

BATCH = 100  # works per search request -- keeps the URL well under any limit
MAX_SUBJECTS = 5  # match the catalog's existing shape


def _get_json(url: str, timeout: int = 40) -> dict | None:
    try:
        return json.load(
            urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)
        )
    except Exception:
        return None


def fetch_subjects(keys: list[str], workers: int = 8, log=print) -> dict[str, list[str]]:
    """work_key -> raw subject list, fetched ``BATCH`` works per request."""
    batches = [keys[i : i + BATCH] for i in range(0, len(keys), BATCH)]
    out: dict[str, list[str]] = {}
    lock = threading.Lock()
    done = 0

    def run(batch: list[str]) -> None:
        nonlocal done
        q = " OR ".join(f"key:/works/{k}" for k in batch)
        params = urllib.parse.urlencode({"q": q, "limit": len(batch), "fields": "key,subject"})
        docs = (_get_json(f"{SEARCH}?{params}") or {}).get("docs", [])
        with lock:
            for d in docs:
                out[d["key"].rsplit("/", 1)[-1]] = d.get("subject") or []
            done += 1
            if done % 20 == 0 or done == len(batches):
                log(f"  fetched {done}/{len(batches)} batches, {len(out)} works")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, batches))
    return out


def normalize_subjects(raw: list[str], vocab: set[str], cap: int = MAX_SUBJECTS) -> list[str]:
    """Normalize OL subjects toward the goodbooks tag vocabulary.

    Tags already in ``vocab`` come first: a shared string is what lets genre
    calibration and filters see an OL book and a goodbooks book as the same genre.
    Remaining tags are kept as filler so a book still carries *something*.
    """
    known: list[str] = []
    other: list[str] = []
    for s in raw:
        if ":" in s or "=" in s:
            continue  # OL machine tag, e.g. 'nyt:hardcover-fiction=2021-05-23'
        for part in s.split(","):  # BISAC headings: 'fiction, fantasy, general'
            t = part.strip().lower().replace(" ", "-")
            if not t or not t.replace("-", "").replace("'", "").isalnum():
                continue  # drop punctuation/non-ascii ('fantasía', 'dragons & myth')
            if not t.isascii() or t in ("general", "fiction-general"):
                continue
            bucket = known if t in vocab else other
            if t not in bucket:
                bucket.append(t)
    out = known + [t for t in other if t not in known]
    return out[:cap]


def goodbooks_vocab(books: list[dict]) -> set[str]:
    """The tag vocabulary of the non-``ol:`` (goodbooks) half of the catalog."""
    return {s for b in books if not b["id"].startswith("ol:") for s in (b.get("subjects") or [])}


def _reembed(data_dir: Path, changed: dict[str, dict], log=print) -> None:
    """Re-embed only the changed books, updating their rows in real_embeddings.npz."""
    from eval.embedders import SentenceTransformerEmbedder

    emb_path = data_dir / "real_embeddings.npz"
    with np.load(emb_path, allow_pickle=True) as z:
        ids, emb, model = z["ids"].astype(str), z["emb"].astype(np.float32), str(z["model"])
    pos = {b: i for i, b in enumerate(ids)}

    to_embed = [k for k in changed if k in pos]
    log(f"Re-embedding {len(to_embed)} changed books with {model}...")
    # The stored value is a *label*; resolve it to the real co-read encoder dir so
    # updated rows land in the same space as the untouched ones.
    vecs = SentenceTransformerEmbedder(_resolve_model(model, data_dir)).encode(
        [book_to_text(changed[k]) for k in to_embed]
    )
    for k, v in zip(to_embed, vecs, strict=True):
        emb[pos[k]] = v

    tmp = emb_path.with_name(emb_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, ids=ids, emb=emb, model=np.array(model))
    os.replace(tmp, emb_path)
    log(f"Updated {len(to_embed)} embedding rows -> {emb_path}")


def refresh_subjects(
    data_dir: Path = DATA,
    limit: int | None = None,
    workers: int = 8,
    dry_run: bool = False,
    log=print,
) -> int:
    books_path = data_dir / "real_books.json"
    books = json.loads(books_path.read_text(encoding="utf-8"))
    vocab = goodbooks_vocab(books)
    targets = [b for b in books if b["id"].startswith("ol:")][: limit or None]
    log(f"{len(targets)} ol: books | goodbooks vocabulary: {len(vocab)} tags")

    keys = [b["id"].split(":", 1)[1] for b in targets]
    log(f"Fetching subjects ({len(keys)} works, {BATCH}/request)...")
    raw = fetch_subjects(keys, workers, log)
    log(f"Got subjects for {len(raw)}/{len(keys)} works.")

    changed: dict[str, dict] = {}
    for b in targets:
        key = b["id"].split(":", 1)[1]
        if key not in raw:
            continue
        new = normalize_subjects(raw[key], vocab)
        if new and new != (b.get("subjects") or []):
            b["subjects"] = new
            changed[b["id"]] = b

    shared = sum(1 for b in targets if any(s in vocab for s in (b.get("subjects") or [])))
    log(f"Changed {len(changed)} books | with >=1 goodbooks-vocab tag: {shared}/{len(targets)}")
    if dry_run:
        log("(dry run -- nothing written)")
        return len(changed)
    if not changed:
        return 0

    tmp = books_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(books, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, books_path)
    log(f"Wrote {books_path}")
    _reembed(data_dir, changed, log)
    return len(changed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh ol: books' subjects from Open Library.")
    ap.add_argument("--limit", type=int, help="Only process the first N ol: books.")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent requests.")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = ap.parse_args()
    n = refresh_subjects(limit=args.limit, workers=args.workers, dry_run=args.dry_run)
    print(f"{n} book(s) updated.")


if __name__ == "__main__":
    main()
