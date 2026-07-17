"""Enrich existing catalog books with the Google Books API.

Roughly half of the goodbooks catalog has no description (Open Library didn't
have one). Descriptions are the richest signal our embeddings use, so filling
them in directly improves recommendation quality. Google Books has excellent
descriptions and categories; this looks up books we already have (by title +
author, since we don't store ISBNs) and fills the blanks.

Unlike add_books, this *updates* existing books in place: it fills empty
``description`` (and empty ``subjects`` from Google's categories), then re-embeds
only the changed rows so ``real_embeddings.npz`` stays consistent.

Rate-limited and cached; pass ``--limit`` to bound the number of HTTP requests
sent (the quota unit -- a book whose lookup retries costs several). The free tier
is ~1000/day, with or without a key.

Run (in the uv/torch env):
    uv run --no-sync python scripts/enrich_google_books.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # sibling-script imports

from add_books import _resolve_model  # noqa: E402

from eval.data import book_to_text  # noqa: E402

DATA = ROOT / "data"
API = "https://www.googleapis.com/books/v1/volumes"
UA = {"User-Agent": "book-rec/0.1 (catalog enrichment)"}
CACHE = ROOT / ".cache" / "google_books.json"


class ApiError(RuntimeError):
    """Base for API failures that carry how many HTTP requests they burned."""

    def __init__(self, message: str, requests: int = 0) -> None:
        super().__init__(message)
        self.requests = requests


class QuotaExceeded(ApiError):
    """Google Books returned 429 -- the anonymous quota is a shared, exhausted pool."""


class BackendUnavailable(ApiError):
    """Google Books search returned 503 backendFailed on every retry (Google-side)."""


# The search endpoint 503s intermittently even when healthy, so one book exhausting
# its retries means nothing. This many *consecutive* exhausted books means the
# endpoint is actually down and there is no point spending the rest of the budget.
MAX_CONSECUTIVE_503 = 10

# Attempts per book before giving up on a 503. Every one costs quota, and a book that
# 503s yields nothing however many we spend -- an observed run burned ~580 of its 1000
# calls on 194 books that failed all 3 attempts anyway, i.e. over half the day's budget
# bought nothing. Retries are only worth what they rescue, so keep this low and let a
# later run (the failures are deliberately not cached) retry the endpoint when healthy.
RETRIES_503 = 2


def _query(
    title: str, author: str, api_key: str | None = None, retries: int = RETRIES_503
) -> tuple[dict | None, int]:
    """Look up one book. Returns (volumeInfo or None, HTTP requests sent).

    The request count is the quota unit -- a retried book costs several.
    """
    q = f"intitle:{title}"
    if author:
        q += f" inauthor:{author.split(',')[0]}"
    params = {"q": q, "maxResults": 1, "country": "US"}
    if api_key:
        params["key"] = api_key
    url = f"{API}?{urllib.parse.urlencode(params)}"
    sent = 0
    for attempt in range(retries):
        try:
            sent += 1
            return _pick(
                json.load(
                    urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=15)
                )
            ), sent
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise QuotaExceeded(
                    "Google Books returned 429 (quota exceeded). The unauthenticated "
                    "quota is a shared pool; set GOOGLE_BOOKS_API_KEY or pass --api-key "
                    "with a free key from console.cloud.google.com.",
                    sent,
                ) from e
            if e.code == 503 and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))  # backendFailed is often transient
                continue
            if e.code == 503:
                raise BackendUnavailable(
                    "Google Books search returned 503 (backendFailed) on every retry.",
                    sent,
                ) from e
            return None, sent
        except Exception:
            return None, sent
    return None, sent


def _pick(data: dict) -> dict | None:
    items = data.get("items") or []
    return items[0].get("volumeInfo", {}) if items else None


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a repo-root .env into the environment (no overwrite)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def enrich(
    data_dir: Path = DATA,
    limit: int = 200,
    re_embed: bool = True,
    api_key: str | None = None,
    retries: int = RETRIES_503,
) -> int:
    _load_dotenv()
    api_key = api_key or os.environ.get("GOOGLE_BOOKS_API_KEY")
    books = json.loads((data_dir / "real_books.json").read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    todo = [b for b in books if not (b.get("description") or "").strip()]
    print(
        f"{len(todo)} books missing a description; enriching up to {limit}"
        f"{' (with API key)' if api_key else ' (no API key -- likely 429)'}..."
    )

    changed = {}  # book_id -> book (for re-embedding)
    calls = 0  # HTTP requests sent -- the quota unit, not books looked up
    skipped = 0  # books given up on after their retries all 503'd
    consecutive_503 = 0
    checkpoint = 0
    for b in todo:
        if calls >= limit:
            break
        key = b["id"]
        if key not in cache:
            try:
                info, sent = _query(b["title"], b.get("author", ""), api_key, retries)
                calls += sent
                consecutive_503 = 0
            except QuotaExceeded as e:
                calls += e.requests
                print(f"Aborting: {e}")
                break
            except BackendUnavailable as e:
                # One book's retries all 503'd. That is normal for this endpoint, so
                # drop the book and move on -- but don't cache the failure, or a later
                # healthy run would never retry it.
                calls += e.requests
                skipped += 1
                consecutive_503 += 1
                if consecutive_503 >= MAX_CONSECUTIVE_503:
                    print(
                        f"Aborting: {consecutive_503} books in a row failed with 503. "
                        "The Google Books search endpoint looks down; try again later."
                    )
                    break
                continue
            cache[key] = info or {}
            time.sleep(0.2)  # be polite to the API
            if calls - checkpoint >= 25:
                CACHE.write_text(json.dumps(cache))
                checkpoint = calls
                print(f"  ...{calls} calls, {len(changed)} filled, {skipped} skipped")
        info = cache[key]
        desc = (info.get("description") or "").strip()
        if desc:
            b["description"] = desc[:800]
            if not b.get("subjects") and info.get("categories"):
                b["subjects"] = [c.lower() for c in info["categories"][:3]]
            if not (b.get("image") or "").strip():
                b["image"] = (info.get("imageLinks") or {}).get("thumbnail", "")
            changed[key] = b
    CACHE.write_text(json.dumps(cache))
    print(
        f"Filled {len(changed)} descriptions from {calls} API calls"
        f"{f' ({skipped} books skipped on 503)' if skipped else ''}."
    )

    if not changed:
        return 0

    (data_dir / "real_books.json").write_text(
        json.dumps(books, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if re_embed:
        _reembed(data_dir, changed)
    return len(changed)


def _reembed(data_dir: Path, changed: dict) -> None:
    """Re-embed only the changed books, updating their rows in real_embeddings.npz."""
    from eval.embedders import SentenceTransformerEmbedder

    emb_path = data_dir / "real_embeddings.npz"
    with np.load(emb_path, allow_pickle=True) as z:
        ids, emb, model = z["ids"].astype(str), z["emb"].astype(np.float32), str(z["model"])
    pos = {b: i for i, b in enumerate(ids)}

    ids_to_embed = [k for k in changed if k in pos]
    print(f"Re-embedding {len(ids_to_embed)} changed books with {model}...")
    # The stored value is a *label* ('coread-finetuned bge-small'), not something
    # SentenceTransformer can load -- passing it straight through makes HF read it
    # as a repo id and raise on the space. Resolve it to the encoder directory so
    # the rewritten rows land in the same space as the untouched ones.
    vecs = SentenceTransformerEmbedder(_resolve_model(model, data_dir)).encode(
        [book_to_text(changed[k]) for k in ids_to_embed]
    )
    for k, v in zip(ids_to_embed, vecs, strict=True):
        emb[pos[k]] = v

    tmp = emb_path.with_name(emb_path.stem + ".tmp.npz")
    # Keep writing the label, not the resolved path -- add_books guards against
    # mixing spaces by comparing this string.
    np.savez_compressed(tmp, ids=ids, emb=emb, model=np.array(model))
    os.replace(tmp, emb_path)
    print(f"Updated {len(ids_to_embed)} embedding rows -> {emb_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich catalog books via Google Books.")
    ap.add_argument(
        "--limit", type=int, default=200, help="Max HTTP requests this run (the quota unit)."
    )
    ap.add_argument("--no-embed", action="store_true", help="Skip re-embedding.")
    ap.add_argument("--api-key", help="Google Books API key (or set GOOGLE_BOOKS_API_KEY).")
    ap.add_argument(
        "--retries",
        type=int,
        default=RETRIES_503,
        help=f"Attempts per book before giving up on a 503 (default {RETRIES_503}). "
        "Each attempt costs quota; raise only when the endpoint is healthy.",
    )
    args = ap.parse_args()
    enrich(
        limit=args.limit,
        re_embed=not args.no_embed,
        api_key=args.api_key,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
