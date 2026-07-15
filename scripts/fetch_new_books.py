"""Pull genuinely-new books from Open Library, mapped to the catalog schema.

This is the *source* side of catalog freshness: goodbooks-10k is a frozen dump,
so new releases have to come from a live feed. We query Open Library's search
API, keep recent-year results, and shape each into the record format
``add_books.py`` / ``refresh.py`` consume.

New books get ``ol:``-prefixed ids (e.g. ``ol:OL12345W`` from the work key) so
they never collide with goodbooks' numeric ids. They carry no ratings, so once
ingested they start CF-cold and are ranked by content until reactions accrue.

Two entry points:

* :func:`fetch_new_books` -- small top-up pull (tens of books) from the popular
  head of a few genres. Used by ``refresh.py --fetch-new N``.
* :func:`fetch_diverse` -- bulk expansion (thousands) built to spread across
  genre, author, and popularity rather than piling onto the head. See below.

Why bulk needs its own path: a single search query returns the same popular head
no matter how large ``limit`` gets, so ``want=10000`` off one query list is
impossible. Instead we partition the space into (subject x year) cells, page
through each, and select against explicit diversity quotas:

* **Genre** -- ~40 subjects across fiction AND nonfiction, with a per-subject cap
  so no one genre can dominate the batch.
* **Year** -- an equal quota per publication year. Load-bearing: Open Library
  holds far more well-read 2015 books than 2025 ones, so anything that pools
  years together returns a backlist rather than a modern catalog.
* **Popularity** -- ``readinglog_count`` (free in search results) buckets each
  book head/mid/tail. Bands are within-year percentiles, since readinglog
  accumulates with age and absolute cutoffs would file every recent book as tail.
* **Author** -- a hard per-author cap, so prolific series writers can't flood it.

Descriptions are the one unavoidable N+1 (search won't return them), so they are
fetched concurrently and only for books that survive selection. A book without
one is still kept if it has subjects -- about half the existing catalog has no
description either, and ``scripts/enrich_google_books.py`` backfills them later.

Network failures degrade gracefully: a failed cell is skipped, a failed
description drops that one book. ``--out`` is written incrementally so a long
run that dies partway keeps what it earned.

Run:
    uv run --no-sync python scripts/fetch_new_books.py --out new_books.json --want 20
    uv run --no-sync python scripts/fetch_new_books.py --out big.json --want 10000 --diverse
Then ingest with:
    uv run --no-sync python scripts/refresh.py --add new_books.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UA = {"User-Agent": "book-rec/0.1 (catalog refresh)"}
SEARCH = "https://openlibrary.org/search.json"

LANG_MAP = {"eng": "en", "en": "en", "fre": "fr", "spa": "es", "ger": "de", "ita": "it"}

# Default subjects to pull from -- broad, matches the catalog's popular genres.
DEFAULT_SUBJECTS = (
    "fantasy",
    "science fiction",
    "romance",
    "mystery",
    "historical fiction",
    "young adult",
    "thriller",
    "horror",
)

# Bulk-expansion subjects. Deliberately wider than DEFAULT_SUBJECTS, which is all
# popular fiction -- 10k books drawn from those 8 alone would be 10k
# fantasy/romance/thrillers. Nonfiction is over half the list on purpose.
DIVERSE_SUBJECTS = (
    # fiction
    "fantasy",
    "science fiction",
    "romance",
    "mystery",
    "historical fiction",
    "young adult",
    "thriller",
    "horror",
    "literary fiction",
    "short stories",
    "graphic novels",
    "crime",
    "adventure",
    "humor",
    "war stories",
    "dystopia",
    "magical realism",
    "westerns",
    # nonfiction
    "biography",
    "memoir",
    "history",
    "science",
    "philosophy",
    "psychology",
    "economics",
    "politics",
    "travel",
    "cooking",
    "art",
    "music",
    "nature",
    "technology",
    "mathematics",
    "medicine",
    "religion",
    "sports",
    "education",
    "business",
    "environment",
    "essays",
    "poetry",
    "true crime",
    "anthropology",
    "architecture",
)

# Popularity mix for a bulk pull, as shares of each year's quota.
# Without this the batch is all head: sort=readinglog returns bestsellers first,
# and the tail is where genuinely different books live.
#
# Bands are *within-year percentiles* of ``readinglog_count``, not absolute
# counts, because readinglog accumulates with age: median readinglog runs ~39 for
# 2015 books but ~5 for 2025 ones. Absolute thresholds would file nearly every
# recent book under 'tail' and hand the head/mid quota to a decade-old backlist.
#   name -> share of the year's quota (ordered most popular first)
POPULARITY_BUCKETS = (
    ("head", 0.30),
    ("mid", 0.40),
    ("tail", 0.30),
)

# Head-only mix, for topping the catalog up with the books people actually read.
# Every candidate lands in one ranked 'head' band, so selection walks each
# (subject, year) strictly in readership order -- the marquee titles first.
HEAD_ONLY_BUCKETS = (("head", 1.0),)

MAX_PER_AUTHOR = 4  # a prolific series writer must not flood the batch
SEARCH_PAGE = 100  # Open Library's max page size
MAX_OFFSET = 900  # deep paging gets unreliable past ~1k


def _get_json(url: str, timeout: int = 20, retries: int = 2) -> dict | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.5 * (attempt + 1))  # back off; OL rate-limits bursts
    return None


def _work_description(work_key: str) -> str:
    """Fetch a work's description text (best-effort)."""
    data = _get_json(f"https://openlibrary.org{work_key}.json", timeout=12)
    if not data:
        return ""
    desc = data.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")
    return (desc or "").split("--")[0].strip()[:800] if isinstance(desc, str) else ""


def _search(query: str, limit: int, offset: int = 0) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "sort": "readinglog",
            "limit": limit,
            "offset": offset,
            "fields": (
                "key,title,author_name,first_publish_year,subject,language,cover_i,readinglog_count"
            ),
        }
    )
    data = _get_json(f"{SEARCH}?{params}")
    return (data or {}).get("docs", [])


def _usable(doc: dict, min_year: int) -> bool:
    """Author + cover are cheap signals of a real, catalog-worthy book."""
    return (
        (doc.get("first_publish_year") or 0) >= min_year
        and bool(doc.get("author_name"))
        and bool(doc.get("cover_i"))
        and bool(doc.get("key", "").startswith("/works/"))
        and bool(doc.get("title"))
    )


def _search_subject(subject: str, limit: int, min_year: int) -> list[dict]:
    # Recent English books in this subject, ranked by reader count (a quality
    # proxy) rather than raw recency -- 'sort=new' surfaces obscure/foreign noise.
    query = f"subject:{subject} AND language:eng AND first_publish_year:[{min_year} TO 2035]"
    return [d for d in _search(query, limit) if _usable(d, min_year)]


def _to_record(doc: dict, with_description: bool = True) -> dict | None:
    key = doc.get("key", "")  # "/works/OL...W"
    title = doc.get("title")
    if not key.startswith("/works/") or not title:
        return None
    # Keep hyphens: the catalog's own tags are hyphenated ('science-fiction',
    # 'sci-fi', 'young-adult'), and a bare .isalpha() would drop exactly those --
    # leaving a book tagged only 'fiction' and breaking genre filters. Still
    # excludes Open Library's machine tags ('nyt:hardcover-fiction=2021-05-23').
    subjects = [
        s.lower() for s in doc.get("subject", []) if s.replace(" ", "").replace("-", "").isalpha()
    ][:5]
    langs = doc.get("language") or ["eng"]
    language = "en" if "eng" in langs else LANG_MAP.get(langs[0], "en")
    cover = doc.get("cover_i")
    return {
        "id": "ol:" + key.rsplit("/", 1)[-1],
        "title": title,
        "author": ", ".join(doc.get("author_name", [])[:2]),
        "subjects": subjects,
        "language": language,
        "year": doc.get("first_publish_year"),
        "image": f"https://covers.openlibrary.org/b/id/{cover}-M.jpg" if cover else "",
        "description": _work_description(key) if with_description else "",
    }


def _title_key(title: str, author: str) -> str:
    return f"{title.strip().lower()}|{author.split(',')[0].strip().lower()}"


def _existing(data_dir: Path):
    """(ids, normalized title|author keys) already in the catalog, for dedup."""
    books = json.loads((data_dir / "real_books.json").read_text(encoding="utf-8"))
    ids = {b["id"] for b in books}
    titles = {_title_key(b["title"], b.get("author") or "") for b in books}
    return ids, titles


def fetch_new_books(
    subjects=DEFAULT_SUBJECTS,
    want: int = 20,
    min_year: int = 2022,
    per_subject: int = 15,
    data_dir: Path = DATA,
) -> list[dict]:
    """Return up to ``want`` new-book records not already in the catalog.

    Small top-up pull. For thousands of books use :func:`fetch_diverse`, which
    paginates and spreads across genre/author/popularity instead of taking the
    head of a handful of genre queries.
    """
    known_ids, known_titles = _existing(data_dir)
    out: list[dict] = []
    seen: set = set()
    for subject in subjects:
        if len(out) >= want:
            break
        for doc in _search_subject(subject, per_subject, min_year):
            rec = _to_record(doc, with_description=False)  # cheap pass first
            if not rec:
                continue
            tkey = _title_key(rec["title"], rec["author"])
            if rec["id"] in known_ids or rec["id"] in seen or tkey in known_titles:
                continue
            rec["description"] = _work_description(doc["key"])  # enrich only the keepers
            seen.add(rec["id"])
            out.append(rec)
            if len(out) >= want:
                break
    return out


# ---- bulk diverse expansion --------------------------------------------------


def _sweep_candidates(
    subjects, min_year: int, max_year: int, workers: int, log=print
) -> dict[int, dict[str, list[dict]]]:
    """Collect candidate docs per (year, subject) by paging (subject x year) cells.

    Partitioning by year matters twice over: one query per subject would return
    only that subject's popular head and deep paging a single huge result set is
    unreliable past ~1k, and keeping candidates bucketed by year is what lets the
    selector hold a per-year quota instead of drowning in the older backlist.
    """
    cells = [
        (s, y, off)
        for s in subjects
        for y in range(min_year, max_year + 1)
        for off in range(0, MAX_OFFSET + 1, SEARCH_PAGE)
    ]
    by_year: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    lock = threading.Lock()
    done = 0

    def run(cell):
        nonlocal done
        subject, year, offset = cell
        query = f"subject:{subject} AND language:eng AND first_publish_year:[{year} TO {year}]"
        docs = [d for d in _search(query, SEARCH_PAGE, offset) if _usable(d, year)]
        with lock:
            by_year[year][subject].extend(docs)
            done += 1
            if done % 100 == 0 or done == len(cells):
                total = sum(len(v) for subs in by_year.values() for v in subs.values())
                log(f"  swept {done}/{len(cells)} cells, {total} candidates")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, cells))
    return by_year


def _bucket_within_year(docs: list[dict], buckets=POPULARITY_BUCKETS) -> dict[str, str]:
    """Map work_key -> head/mid/tail by *within-year* readinglog rank.

    Ranking within the year is what keeps the popularity mix honest across an
    11-year span -- see POPULARITY_BUCKETS.
    """
    ranked = sorted(docs, key=lambda d: -(d.get("readinglog_count") or 0))
    out: dict[str, str] = {}
    start = 0
    for i, (name, share) in enumerate(buckets):
        end = len(ranked) if i == len(buckets) - 1 else start + int(len(ranked) * share)
        for d in ranked[start:end]:
            out[d["key"]] = name
        start = end
    return out


class _Selector:
    """Draws docs spread across year, subject, popularity bucket, and author.

    Stateful across calls so a caller can top up after losing books to missing
    descriptions, without re-picking anything already handed out.

    Each year gets an equal quota, and within a year each subject is round-robined
    and each popularity band filled to its share. The year quota is the load-bearing
    part: Open Library holds far more well-read 2015 books than 2025 ones, so any
    selection that pools years together returns a backlist, not a modern catalog.
    """

    def __init__(
        self,
        by_year: dict[int, dict[str, list[dict]]],
        known_ids: set,
        known_titles: set,
        max_per_author: int,
        seed: int,
        buckets=POPULARITY_BUCKETS,
    ) -> None:
        rng = random.Random(seed)
        self.buckets = buckets
        # year -> subject -> bucket -> [docs], each list popped from the END
        self.pools: dict[int, dict[str, dict[str, list[dict]]]] = {}
        for year, by_subject in by_year.items():
            flat = [d for docs in by_subject.values() for d in docs]
            bucket_of = _bucket_within_year(flat, buckets)  # percentiles over the whole year
            per_subject: dict[str, dict[str, list[dict]]] = {}
            for subject, docs in by_subject.items():
                # NB: not named `buckets` -- that would clobber the parameter of the
                # same name, and the next year would bucket against this dict.
                by_bucket: dict[str, list[dict]] = defaultdict(list)
                for d in docs:
                    by_bucket[bucket_of[d["key"]]].append(d)
                for b in by_bucket:
                    if b == "head":
                        # Take the head in readership order -- it's a wide band (the
                        # top 30% of a year is thousands of books), so shuffling it
                        # turns the bestseller tier into a lottery and the genuinely
                        # famous titles get left out. Ascending, because we pop().
                        by_bucket[b].sort(key=lambda d: d.get("readinglog_count") or 0)
                    else:
                        # mid/tail are where breadth lives: sample, don't rank.
                        rng.shuffle(by_bucket[b])
                per_subject[subject] = by_bucket
            self.pools[year] = per_subject
        self.years = sorted(self.pools)
        self.known_ids, self.known_titles = known_ids, known_titles
        self.max_per_author = max_per_author
        self.seen_ids: set = set()
        self.seen_titles: set = set()
        self.author_count: dict[str, int] = defaultdict(int)
        self.taken_per_subject: dict[tuple, int] = defaultdict(int)

    def _take_one(self, year: int, subject: str, bucket: str) -> dict | None:
        pool = self.pools[year][subject].get(bucket) or []
        while pool:
            doc = pool.pop()
            key = "ol:" + doc["key"].rsplit("/", 1)[-1]
            author = (doc.get("author_name") or [""])[0].strip().lower()
            tkey = _title_key(doc["title"], author)
            if key in self.known_ids or key in self.seen_ids:
                continue
            if tkey in self.known_titles or tkey in self.seen_titles:
                continue
            if self.author_count[author] >= self.max_per_author:
                continue
            self.seen_ids.add(key)
            self.seen_titles.add(tkey)
            self.author_count[author] += 1
            self.taken_per_subject[(year, subject)] += 1
            return doc
        return None

    def take(self, want: int, per_subject_cap: int, log=print) -> list[dict]:
        """Draw up to ``want`` fresh docs, spread evenly over years.

        A year that runs dry does not forfeit its share -- leftovers roll into the
        remaining years, so a thin year (2025 is ~1/3 the size of 2015) costs us
        volume rather than the whole target.
        """
        picked: list[dict] = []
        remaining_years = list(self.years)
        for n, year in enumerate(self.years):
            if not remaining_years:
                break
            # Re-divide what's still owed across the years still to come.
            year_quota = max(0, (want - len(picked)) // max(1, len(self.years) - n))
            got = self._take_year(year, year_quota, per_subject_cap)
            picked.extend(got)
            remaining_years.remove(year)
            log(f"  {year}: {len(got)}/{year_quota}")
        return picked

    def _take_year(self, year: int, want: int, per_subject_cap: int) -> list[dict]:
        picked: list[dict] = []
        subjects = sorted(self.pools[year])
        for bucket, share in self.buckets:
            quota = int(want * share)
            filled = 0
            exhausted: set = set()
            while filled < quota and len(exhausted) < len(subjects):
                for subject in subjects:
                    if filled >= quota:
                        break
                    if subject in exhausted:
                        continue
                    if self.taken_per_subject[(year, subject)] >= per_subject_cap:
                        exhausted.add(subject)
                        continue
                    doc = self._take_one(year, subject, bucket)
                    if doc is None:
                        exhausted.add(subject)
                        continue
                    picked.append(doc)
                    filled += 1
        return picked


def _attach_descriptions(docs: list[dict], workers: int, log=print, on_progress=None):
    """Fetch descriptions concurrently; keep books that have some content either way.

    The one unavoidable N+1 -- search won't return descriptions. Only books that
    survive selection get fetched, since this is the expensive phase.

    A missing description is NOT disqualifying: ~51% of the existing catalog has
    none (goodbooks classics included), ``book_to_text`` also embeds title,
    author and subjects, and ``scripts/enrich_google_books.py`` backfills
    descriptions later. Dropping them would also silently skew the batch old --
    Open Library describes a 2015 book far more often than a 2025 one, so the
    filter would correlate with exactly the axis we're trying to spread across.
    Only a book with neither description nor subjects is dropped, as there'd be
    nothing but a title for the content vector to work with.
    """
    records: list[dict] = []
    lock = threading.Lock()
    done = 0
    t0 = time.time()

    def run(doc):
        nonlocal done
        rec = _to_record(doc, with_description=False)
        if rec:
            rec["description"] = _work_description(doc["key"])
        with lock:
            done += 1
            if rec and (len(rec["description"]) > 80 or rec["subjects"]):
                records.append(rec)
            if done % 250 == 0 or done == len(docs):
                rate = done / max(1e-9, time.time() - t0)
                eta = (len(docs) - done) / max(1e-9, rate) / 60
                log(
                    f"  described {done}/{len(docs)} ({len(records)} kept) "
                    f"{rate:.1f}/s, ~{eta:.0f}m left"
                )
                if on_progress:
                    on_progress(list(records))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, docs))
    return records


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_diverse(
    want: int = 10000,
    min_year: int = 2015,
    max_year: int = 2025,
    subjects=DIVERSE_SUBJECTS,
    max_per_author: int = MAX_PER_AUTHOR,
    workers: int = 8,
    seed: int = 0,
    data_dir: Path = DATA,
    out_path: Path | None = None,
    buckets=POPULARITY_BUCKETS,
    log=print,
) -> list[dict]:
    """Bulk-pull ``want`` books spread across year, genre, author, and popularity.

    Pass ``buckets=HEAD_ONLY_BUCKETS`` to top up with the most-read books per
    (subject, year) instead of a head/mid/tail spread.
    """
    known_ids, known_titles = _existing(data_dir)
    n_years = max_year - min_year + 1
    log(
        f"Catalog has {len(known_ids)} books; sweeping {len(subjects)} subjects "
        f"x {n_years} years..."
    )
    by_year = _sweep_candidates(subjects, min_year, max_year, workers, log)
    total = sum(len(v) for subs in by_year.values() for v in subs.values())
    log(f"Swept {total} candidates across {len(by_year)} years x {len(subjects)} subjects.")

    selector = _Selector(by_year, known_ids, known_titles, max_per_author, seed, buckets)
    # Nearly everything selected survives now (only books with neither
    # description nor subjects drop out), but top up in rounds anyway so a thin
    # year or a run of failed fetches can't quietly leave us short.
    per_subject_cap = max(1, int(want / max(1, n_years * len(subjects)) * 2.5))
    records: list[dict] = []
    for round_no in range(1, 5):
        need = want - len(records)
        if need <= 0:
            break
        picked = selector.take(int(need * 1.05), per_subject_cap, log)
        if not picked:
            log("  candidate pool exhausted -- returning what we have")
            break
        log(f"Round {round_no}: describing {len(picked)} candidates ({workers} workers)...")
        got = _attach_descriptions(
            picked,
            workers,
            log,
            on_progress=(lambda r: _write(out_path, records + r)) if out_path else None,
        )
        records.extend(got)
        log(f"Round {round_no}: kept {len(got)}/{len(picked)}; total {len(records)}/{want}")
    return records[:want]


def _summarize(records: list[dict], log=print) -> None:
    by_year: dict = defaultdict(int)
    authors: dict = defaultdict(int)
    for r in records:
        by_year[r["year"]] += 1
        authors[r["author"].split(",")[0]] += 1
    years = ", ".join(f"{y}:{by_year[y]}" for y in sorted(by_year) if y)
    top = sorted(authors.items(), key=lambda kv: -kv[1])[:3]
    log(f"  years -> {years}")
    log(f"  {len(authors)} distinct authors; most-repeated: {top}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch new books from Open Library.")
    ap.add_argument("--out", required=True, help="Path to write the JSON records.")
    ap.add_argument("--want", type=int, default=20, help="How many books to collect.")
    ap.add_argument("--min-year", type=int, default=2022, help="Earliest publish year.")
    ap.add_argument("--subjects", help="Comma-separated subjects (default: broad set).")
    ap.add_argument(
        "--diverse",
        action="store_true",
        help="Bulk mode: paginate and spread across genre/author/popularity (for thousands).",
    )
    ap.add_argument("--max-year", type=int, default=2025, help="Latest publish year (--diverse).")
    ap.add_argument(
        "--max-per-author", type=int, default=MAX_PER_AUTHOR, help="Per-author cap (--diverse)."
    )
    ap.add_argument("--workers", type=int, default=8, help="Concurrent requests (--diverse).")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed (--diverse).")
    ap.add_argument(
        "--head-only",
        action="store_true",
        help="Take the most-read books per subject/year instead of a head/mid/tail mix "
        "(--diverse). Use to top the catalog up with the books people actually read.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if args.diverse:
        subjects = (
            tuple(s.strip() for s in args.subjects.split(","))
            if args.subjects
            else DIVERSE_SUBJECTS
        )
        records = fetch_diverse(
            want=args.want,
            min_year=args.min_year if args.min_year != 2022 else 2015,
            max_year=args.max_year,
            subjects=subjects,
            max_per_author=args.max_per_author,
            workers=args.workers,
            seed=args.seed,
            out_path=out,
            buckets=HEAD_ONLY_BUCKETS if args.head_only else POPULARITY_BUCKETS,
        )
        _summarize(records)
    else:
        subjects = (
            tuple(s.strip() for s in args.subjects.split(","))
            if args.subjects
            else DEFAULT_SUBJECTS
        )
        records = fetch_new_books(subjects=subjects, want=args.want, min_year=args.min_year)

    _write(out, records)
    print(f"Fetched {len(records)} new book(s) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
