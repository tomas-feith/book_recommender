"""Fetch a book the catalog doesn't have, on demand, from Open Library.

When a user searches for a title we don't hold, we look it up live and (if they
add it) ingest it CF-cold -- so it can join their taste profile even though it
brings no collaborative signal. Pure urllib, no torch, so it stays on the
serving side. Network failures degrade to an empty result.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

UA = {"User-Agent": "book-rec/0.1 (on-demand fetch)"}
SEARCH = "https://openlibrary.org/search.json"
_LANG = {"eng": "en", "fre": "fr", "spa": "es", "ger": "de", "ita": "it"}
_FIELDS = "key,title,author_name,first_publish_year,subject,language,cover_i"


def _get(url: str, timeout: int = 12) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception:
        return None


def _description(work_key: str) -> str:
    data = _get(f"https://openlibrary.org{work_key}.json")
    if not data:
        return ""
    desc = data.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")
    return (desc or "").split("--")[0].strip()[:800] if isinstance(desc, str) else ""


def to_record(doc: dict, with_description: bool = False) -> dict | None:
    """Shape an Open Library search doc into a catalog record (``ol:`` id)."""
    key = doc.get("key", "")
    title = doc.get("title")
    if not key.startswith("/works/") or not title or not doc.get("author_name"):
        return None
    subjects = [s.lower() for s in doc.get("subject", []) if s.replace(" ", "").isalpha()][:5]
    langs = doc.get("language") or ["eng"]
    cover = doc.get("cover_i")
    return {
        "id": "ol:" + key.rsplit("/", 1)[-1],
        "title": title,
        "author": ", ".join(doc.get("author_name", [])[:2]),
        "subjects": subjects,
        "language": "en" if "eng" in langs else _LANG.get(langs[0], "en"),
        "year": doc.get("first_publish_year"),
        "image": f"https://covers.openlibrary.org/b/id/{cover}-M.jpg" if cover else "",
        "description": _description(key) if with_description else "",
    }


def search_books(query: str, k: int = 5) -> list[dict]:
    """Look up a title on Open Library; return up to ``k`` catalog records.

    Descriptions are NOT fetched here (one extra request each) -- call
    ``enrich_description`` on the one the user actually adds.
    """
    query = query.strip()
    if not query:
        return []
    # Match on title (cleaner than a full-text `q`, which surfaces study guides and
    # noise). Split a trailing "... by Author" and pass the author too, so common
    # titles disambiguate ("James by Percival Everett"). Author matching can be
    # strict, so fall back to title-only if it comes back empty. Sort by readers.
    parts = re.split(r"\s+by\s+", query, maxsplit=1, flags=re.IGNORECASE)
    title = parts[0].strip()
    author = parts[1].strip() if len(parts) > 1 else ""
    docs = _search_docs(title, author, k * 3)
    if not docs and author:
        docs = _search_docs(title, "", k * 3)
    out: list[dict] = []
    for doc in docs:
        rec = to_record(doc)
        if rec:
            out.append(rec)
        if len(out) >= k:
            break
    return out


def _search_docs(title: str, author: str, limit: int) -> list[dict]:
    fields = {"title": title, "sort": "readinglog", "limit": limit, "fields": _FIELDS}
    if author:
        fields["author"] = author
    return (_get(f"{SEARCH}?{urllib.parse.urlencode(fields)}") or {}).get("docs", [])


def enrich_description(record: dict) -> dict:
    """Fill a record's description from its Open Library work (best-effort)."""
    if record.get("description") or not record.get("id", "").startswith("ol:"):
        return record
    record = dict(record)
    record["description"] = _description("/works/" + record["id"].split(":", 1)[1])
    return record
