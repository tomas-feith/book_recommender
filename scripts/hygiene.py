"""Ingest-time data hygiene: dedup near-duplicate works, guess real languages.

Breadth sources (the Open Library dump especially) carry many near-duplicate works
-- reprints, editions and translations of the same book -- and no reliable language,
so ``ingest_openlibrary_dump`` used to stamp every book ``"en"``. At 1M books that
fills recommendation lists with the same title and makes the language filter
meaningless. These helpers run before embedding, so duplicates never enter the
catalog and non-English books get a real language code.

Pure-stdlib (no serving deps): dedup is a normalized-key group-and-pick, language a
Unicode-script heuristic (reliable for the non-Latin scripts that were mislabeled;
Latin-script languages fall back to a default).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

_PARENS = re.compile(r"[(\[].*?[)\]]")  # "(Discworld, #5)", "[UK edition]"
_NONWORD = re.compile(r"[^a-z0-9]+")


def norm_title(title: str) -> str:
    """Lowercase, drop series/edition parens + punctuation, collapse whitespace."""
    t = _PARENS.sub(" ", (title or "").lower())
    return " ".join(_NONWORD.sub(" ", t).split())


def dedup_key(book: dict) -> tuple[str, str]:
    """(normalized title, normalized first-credited author) -- the duplicate key."""
    return (
        norm_title(book.get("title", "")),
        norm_title((book.get("author", "") or "").split(",")[0]),
    )


def _completeness(book: dict) -> tuple[int, int, int, int, int]:
    """Rank a record so the *canonical* edition wins a duplicate tie: most-rated,
    then richest metadata (description / subjects / cover / dated)."""
    return (
        int(book.get("ratings_count", book.get("_rc", 0)) or 0),
        len(book.get("description") or ""),
        len(book.get("subjects") or []),
        1 if (book.get("image") or "") else 0,
        1 if book.get("year") else 0,
    )


def dedup_records(records: Iterable[dict], key: Callable[[dict], tuple] = dedup_key) -> list[dict]:
    """Keep one canonical record per key; drop near-duplicate works.

    Precision over recall: a record is only deduped when **both** its title and
    author are present -- two untitled or unattributed works can't be confirmed the
    same, so they're all kept (a wrong merge silently deletes a real book).
    Order-stable: the first-seen canonical record keeps its position.
    """
    best: dict = {}
    order: list = []
    for b in records:
        k = key(b)
        if not (k[0] and k[1]):  # need title AND author to confirm a duplicate
            k = ("", id(b))
        if k not in best:
            best[k] = b
            order.append(k)
        elif _completeness(b) > _completeness(best[k]):
            best[k] = b  # better edition supersedes, keeps the slot
    return [best[k] for k in order]


# --- language guess (Unicode script) -----------------------------------------
# (label, inclusive codepoint range). CJK is ambiguous zh/ja; kana forces ja.
_SCRIPTS: list[tuple[str, int, int]] = [
    ("ar", 0x0600, 0x06FF),  # Arabic
    ("he", 0x0590, 0x05FF),  # Hebrew
    ("ru", 0x0400, 0x04FF),  # Cyrillic
    ("el", 0x0370, 0x03FF),  # Greek
    ("hi", 0x0900, 0x097F),  # Devanagari
    ("th", 0x0E00, 0x0E7F),  # Thai
    ("ja", 0x3040, 0x30FF),  # Hiragana + Katakana
    ("ko", 0xAC00, 0xD7AF),  # Hangul
    ("zh", 0x4E00, 0x9FFF),  # CJK ideographs
]


def guess_language(text: str, default: str = "en") -> str:
    """Best-effort language from the dominant script.

    Reliable for the non-Latin scripts the OL dump mislabeled ``en``; Latin-script
    languages (en/fr/es/de/...) aren't separable by script and fall back to
    ``default``. Kana anywhere forces Japanese over Chinese.
    """
    counts: dict[str, int] = {}
    for ch in text or "":
        o = ord(ch)
        for label, lo, hi in _SCRIPTS:
            if lo <= o <= hi:
                counts[label] = counts.get(label, 0) + 1
                break
    if not counts:
        return default
    if counts.get("ja"):  # kana present -> Japanese, not Chinese
        return "ja"
    return max(counts, key=lambda k: counts[k])
