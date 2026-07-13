"""Load the book catalog and user profiles, and turn books into embedding text.

The ``book_to_text`` function is deliberately isolated: it decides *what* text
represents a book for embedding. Note it does NOT include hard-filter metadata
(language, exact year) -- those belong in structured filter columns, not in the
semantic vector. Subjects/genres are included because they carry real semantic
signal that helps clustering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_books(path: Path | None = None) -> List[Dict]:
    path = path or DATA_DIR / "sample_books.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_profiles(path: Path | None = None) -> List[Dict]:
    path = path or DATA_DIR / "sample_profiles.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def book_to_text(book: Dict, mode: str = "full") -> str:
    """Compose the semantic text used to embed a book.

    ``mode`` controls how much explicit genre signal leaks into the text:

    * ``full``       -- title + description + subject/genre words.
    * ``no-subjects``-- title + description only. Drops the literal genre words,
      which is a diagnostic: if a lexical baseline's lead evaporates here, it was
      riding keyword overlap rather than understanding content.
    """
    base = f"{book['title']} by {book['author']}. {book.get('description', '')}"
    if mode == "no-subjects":
        return base
    if mode != "full":
        raise ValueError(f"unknown text mode: {mode!r}")
    subjects = ", ".join(book.get("subjects", []))
    return f"{base} Themes: {subjects}."
