"""Representation programs: functions that turn a raw book record into the text
we embed.

This is the one transferable idea from AutoIndex (arXiv 2607.18603) applied to
our content channel: the item->text transform is a *tunable artifact*, not a
fixed given. We hold the encoder fixed (the served co-read bge-small) and vary
only what text it sees, then let the held-out harness decide whether the change
helped -- exactly the paper's validation-guided selection, minus its expensive
LLM search loop.

``REGISTRY`` maps a name to a callable ``book -> str``. ``full`` is the current
production representation (``eval.data.book_to_text`` in ``full`` mode); every
other entry is a grounded edit measured against it in ``eval.repr_sweep``.

Grounding
---------
Google-Books descriptions in ``real_books.json`` are hard-capped at 800 chars
(p50 783, max 800), so a leading marketing sentence literally *displaces plot*
from the text the encoder sees. Real examples that motivated the stripper:

* "An incredible, brand-new spin-off series from the best-selling creator of
  Percy Jackson." -- a whole sentence of promo before any plot.
* "Book two, a continuation of the Tom Gray series." -- leading series bookkeeping.
* "...Brent Weeks' New York Times bestselling Night Angel trilogy." -- promo
  tokens interleaved with real content.
* "The Boudreaux Series--Sexy. Intriguing. Easy." -- a tagline lead-in.

The stripper is deliberately conservative (removing real content is worse than
leaving promo in); the harness, not the regex, is the arbiter.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .data import book_to_text

# --- boilerplate stripping ---------------------------------------------------

# Bestseller / award marketing. The first alternative catches the common
# "<outlet> bestselling" forms as a unit; the rest catch standalone promo tokens.
_MARKETING = re.compile(
    r"""(?:
        \b(?:\#?\s?1\s+)?(?:new\s+york\s+times|usa\s+today|sunday\s+times|
            wall\s+street\s+journal|international|national|instant|\#?\s?1)\s+
            (?:number\s+one\s+)?bestsell(?:er|ing)\b
      | \bbest[\s-]?sell(?:er|ing)\b
      | \baward[\s-]?winning\b
      | \bnew\s+york\s+times\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# "From the (best-selling) author/creator of <X>." -- pure attribution marketing.
_FROM_THE = re.compile(
    r"from the (?:[\w'-]+\s+){0,4}?(?:author|creator|writer|mind|team|pen)\s+of\b[^.]*\.\s*",
    re.IGNORECASE,
)

# Leading series-bookkeeping sentence: "Book two, a continuation of the X series."
_SERIES_LEAD = re.compile(
    r"^\s*(?:book\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    r"|the\s+\w+\s+book|volume\s+\w+)\b[^.]*\bseries\b[^.]*\.\s*",
    re.IGNORECASE,
)

_MULTISPACE = re.compile(r"\s{2,}")
_LEADING_PUNCT = re.compile(r"^[\s.,;:!?/'\"-]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")


def strip_boilerplate(desc: str) -> str:
    """Remove marketing / series-bookkeeping noise from a description.

    Conservative by design: it deletes promo phrases in place and tidies the
    seams, but never touches plot text. Idempotent.
    """
    if not desc:
        return desc
    d = _SERIES_LEAD.sub("", desc)
    d = _FROM_THE.sub("", d)
    d = _MARKETING.sub("", d)
    d = _SPACE_BEFORE_PUNCT.sub(r"\1", d)
    d = _MULTISPACE.sub(" ", d)
    d = _LEADING_PUNCT.sub("", d)
    return d.strip()


# --- representation programs (book -> embedding text) ------------------------


def _compose(title: str, author: str, desc: str, subjects: list[str], subj_repeat: int = 1) -> str:
    """Assemble in the exact shape of the production template, so a variant
    differs from ``full`` only in the ways we intend.

    ``full`` is ``"{title} by {author}. {desc} Themes: {subjects}."``;
    ``subj_repeat`` > 1 appends the Themes clause again (a dense-pooling analogue
    of BM25F field up-weighting: repeating a phrase raises its weight in the
    mean-pooled vector).
    """
    base = f"{title} by {author}. {desc}"
    themes = ", ".join(subjects)
    return base + (f" Themes: {themes}." * subj_repeat)


def rep_full(book: dict) -> str:
    # The true production baseline -- call it directly so it stays identical.
    return book_to_text(book, mode="full")


def rep_strip_boilerplate(book: dict) -> str:
    return _compose(
        book["title"],
        book["author"],
        strip_boilerplate(book.get("description", "")),
        book.get("subjects", []),
    )


def rep_repeat_subjects(book: dict) -> str:
    return _compose(
        book["title"],
        book["author"],
        book.get("description", ""),
        book.get("subjects", []),
        subj_repeat=2,
    )


def rep_strip_repeat(book: dict) -> str:
    return _compose(
        book["title"],
        book["author"],
        strip_boilerplate(book.get("description", "")),
        book.get("subjects", []),
        subj_repeat=2,
    )


REGISTRY: dict[str, Callable[[dict], str]] = {
    "full": rep_full,  # production baseline
    "strip_boilerplate": rep_strip_boilerplate,  # noise removal (most likely to transfer)
    "repeat_subjects": rep_repeat_subjects,  # BM25F-style genre up-weighting
    "strip_repeat": rep_strip_repeat,  # both
}
