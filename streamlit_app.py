"""Streamlit UI over the recommender service.

Onboard by naming books you love, then swipe (like / interested / haven't read /
pass). The adaptive-hybrid recommender updates your taste after every swipe.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import csv
import io
import random

import streamlit as st

from app.library import parse_library
from app.search import Match
from app.service import BookRecommenderService
from app.store import DATA

st.set_page_config(page_title="Book match", page_icon=":material/menu_book:")

CARDS_PER_FETCH = 12
MIN_SEEDS = 3

# The catalog runs back to -1750 (the Epic of Gilgamesh), but ~98% of it is
# post-1859 -- a linear slider over the true min/max gives the antiquity tail 97%
# of the travel and squeezes the half of the catalog published since 2016 into a
# few pixels. So the slider bottoms out at this percentile and treats that floor
# as open-ended ("and earlier"), which keeps the old books reachable.
YEAR_FLOOR_PCT = 1


# --- engine (shared across sessions) ------------------------------------------


@st.cache_resource
def get_service() -> BookRecommenderService:
    # One shared service + DB, keyed by user_id; safe across Streamlit threads.
    return BookRecommenderService(db_path=DATA / "app.db", check_same_thread=False)


@st.cache_resource
def filter_options():
    svc = get_service()
    langs = sorted({(b.get("language") or "en") for b in svc.catalog.books})
    years = sorted(b["year"] for b in svc.catalog.books if b.get("year"))
    floor = years[len(years) * YEAR_FLOOR_PCT // 100] // 10 * 10  # round down to the decade
    return {
        "languages": langs,
        "genres": svc.genres()[:40],
        "year_floor": min(floor, years[-1]),  # a tiny catalog could put the floor at the top
        "year_max": years[-1],
    }


svc = get_service()
opts = filter_options()


# --- durable identity ---------------------------------------------------------
# The user_id lives in the URL (?uid=...), so a page reload or server restart
# resumes the same profile instead of silently minting a throwaway one. Swipes
# are keyed by this id in SQLite, so nothing is lost across sessions.


def bind_user(uid: str) -> None:
    """Point this session at ``uid`` and rebuild its state from the DB."""
    st.session_state.user_id = uid
    st.query_params["uid"] = uid
    seeds = svc.liked_titles(uid)
    st.session_state.seeds = seeds  # book_id -> title, from stored likes
    st.session_state.phase = "swiping" if len(seeds) >= MIN_SEEDS else "onboarding"
    st.session_state.search_hits = []
    st.session_state.queue = []


if "user_id" not in st.session_state:
    qp_uid = st.query_params.get("uid")
    bind_user(qp_uid if qp_uid and svc.user_exists(qp_uid) else svc.new_user(""))

st.session_state.setdefault("filters_sig", None)


# --- callbacks ----------------------------------------------------------------


def save_profile() -> None:
    name = (st.session_state.get("new_profile_name") or "").strip()
    if name:
        svc.name_profile(st.session_state.user_id, name)


def new_profile() -> None:
    bind_user(svc.new_user(""))


def switch_profile() -> None:
    pick = st.session_state.get("profile_switch")
    ids = {p["name"]: p["id"] for p in svc.list_profiles()}
    if pick in ids and ids[pick] != st.session_state.user_id:
        bind_user(ids[pick])


def add_seed(book_id: str, title: str) -> None:
    svc.swipe(st.session_state.user_id, book_id, "like")
    st.session_state.seeds[book_id] = title


def remove_seed(book_id: str) -> None:
    svc.swipe(st.session_state.user_id, book_id, "skip")  # unlike -> mark seen
    st.session_state.seeds.pop(book_id, None)


def import_library_cb() -> None:
    """Parse an uploaded reading list, match it to the catalog, and seed likes."""
    up = st.session_state.get("library_file")
    if not up:
        return
    entries = parse_library(up.name, up.getvalue())
    result = svc.import_library(st.session_state.user_id, entries)
    for me in result.matched:
        st.session_state.seeds[me.match.book_id] = me.match.title
    st.session_state.import_summary = {
        "matched": result.n_matched,
        "total": len(entries),
        "unmatched": [e.label() for e in result.unmatched],
    }


def web_search_cb() -> None:
    q = st.session_state.get("last_query", "")
    st.session_state.web_hits = svc.external_search(q, k=5) if q else []


def add_web_book(record: dict) -> None:
    """Ingest a book we didn't have (from Open Library) and seed it as a like."""
    bid = svc.add_external_book(record)
    svc.swipe(st.session_state.user_id, bid, "like")
    st.session_state.seeds[bid] = record["title"]
    st.session_state.web_hits = [h for h in st.session_state.get("web_hits", []) if h["id"] != bid]


def start_swiping() -> None:
    st.session_state.phase = "swiping"
    st.session_state.queue = []


def do_swipe(book_id: str, reaction: str) -> None:
    svc.swipe(st.session_state.user_id, book_id, reaction)
    st.session_state.queue = [c for c in st.session_state.queue if c.book["id"] != book_id]


def react(book_id: str, reaction: str) -> None:
    """Record a reaction from For You / Reading list (no swipe queue to prune)."""
    svc.swipe(st.session_state.user_id, book_id, reaction)


def restart() -> None:
    bind_user(svc.new_user(""))


def current_filters() -> dict:
    lo, hi = st.session_state.get("f_years") or (opts["year_floor"], opts["year_max"])
    return {
        "languages": st.session_state.get("f_langs") or None,
        "genres": st.session_state.get("f_genres") or None,
        # At the floor the bound is open-ended: drop it so the pre-floor books
        # (Homer, Gilgamesh) still come through rather than being filtered out.
        "year_min": None if lo <= opts["year_floor"] else lo,
        "year_max": hi,
    }


# --- shared render helpers ----------------------------------------------------


def genre_badges(book: dict, n: int = 3) -> str:
    colors = ["blue", "violet", "green", "orange"]
    subs = book.get("subjects", [])[:n]
    return " ".join(f":{colors[i % len(colors)]}-badge[{s}]" for i, s in enumerate(subs))


def cover(container, book: dict) -> None:
    img = book.get("image") or ""
    if img and "nophoto" not in img:
        container.image(img, width="stretch")
    else:
        container.markdown(":material/menu_book:")


def more_like_this(book_id: str, key: str) -> None:
    """A 'More like this' popover listing similar books, each saveable."""
    with st.popover("More like this", icon=":material/travel_explore:"):
        for s in svc.similar_books(book_id, n=6):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{s.book['title'][:34]}** · {(s.book.get('author') or '')[:22]}")
                st.button(
                    "Save",
                    icon=":material/bookmark_add:",
                    key=f"more_{key}_{s.book['id']}",
                    on_click=react,
                    args=(s.book["id"], "interested"),
                )


def reading_list_csv(books: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Title", "Author", "Year"])
    for b in books:
        w.writerow([b.get("title", ""), b.get("author", ""), b.get("year", "") or ""])
    return buf.getvalue().encode("utf-8")


# --- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.header("Book match", anchor=False)
    st.caption("Swipe your way to your next read.")

    st.subheader("Profile", anchor=False)
    current_name = svc.profile_name(st.session_state.user_id)
    if current_name:
        st.caption(f":material/account_circle: Signed in as **{current_name}**")
    else:
        st.caption("Unsaved profile — name it to keep it and switch back later.")
        with (
            st.form("save_profile", border=False, clear_on_submit=True),
            st.container(horizontal=True, vertical_alignment="bottom"),
        ):
            st.text_input(
                "Profile name",
                key="new_profile_name",
                placeholder="e.g. Alex",
                label_visibility="collapsed",
            )
            st.form_submit_button("Save", icon=":material/save:", on_click=save_profile)

    profiles = svc.list_profiles()
    if profiles:
        names = [p["name"] for p in profiles]
        st.selectbox(
            "Switch profile",
            ["Switch profile…", *names],
            key="profile_switch",
            on_change=switch_profile,
            label_visibility="collapsed",
        )
    st.button("New profile", icon=":material/person_add:", on_click=new_profile)

    st.subheader("Filters", anchor=False)
    st.multiselect("Language", opts["languages"], key="f_langs", placeholder="Any language")
    st.multiselect("Genres", opts["genres"], key="f_genres", placeholder="Any genre")
    st.slider(
        "Publication year",
        opts["year_floor"],
        opts["year_max"],
        value=(opts["year_floor"], opts["year_max"]),
        key="f_years",
    )
    _lo, _hi = st.session_state.get("f_years") or (opts["year_floor"], opts["year_max"])
    if _lo <= opts["year_floor"]:
        st.caption(f"{opts['year_floor']} and earlier → {_hi} (includes all of antiquity)")

    counts = svc.profile_summary(st.session_state.user_id)
    st.subheader("Your taste so far", anchor=False)
    with st.container(horizontal=True):
        st.metric("Liked", counts["like"])
        st.metric("Wishlist", counts["interested"])
        st.metric("Passed", counts["dislike"])
        st.metric("Skipped", counts["skip"])

    st.button("Start over", icon=":material/restart_alt:", on_click=restart)


# clear the swipe queue when filters change, so new filters take effect
sig = str(current_filters())
if sig != st.session_state.filters_sig:
    st.session_state.filters_sig = sig
    st.session_state.queue = []


# --- onboarding phase ---------------------------------------------------------

if st.session_state.phase == "onboarding":
    st.title("Name a few books you love", anchor=False)
    st.caption(f"We'll find your next favourites from there. Add at least {MIN_SEEDS}.")

    by_meaning = st.toggle(
        "Search by meaning",
        key="semantic_mode",
        help="Describe what you're after (e.g. 'a lonely lighthouse keeper') instead of a title.",
    )
    with (
        st.form("seed_search", border=False, clear_on_submit=True),
        st.container(horizontal=True, vertical_alignment="bottom"),
    ):
        placeholder = "e.g. slow-burn space opera" if by_meaning else "e.g. The Hobbit"
        query = st.text_input("Search", placeholder=placeholder, label_visibility="collapsed")
        submitted = st.form_submit_button("Search", icon=":material/search:")
    if submitted and query:
        st.session_state.last_query = query
        st.session_state.web_hits = []  # reset any prior online results
        if by_meaning:
            hits = svc.semantic_search(query, k=6)
            if hits:
                st.session_state.search_hits = [
                    Match(b["id"], b["title"], b.get("author", ""), 0.0) for b in hits
                ]
            else:  # encoder unavailable -> fall back to title search
                st.session_state.search_hits = svc.search_titles(query, k=5)
        else:
            st.session_state.search_hits = svc.search_titles(query, k=5)

    with st.expander("…or import your reading list (CSV, TSV, TXT, XLSX)"):
        st.file_uploader(
            "Reading list",
            type=["csv", "tsv", "txt", "xlsx"],
            key="library_file",
            on_change=import_library_cb,
            label_visibility="collapsed",
            help="A Goodreads/StoryGraph export or any list with a title (and optional author) column.",
        )
        summary = st.session_state.get("import_summary")
        if summary:
            st.success(f"Matched {summary['matched']} of {summary['total']} books to the catalog.")
            if summary["unmatched"]:
                st.caption(f"Couldn't match {len(summary['unmatched'])}:")
                st.markdown("\n".join(f"- {u}" for u in summary["unmatched"][:50]))
                if len(summary["unmatched"]) > 50:
                    st.caption(f"…and {len(summary['unmatched']) - 50} more.")

    for m in st.session_state.search_hits:
        if m.book_id in st.session_state.seeds:
            continue
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{m.title}** — {m.author}")
            st.button(
                "Add",
                icon=":material/add:",
                key=f"add_{m.book_id}",
                on_click=add_seed,
                args=(m.book_id, m.title),
            )

    # Not in our catalog? Fetch it from Open Library and add it (CF-cold).
    last_q = st.session_state.get("last_query", "")
    if last_q and not st.session_state.get("semantic_mode"):
        with st.expander(f"Not seeing it? Search the web for '{last_q[:40]}'"):
            st.button("Search Open Library", icon=":material/public:", on_click=web_search_cb)
            for r in st.session_state.get("web_hits", []):
                if r["id"] in st.session_state.seeds:
                    continue
                yr = f" ({r['year']})" if r.get("year") else ""
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"**{r['title'][:44]}** — {(r.get('author') or '')[:26]}{yr}")
                    st.button(
                        "Add",
                        icon=":material/add:",
                        key=f"web_add_{r['id']}",
                        on_click=add_web_book,
                        args=(r,),
                    )

    if st.session_state.seeds:
        st.subheader("Your picks", anchor=False)
        for bid, title in list(st.session_state.seeds.items()):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f":material/favorite: {title}")
                st.button(
                    "Remove",
                    icon=":material/close:",
                    key=f"rm_{bid}",
                    on_click=remove_seed,
                    args=(bid,),
                )

    st.button(
        f"Start swiping ({len(st.session_state.seeds)}/{MIN_SEEDS})",
        icon=":material/swipe:",
        type="primary",
        disabled=len(st.session_state.seeds) < MIN_SEEDS,
        on_click=start_swiping,
    )


# --- swiping phase ------------------------------------------------------------

else:
    wish_n = svc.profile_summary(st.session_state.user_id)["interested"]
    discover, for_you, surprise_tab, reading = st.tabs(
        ["Discover", "For you", "Surprise me", f"Reading list ({wish_n})"]
    )

    with discover:
        if not st.session_state.queue:
            st.session_state.queue = svc.next_cards(
                st.session_state.user_id,
                n=CARDS_PER_FETCH,
                rng=random.Random(),
                **current_filters(),
            )

        if not st.session_state.queue:
            st.info(
                "No more books match your filters — loosen them in the sidebar.",
                icon=":material/filter_alt_off:",
            )
        else:
            card = st.session_state.queue[0]
            book = card.book
            with st.container(border=True):
                left, right = st.columns([1, 2])
                cover(left, book)
                with right:
                    st.subheader(book["title"], anchor=False)
                    yr = f" · {book['year']}" if book.get("year") else ""
                    st.caption(f"{book.get('author', '')}{yr}")
                    if genre_badges(book):
                        st.markdown(genre_badges(book))
                    desc = (book.get("description") or "").strip()
                    st.write(desc[:400] + ("…" if len(desc) > 400 else "") or "_No description._")
                    driver = "readers like you" if card.cf_weight >= 0.5 else "similar themes"
                    st.caption(f":material/recommend: Suggested from {driver}")

            with st.container(horizontal=True, horizontal_alignment="center"):
                st.button(
                    "Pass",
                    icon=":material/thumb_down:",
                    on_click=do_swipe,
                    args=(book["id"], "dislike"),
                )
                st.button(
                    "Haven't read",
                    icon=":material/help:",
                    on_click=do_swipe,
                    args=(book["id"], "skip"),
                )
                st.button(
                    "Interested",
                    icon=":material/bookmark_add:",
                    on_click=do_swipe,
                    args=(book["id"], "interested"),
                )
                st.button(
                    "Like",
                    icon=":material/thumb_up:",
                    type="primary",
                    on_click=do_swipe,
                    args=(book["id"], "like"),
                )
            st.caption(
                f"{len(st.session_state.queue)} cards queued · "
                f"{svc.profile_summary(st.session_state.user_id)['like']} liked so far"
            )

    with for_you:
        recs = svc.recommendations(st.session_state.user_id, n=12, **current_filters())
        if not recs:
            st.caption("Like a few books to unlock recommendations.")
        else:
            st.caption(
                "Save a book to your reading list, or dismiss it to refine "
                "your taste — the list refreshes as you go."
            )
            cols = st.columns(3)
            for i, r in enumerate(recs):
                with cols[i % 3], st.container(border=True):
                    cover(st, r.book)
                    st.markdown(f"**{r.book['title'][:44]}**")
                    st.caption(r.book.get("author", ""))
                    if genre_badges(r.book, n=2):
                        st.markdown(genre_badges(r.book, n=2))
                    if r.explanation:
                        st.caption(f":material/lightbulb: {r.explanation}")
                    with st.container(horizontal=True):
                        st.button(
                            "Save",
                            icon=":material/bookmark_add:",
                            key=f"fy_save_{r.book['id']}",
                            on_click=react,
                            args=(r.book["id"], "interested"),
                        )
                        st.button(
                            "Not for me",
                            icon=":material/close:",
                            key=f"fy_no_{r.book['id']}",
                            on_click=react,
                            args=(r.book["id"], "dislike"),
                        )
                    more_like_this(r.book["id"], key=f"fy_{r.book['id']}")

    with surprise_tab:
        surprises = svc.surprises(st.session_state.user_id, n=9, **current_filters())
        if not surprises:
            st.caption(
                "Like a few books first — then we'll find reads that are "
                "**nothing like** your usual taste but that readers like "
                "you still love."
            )
        else:
            st.caption(
                "Wildcards: far from what you usually read, but readers "
                "with your taste rate them highly. Save one or dismiss it."
            )
            cols = st.columns(3)
            for i, s in enumerate(surprises):
                with cols[i % 3], st.container(border=True):
                    cover(st, s.book)
                    st.markdown(f"**{s.book['title'][:44]}**")
                    st.caption(s.book.get("author", ""))
                    if genre_badges(s.book, n=2):
                        st.markdown(genre_badges(s.book, n=2))
                    st.caption(
                        f":material/auto_awesome: {round(s.novelty * 100)}% "
                        "outside your usual — CF-driven"
                    )
                    with st.container(horizontal=True):
                        st.button(
                            "Save",
                            icon=":material/bookmark_add:",
                            key=f"sp_save_{s.book['id']}",
                            on_click=react,
                            args=(s.book["id"], "interested"),
                        )
                        st.button(
                            "Not for me",
                            icon=":material/close:",
                            key=f"sp_no_{s.book['id']}",
                            on_click=react,
                            args=(s.book["id"], "dislike"),
                        )

    with reading:
        wish = svc.wishlist(st.session_state.user_id)
        if not wish:
            st.caption("Books you mark **Interested** land here — your saved reading list.")
        else:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"{len(wish)} saved. Mark one read, remove it, or export the list.")
                st.download_button(
                    "Export CSV",
                    data=reading_list_csv(wish),
                    file_name="reading-list.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
            cols = st.columns(3)
            for i, book in enumerate(wish):
                with cols[i % 3], st.container(border=True):
                    cover(st, book)
                    st.markdown(f"**{book['title'][:44]}**")
                    st.caption(book.get("author", ""))
                    if genre_badges(book, n=2):
                        st.markdown(genre_badges(book, n=2))
                    with st.container(horizontal=True):
                        st.button(
                            "Read + liked",
                            icon=":material/check:",
                            key=f"wl_read_{book['id']}",
                            help="Mark as read and add to your likes (sharpens recommendations).",
                            on_click=react,
                            args=(book["id"], "like"),
                        )
                        st.button(
                            "Remove",
                            icon=":material/close:",
                            key=f"wl_rm_{book['id']}",
                            on_click=react,
                            args=(book["id"], "skip"),
                        )
                    more_like_this(book["id"], key=f"wl_{book['id']}")
