"""Tinder for books — Streamlit UI over the recommender service.

Onboard by naming books you love, then swipe (like / haven't read / dislike).
The adaptive-hybrid recommender updates your taste after every swipe.

Run:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import random
from pathlib import Path

import streamlit as st

from app.service import BookRecommenderService
from app.store import DATA

st.set_page_config(page_title="Book match", page_icon=":material/menu_book:")

CARDS_PER_FETCH = 12
MIN_SEEDS = 3


# --- engine (shared across sessions) ------------------------------------------

@st.cache_resource
def get_service() -> BookRecommenderService:
    # One shared service + DB, keyed by user_id; safe across Streamlit threads.
    return BookRecommenderService(db_path=DATA / "app.db", check_same_thread=False)


@st.cache_resource
def filter_options():
    svc = get_service()
    langs = sorted({(b.get("language") or "en") for b in svc.catalog.books})
    years = [b["year"] for b in svc.catalog.books if b.get("year")]
    return {
        "languages": langs,
        "genres": svc.genres()[:40],
        "year_min": min(years),
        "year_max": max(years),
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
    st.session_state.seeds = seeds            # book_id -> title, from stored likes
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
    return {
        "languages": st.session_state.get("f_langs") or None,
        "genres": st.session_state.get("f_genres") or None,
        "year_min": (st.session_state.get("f_years") or (opts["year_min"], opts["year_max"]))[0],
        "year_max": (st.session_state.get("f_years") or (opts["year_min"], opts["year_max"]))[1],
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
        with st.form("save_profile", border=False, clear_on_submit=True):
            with st.container(horizontal=True, vertical_alignment="bottom"):
                st.text_input("Profile name", key="new_profile_name",
                              placeholder="e.g. Alex", label_visibility="collapsed")
                st.form_submit_button("Save", icon=":material/save:",
                                      on_click=save_profile)

    profiles = svc.list_profiles()
    if profiles:
        names = [p["name"] for p in profiles]
        st.selectbox("Switch profile", ["Switch profile…"] + names,
                     key="profile_switch", on_change=switch_profile,
                     label_visibility="collapsed")
    st.button("New profile", icon=":material/person_add:", on_click=new_profile)

    st.subheader("Filters", anchor=False)
    st.multiselect("Language", opts["languages"], key="f_langs",
                   placeholder="Any language")
    st.multiselect("Genres", opts["genres"], key="f_genres",
                   placeholder="Any genre")
    st.slider("Publication year", opts["year_min"], opts["year_max"],
              value=(opts["year_min"], opts["year_max"]), key="f_years")

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
    st.caption("We'll find your next favourites from there. Add at least "
               f"{MIN_SEEDS}.")

    with st.form("seed_search", border=False, clear_on_submit=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            query = st.text_input("Search a title", placeholder="e.g. The Hobbit",
                                  label_visibility="collapsed")
            submitted = st.form_submit_button("Search", icon=":material/search:")
    if submitted and query:
        st.session_state.search_hits = svc.search_titles(query, k=5)

    for m in st.session_state.search_hits:
        if m.book_id in st.session_state.seeds:
            continue
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{m.title}** — {m.author}")
            st.button("Add", icon=":material/add:", key=f"add_{m.book_id}",
                      on_click=add_seed, args=(m.book_id, m.title))

    if st.session_state.seeds:
        st.subheader("Your picks", anchor=False)
        for bid, title in list(st.session_state.seeds.items()):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f":material/favorite: {title}")
                st.button("Remove", icon=":material/close:", key=f"rm_{bid}",
                          on_click=remove_seed, args=(bid,))

    st.button(
        f"Start swiping ({len(st.session_state.seeds)}/{MIN_SEEDS})",
        icon=":material/swipe:", type="primary",
        disabled=len(st.session_state.seeds) < MIN_SEEDS,
        on_click=start_swiping,
    )


# --- swiping phase ------------------------------------------------------------

else:
    wish_n = svc.profile_summary(st.session_state.user_id)["interested"]
    discover, for_you, reading = st.tabs(
        ["Discover", "For you", f"Reading list ({wish_n})"]
    )

    with discover:
        if not st.session_state.queue:
            st.session_state.queue = svc.next_cards(
                st.session_state.user_id, n=CARDS_PER_FETCH,
                rng=random.Random(), **current_filters(),
            )

        if not st.session_state.queue:
            st.info("No more books match your filters — loosen them in the sidebar.",
                    icon=":material/filter_alt_off:")
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
                st.button("Pass", icon=":material/thumb_down:",
                          on_click=do_swipe, args=(book["id"], "dislike"))
                st.button("Haven't read", icon=":material/help:",
                          on_click=do_swipe, args=(book["id"], "skip"))
                st.button("Interested", icon=":material/bookmark_add:",
                          on_click=do_swipe, args=(book["id"], "interested"))
                st.button("Like", icon=":material/thumb_up:", type="primary",
                          on_click=do_swipe, args=(book["id"], "like"))
            st.caption(f"{len(st.session_state.queue)} cards queued · "
                       f"{svc.profile_summary(st.session_state.user_id)['like']} liked so far")

    with for_you:
        recs = svc.recommendations(st.session_state.user_id, n=12, **current_filters())
        if not recs:
            st.caption("Like a few books to unlock recommendations.")
        else:
            st.caption("Save a book to your reading list, or dismiss it to refine "
                       "your taste — the list refreshes as you go.")
            cols = st.columns(3)
            for i, r in enumerate(recs):
                with cols[i % 3]:
                    with st.container(border=True):
                        cover(st, r.book)
                        st.markdown(f"**{r.book['title'][:44]}**")
                        st.caption(r.book.get("author", ""))
                        if genre_badges(r.book, n=2):
                            st.markdown(genre_badges(r.book, n=2))
                        with st.container(horizontal=True):
                            st.button("Save", icon=":material/bookmark_add:",
                                      key=f"fy_save_{r.book['id']}", on_click=react,
                                      args=(r.book["id"], "interested"))
                            st.button("Not for me", icon=":material/close:",
                                      key=f"fy_no_{r.book['id']}", on_click=react,
                                      args=(r.book["id"], "dislike"))

    with reading:
        wish = svc.wishlist(st.session_state.user_id)
        if not wish:
            st.caption("Books you mark **Interested** land here — your saved "
                       "reading list.")
        else:
            st.caption("Your saved books. Remove one to take it off the list.")
            cols = st.columns(3)
            for i, book in enumerate(wish):
                with cols[i % 3]:
                    with st.container(border=True):
                        cover(st, book)
                        st.markdown(f"**{book['title'][:44]}**")
                        st.caption(book.get("author", ""))
                        if genre_badges(book, n=2):
                            st.markdown(genre_badges(book, n=2))
                        st.button("Remove", icon=":material/close:",
                                  key=f"wl_rm_{book['id']}", on_click=react,
                                  args=(book["id"], "skip"))
