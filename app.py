"""What Can I Cook? - Streamlit UI.

Wired to the full local RAG pipeline: CSV -> chunking -> embeddings -> FAISS
semantic search (`src.retriever.retrieve()`) for retrieval, and a local LLM
via Ollama (`src.chatbot.generate_response()`) for the grounded answer.
100% free and local - no paid API, nothing leaves your machine. One-time
setup: the Sentence Transformers model downloads on first use, and Ollama
needs `ollama pull <model>` run once (see README).
"""

import html

import pandas as pd
import streamlit as st

from src import chatbot, config
from src.document_loader import load_recipes, load_substitutions
from src.retriever import CUISINE_GROUPS, TIME_LIMITS, display_name, match_ingredients, retrieve

CUISINE_OPTIONS = ["Any", "Indian", "Italian", "Asian"]
DIET_OPTIONS = ["Any", "Vegetarian", "Vegan"]
TIME_OPTIONS = ["Any", "Under 15 min", "Under 30 min", "Under 60 min"]
DIFFICULTY_OPTIONS = ["Any", "Easy", "Medium"]

st.set_page_config(
    page_title="What Can I Cook?",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1100px; }
    #MainMenu, footer { visibility: hidden; }

    .hero { padding: 2.2rem 0 1.4rem 0; border-bottom: 1px solid rgba(128,128,128,.18); margin-bottom: 2rem; }
    .hero h1 { font-size: 2.9rem; font-weight: 700; margin: 0 0 .5rem 0; letter-spacing: -.02em; }
    .hero p { font-size: 1.15rem; color: #b06a2c; margin: 0; font-weight: 500; }

    .section-title { font-size: 1.05rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: .08em; color: #8a8a8a; margin: 2.2rem 0 .9rem 0; }

    .card { border: 1px solid rgba(128,128,128,.22); border-radius: 12px;
        padding: 1.2rem 1.35rem; margin-bottom: 1rem; background: rgba(176,106,44,.04); }
    .card h4 { margin: 0 0 .5rem 0; font-size: 1.15rem; font-weight: 700; }
    .card .meta { font-size: .82rem; color: #8a8a8a; margin-bottom: .7rem; }
    .card .body { font-size: .92rem; line-height: 1.55; }

    .pill { display: inline-block; padding: .2rem .65rem; margin: 0 .35rem .35rem 0;
        border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: rgba(176,106,44,.14); color: #b06a2c; }
    .pill-missing { background: rgba(192,57,43,.12); color: #c0392b; }

    .match-badge { float: right; font-size: .76rem; font-weight: 600; color: #b06a2c;
        background: rgba(176,106,44,.12); padding: .15rem .65rem; border-radius: 999px; }

    .example { border-left: 3px solid #b06a2c; padding: .55rem .9rem; margin-bottom: .6rem;
        background: rgba(128,128,128,.06); border-radius: 0 6px 6px 0; font-size: .92rem; }

    /* Reskin the example question buttons to match the .example look above */
    div[data-testid="stButton"] button[kind="secondary"] {
        text-align: left; white-space: normal; width: 100%;
        border-left: 3px solid #b06a2c; border-top: none; border-right: none; border-bottom: none;
        background: rgba(128,128,128,.06); border-radius: 0 6px 6px 0;
        font-size: .92rem; padding: .55rem .9rem;
    }

    .step { display: flex; gap: .85rem; margin-bottom: .9rem; align-items: flex-start; }
    .step .num { flex: 0 0 26px; height: 26px; border-radius: 50%; background: #b06a2c;
        color: #fff; font-size: .8rem; font-weight: 700; display: flex;
        align-items: center; justify-content: center; }
    .step .txt { font-size: .93rem; line-height: 1.5; padding-top: .15rem; }

    .placeholder { border: 1px dashed rgba(128,128,128,.4); border-radius: 12px;
        padding: 1.6rem; text-align: center; color: #8a8a8a; font-size: .92rem; }

    .sidebar-stat { font-size: 2.1rem; font-weight: 700; color: #b06a2c; line-height: 1; }
    .sidebar-label { font-size: .82rem; color: #8a8a8a; text-transform: uppercase; letter-spacing: .06em; }
</style>
"""

# Each example fully specifies the filter state it wants, so clicking one is
# reproducible regardless of whatever the user had selected before.
EXAMPLES = [
    {"text": "I have potatoes and onions", "cuisine": "Any", "diet": "Any", "cooking_time": "Any", "difficulty": "Any"},
    {"text": "What can I make with paneer?", "cuisine": "Any", "diet": "Any", "cooking_time": "Any", "difficulty": "Any"},
    {"text": "I have rice and vegetables", "cuisine": "Any", "diet": "Any", "cooking_time": "Any", "difficulty": "Any"},
    {"text": "Give me something quick under 20 minutes", "cuisine": "Any", "diet": "Any", "cooking_time": "Under 30 min", "difficulty": "Any"},
]

RETRIEVAL_ERROR_MESSAGE = (
    "⚠️ We're having trouble preparing your recipe recommendations right now. "
    "Please try again in a moment."
)


@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both knowledge bases once per session (used for the sidebar stats)."""
    return load_recipes(), load_substitutions()


def _passes_filters(result: dict, cuisine: str, diet: str, cooking_time: str, difficulty: str) -> bool:
    """Apply the sidebar's cuisine/diet/time/difficulty filters to one
    semantically-retrieved result. Substitution-type results carry none of
    these fields, so they always pass through untouched - the filters only
    ever narrow down recipes.
    """
    if result.get("document_type") != "recipe":
        return True

    metadata = result.get("metadata", {})

    if cuisine != "Any":
        allowed = CUISINE_GROUPS.get(cuisine, {cuisine})
        if metadata.get("cuisine") not in allowed:
            return False

    if diet != "Any" and metadata.get("diet") != diet:
        return False

    if cooking_time != "Any":
        limit = TIME_LIMITS.get(cooking_time)
        value = metadata.get("cooking_time")
        if limit is not None and (value is None or value > limit):
            return False

    if difficulty != "Any" and metadata.get("difficulty") != difficulty:
        return False

    return True


def _match_substitution(missing_ingredient: str, substitution_results: list[dict]) -> dict | None:
    """Find a retrieved substitution result whose `missing_ingredient` matches.

    Only looks within what `retrieve()` actually returned for this query -
    no separate CSV lookup - so if nothing relevant was retrieved for this
    ingredient, nothing is shown. That's the honest RAG behavior rather
    than a hard-coded fallback.
    """
    ingredient_norm = missing_ingredient.strip().lower()
    for sub_result in substitution_results:
        name = str(sub_result["metadata"].get("missing_ingredient", "")).strip().lower()
        if name and (name in ingredient_norm or ingredient_norm in name):
            return sub_result
    return None


def _run_search(query: str, cuisine: str, diet: str, cooking_time: str, difficulty: str) -> None:
    """The full RAG pipeline for one query: retrieve -> filter -> generate.

    Persists everything into session_state so it survives unrelated reruns
    (e.g. expanding a "View Recipe" panel).
    """
    st.session_state["has_searched"] = True
    st.session_state["retrieval_error"] = None
    st.session_state["ai_answer"] = None
    st.session_state["results"] = []

    with st.spinner("Finding recipes and preparing your personalized suggestions..."):
        try:
            raw_results = retrieve(query)
        except Exception as exc:
            # Typically the local embedding model failing to load - see README.
            print(f"[app] retrieval failed: {type(exc).__name__}: {exc}")
            st.session_state["retrieval_error"] = RETRIEVAL_ERROR_MESSAGE
            return

        filtered = [r for r in raw_results if _passes_filters(r, cuisine, diet, cooking_time, difficulty)]
        st.session_state["results"] = filtered

        if filtered:
            # generate_response() never raises - Ollama-unavailable and other
            # generation errors already come back as a friendly message.
            st.session_state["ai_answer"] = chatbot.generate_response(query, filtered)
        # else: no results survived the relevance threshold + filters - the
        # no-results state is rendered below, and the local LLM is never called.


def _clear_filters() -> None:
    """Callback: reset every filter, the ingredient box, and any results."""
    st.session_state["cuisine_filter"] = "Any"
    st.session_state["diet_filter"] = "Any"
    st.session_state["time_filter"] = "Any"
    st.session_state["difficulty_filter"] = "Any"
    st.session_state["ingredient_query"] = ""
    st.session_state["has_searched"] = False
    st.session_state["results"] = []
    st.session_state["ai_answer"] = None
    st.session_state["retrieval_error"] = None


def _apply_filters() -> None:
    """Callback: sidebar search button - run a search with whatever's in the box."""
    st.session_state["trigger_search"] = True


def render_sidebar(recipes: pd.DataFrame, substitutions: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("### 🍳 Recipe Collection")
        st.markdown(
            f'<div class="sidebar-stat">{len(recipes)}</div>'
            '<div class="sidebar-label">Recipes available</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-stat">{len(substitutions)}</div>'
            '<div class="sidebar-label">Substitution ideas</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Filters**")
        st.selectbox("Cuisine", CUISINE_OPTIONS, key="cuisine_filter")
        st.selectbox("Diet", DIET_OPTIONS, key="diet_filter")
        st.selectbox("Cooking time", TIME_OPTIONS, key="time_filter")
        st.selectbox("Difficulty", DIFFICULTY_OPTIONS, key="difficulty_filter")

        col1, col2 = st.columns(2)
        with col1:
            st.button("🔍 Search", on_click=_apply_filters, type="primary", use_container_width=True)
        with col2:
            st.button("Clear filters", on_click=_clear_filters, use_container_width=True)
        st.caption("Search runs the filters above against whatever's in the ingredient box.")

        st.divider()
        st.caption("Your AI Recipe Assistant matches what you have with our recipe collection, "
                   "suggests dietary-friendly options, and recommends smart substitutes - "
                   "completely free, and everything stays on your device.")


def render_hero() -> None:
    st.markdown(
        f'<div class="hero"><h1>{config.APP_TITLE}</h1>'
        f"<p>{config.APP_TAGLINE}</p></div>",
        unsafe_allow_html=True,
    )


def _use_example(example: dict) -> None:
    """Callback: populate the search box + filters from an example and run it."""
    st.session_state["ingredient_query"] = example["text"]
    st.session_state["cuisine_filter"] = example["cuisine"]
    st.session_state["diet_filter"] = example["diet"]
    st.session_state["time_filter"] = example["cooking_time"]
    st.session_state["difficulty_filter"] = example["difficulty"]
    st.session_state["trigger_search"] = True


def render_search() -> None:
    st.markdown('<div class="section-title">Your kitchen</div>', unsafe_allow_html=True)

    st.text_area(
        "Ingredients or question",
        key="ingredient_query",
        placeholder="e.g. paneer, tomatoes, onion, cream, garam masala — what can I cook tonight?",
        height=130,
        label_visibility="collapsed",
    )

    left, right = st.columns([1, 3])
    with left:
        find_clicked = st.button("✨ Find Recipes", type="primary", use_container_width=True)
    with right:
        st.caption("Filters live in the sidebar. List what you have, or ask in plain language.")

    # A search runs either because the button was clicked, or because an
    # example question / the sidebar search button set the trigger flag.
    should_search = find_clicked or st.session_state.get("trigger_search", False)
    st.session_state["trigger_search"] = False

    if should_search:
        query = st.session_state.get("ingredient_query", "")
        if not query.strip():
            st.warning("Tell me what is in your kitchen first.")
        else:
            cuisine = st.session_state.get("cuisine_filter", "Any")
            diet = st.session_state.get("diet_filter", "Any")
            cooking_time = st.session_state.get("time_filter", "Any")
            difficulty = st.session_state.get("difficulty_filter", "Any")
            _run_search(query, cuisine, diet, cooking_time, difficulty)


def render_examples() -> None:
    st.markdown('<div class="section-title">Try asking</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        with left if i % 2 == 0 else right:
            st.button(
                example["text"],
                key=f"example_{i}",
                use_container_width=True,
                on_click=_use_example,
                args=(example,),
            )


def render_ai_answer(results: list[dict]) -> None:
    """The grounded answer from the local LLM, with the retrieved sources
    it was generated from - so it's clear this is a RAG chatbot, not a
    canned response.
    """
    answer = st.session_state.get("ai_answer")
    if not answer:
        return

    safe_answer = html.escape(answer).replace("\n", "<br>")
    sources = "".join(f'<span class="pill">{html.escape(display_name(r))}</span>' for r in results)

    st.markdown(
        f'<div class="card">'
        f"<h4>✨ Recipe Recommendation</h4>"
        f'<div class="body">{safe_answer}</div>'
        f'<div class="body" style="margin-top:.6rem;"><b>📚 Recipe sources:</b><br>{sources}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_recipe_card(result: dict, query: str, substitution_results: list[dict]) -> set:
    """Renders one recipe card + its View Recipe expander.

    Returns the set of substitution_ids (if any) matched into its "make it
    yourself" section, so the caller can skip re-showing those as standalone
    substitution cards further down the page.
    """
    metadata = result.get("metadata", {})
    ingredients = metadata.get("ingredients", [])
    matched, missing = match_ingredients(ingredients, query)
    pct = max(0, round(result.get("score", 0.0) * 100))

    meta = (
        f'{metadata.get("cuisine", "?")} · {metadata.get("diet", "?")} · '
        f'{metadata.get("cooking_time", "?")} min · {metadata.get("difficulty", "?")}'
    )
    matched_pills = "".join(f'<span class="pill">{i}</span>' for i in matched) or "<em>none confirmed yet</em>"
    missing_pills = (
        "".join(f'<span class="pill pill-missing">{i}</span>' for i in missing)
        if missing
        else '<span class="pill">You have everything!</span>'
    )

    st.markdown(
        f'<div class="card">'
        f'<h4>{metadata.get("recipe_name", "Untitled recipe")}<span class="match-badge">{pct}% match</span></h4>'
        f'<div class="meta">{meta}</div>'
        f'<div class="body"><b>You have:</b><br>{matched_pills}</div>'
        f'<div class="body" style="margin-top:.5rem;"><b>Missing:</b><br>{missing_pills}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"🔍 View recipe — {metadata.get('recipe_name', 'Untitled recipe')}"):
        st.markdown(
            f"**Cuisine:** {metadata.get('cuisine', '?')} &nbsp;|&nbsp; **Diet:** {metadata.get('diet', '?')} &nbsp;|&nbsp; "
            f"**Cooking time:** {metadata.get('cooking_time', '?')} min &nbsp;|&nbsp; "
            f"**Difficulty:** {metadata.get('difficulty', '?')}"
        )

        st.markdown("**Full ingredient list**")
        st.write(", ".join(ingredients) if ingredients else "Not available")

        st.markdown("**Instructions**")
        st.write(metadata.get("instructions", "Not available"))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ You already have**")
            st.write(", ".join(matched) if matched else "Nothing confirmed yet")
        with col2:
            st.markdown("**❌ Missing**")
            st.write(", ".join(missing) if missing else "Nothing — you have it all!")

        used_substitution_ids = set()
        if missing:
            st.markdown("**🔄 Substitutes / make it yourself**")
            for ingredient in missing:
                sub_result = _match_substitution(ingredient, substitution_results)
                if sub_result is None:
                    continue
                sub_meta = sub_result["metadata"]
                used_substitution_ids.add(sub_meta.get("substitution_id"))
                st.markdown(
                    f"- **Missing: {ingredient}** → **Alternative:** {sub_meta.get('alternative', '?')}  \n"
                    f"  **Method:** {sub_meta.get('method', '?')}"
                    + (f"  \n  *Notes: {sub_meta['notes']}*" if sub_meta.get("notes") else "")
                )
            if not used_substitution_ids:
                st.caption("No substitution idea is available for the missing ingredient(s) above yet.")

    return used_substitution_ids


def render_substitution_card(result: dict) -> None:
    """A standalone card for a retrieved substitution that wasn't matched to
    any displayed recipe's missing ingredient (e.g. a pure "what can I
    substitute for X" query) - shown so retrieved knowledge is never
    silently dropped.
    """
    metadata = result.get("metadata", {})
    pct = max(0, round(result.get("score", 0.0) * 100))

    st.markdown(
        f'<div class="card">'
        f'<h4>🔄 {metadata.get("missing_ingredient", "?")} → {metadata.get("alternative", "?")}'
        f'<span class="match-badge">{pct}% match</span></h4>'
        f'<div class="body"><b>Method:</b> {metadata.get("method", "?")}</div>'
        + (
            f'<div class="body" style="margin-top:.4rem;"><b>Notes:</b> {metadata["notes"]}</div>'
            if metadata.get("notes")
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def render_results() -> None:
    st.markdown('<div class="section-title">Matching recipes</div>', unsafe_allow_html=True)

    if not st.session_state.get("has_searched", False):
        st.markdown(
            '<div class="placeholder">🍳 <b>What\'s in your kitchen?</b><br>'
            "Enter a few ingredients above, or try one of the example questions below, "
            "and we'll help you discover what you can cook.</div>",
            unsafe_allow_html=True,
        )
        return

    retrieval_error = st.session_state.get("retrieval_error")
    if retrieval_error:
        st.error(retrieval_error)
        return

    results = st.session_state.get("results", [])
    if not results:
        st.markdown(
            '<div class="placeholder"><b>No recipes found.</b><br>'
            "Try: removing a filter · adding another ingredient · choosing \"Any\" cuisine.</div>",
            unsafe_allow_html=True,
        )
        return

    query = st.session_state.get("ingredient_query", "")
    recipe_results = [r for r in results if r.get("document_type") == "recipe"]
    substitution_results = [r for r in results if r.get("document_type") != "recipe"]

    render_ai_answer(results)

    used_substitution_ids = set()
    for result in recipe_results:
        used_substitution_ids |= render_recipe_card(result, query, substitution_results)

    for result in substitution_results:
        sub_id = result["metadata"].get("substitution_id")
        if sub_id not in used_substitution_ids:
            render_substitution_card(result)


def render_how_it_works() -> None:
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        "This assistant looks at our real recipe collection to find what fits what you have, "
        "then puts together a personalized suggestion just for you — no guessing, no generic advice."
    )

    steps = [
        ("1", "Tell us what's in your kitchen."),
        ("2", "We search our recipe and substitution collection for the best matches."),
        ("3", "We compare the matches to your list to see what you're missing."),
        ("4", "Your Recipe Assistant puts together a personalized suggestion, including easy substitutes or how to make a missing ingredient at home."),
    ]
    for num, text in steps:
        st.markdown(
            f'<div class="step"><div class="num">{num}</div><div class="txt">{text}</div></div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    recipes, substitutions = get_data()

    render_sidebar(recipes, substitutions)
    render_hero()
    render_search()
    render_results()
    render_examples()
    render_how_it_works()


if __name__ == "__main__":
    main()
