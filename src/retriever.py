"""Retrieval layer: turn a user's query into relevant context.

This file holds two retrievers side by side:

- `retrieve()` (further down) is the semantic retriever: it embeds the
  query (`embeddings.embed_query`) and searches the persisted FAISS stores
  (`vector_store.py`). This is what the live UI (`app.py`) now calls for
  its main search flow, paired with `chatbot.generate_response()` for the
  grounded answer.

- The keyword/phrase-overlap retriever above it (`search_recipes`,
  `filter_recipes`, `browse_recipes`, `find_substitutions`) is no longer
  called by the UI, but is left as-is rather than deleted (still tested).
  `match_ingredients()` is the one function from this half the UI still
  reuses directly - to compute have/missing ingredients against a
  semantically-retrieved recipe's ingredient list.
"""

import re

import pandas as pd

from src import config
from src.document_loader import parse_ingredients
from src.embeddings import embed_query
from src.vector_store import VectorStore, load_index

# Cuisine filter buckets shown in the UI map to one or more raw CSV values.
CUISINE_GROUPS = {
    "Indian": {"Indian", "South Indian"},
    "Italian": {"Italian"},
    "Asian": {"Chinese"},
}

# Cooking-time filter buckets shown in the UI map to a max-minutes cutoff.
TIME_LIMITS = {
    "Under 15 min": 15,
    "Under 30 min": 30,
    "Under 60 min": 60,
}


STOPWORDS = {
    "fresh", "powder", "seeds", "seed", "leaves", "paste", "ground", "whole",
    "dried", "chopped", "boiled", "grated", "crushed", "roasted", "red",
    "green", "black", "white", "large", "small", "and", "the", "for",
}

NEGATION_CUES = ["no", "without", "out of", "don't have", "dont have", "missing", "not have", "lacking"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _singularize(word: str) -> str:
    """Crude plural stripping so "potatoes" matches an ingredient written as
    "potato" (the CSV isn't consistent about singular vs. plural). Good
    enough for this keyword-overlap stage; not meant to handle every
    irregular plural.
    """
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _keywords(phrase: str) -> set[str]:
    words = re.findall(r"[a-z]+", phrase.lower())
    return {_singularize(w) for w in words if w not in STOPWORDS and len(w) > 2}


def _phrase_in_query(phrase: str, query_norm: str) -> bool:
    pattern = r"\b" + re.escape(_normalize(phrase)) + r"\b"
    return re.search(pattern, query_norm) is not None


def _ingredient_in_query(ingredient: str, query_norm: str) -> bool:
    """True when the ingredient appears in the query, either as the exact
    phrase or by overlap on its meaningful word(s) (e.g. "cream" matches a
    recipe's "fresh cream" since "fresh" is a generic descriptor).
    """
    if _phrase_in_query(ingredient, query_norm):
        return True
    ing_words = _keywords(ingredient)
    if not ing_words:
        return False
    query_words = {_singularize(w) for w in re.findall(r"[a-z]+", query_norm)}
    return bool(ing_words & query_words)


def _is_negated(ingredient: str, query_norm: str) -> bool:
    """True when the query explicitly says this ingredient is missing
    (e.g. "no paneer", "without cream", "don't have curd").
    """
    ing_pattern = re.escape(_normalize(ingredient))
    for cue in NEGATION_CUES:
        pattern = r"\b" + re.escape(cue) + r"\s+(?:any\s+|the\s+|a\s+)?" + ing_pattern + r"\b"
        if re.search(pattern, query_norm):
            return True
    return False


def match_ingredients(recipe_ingredients: list[str], query: str) -> tuple[list[str], list[str]]:
    """Split a recipe's ingredients into ones the user has and ones they don't.

    Matching is based on whether the ingredient name appears in the query text,
    so both a comma-separated list and a full sentence work as input. An
    ingredient explicitly negated in the query ("no paneer", "without cream")
    is always treated as missing, even though the word itself is present.
    """
    query_norm = _normalize(query)
    have, missing = [], []
    for ingredient in recipe_ingredients:
        if _is_negated(ingredient, query_norm):
            missing.append(ingredient)
        elif _ingredient_in_query(ingredient, query_norm):
            have.append(ingredient)
        else:
            missing.append(ingredient)
    return have, missing


def filter_recipes(
    recipes: pd.DataFrame,
    cuisine: str = "Any",
    diet: str = "Any",
    cooking_time: str = "Any",
    difficulty: str = "Any",
) -> pd.DataFrame:
    """Apply the cuisine / diet / cooking time / difficulty filters to the recipe table."""
    filtered = recipes

    if cuisine != "Any":
        allowed = CUISINE_GROUPS.get(cuisine, {cuisine})
        filtered = filtered[filtered["cuisine"].isin(allowed)]

    if diet != "Any":
        filtered = filtered[filtered["diet"] == diet]

    if cooking_time != "Any":
        limit = TIME_LIMITS.get(cooking_time)
        if limit is not None:
            filtered = filtered[filtered["cooking_time_minutes"] <= limit]

    if difficulty != "Any":
        filtered = filtered[filtered["difficulty"] == difficulty]

    return filtered


def browse_recipes(
    recipes: pd.DataFrame,
    cuisine: str = "Any",
    diet: str = "Any",
    cooking_time: str = "Any",
    difficulty: str = "Any",
    top_k: int = 10,
) -> list[dict]:
    """List recipes matching a filter directly, with no ingredient query.

    Used by the sidebar's cuisine/diet quick-filter chips, where the user
    is browsing a category rather than searching by what they have. Shaped
    like `search_recipes` output (matched/missing/score/name_matched) so
    the same recipe-card rendering works for both.
    """
    candidates = filter_recipes(recipes, cuisine, diet, cooking_time, difficulty)

    results = []
    for _, row in candidates.head(top_k).iterrows():
        result = row.to_dict()
        result["matched"] = []
        result["missing"] = parse_ingredients(row["ingredients"])
        result["match_count"] = 0
        result["name_matched"] = False
        result["score"] = 0.0
        results.append(result)

    return results


def search_recipes(
    query: str,
    recipes: pd.DataFrame,
    cuisine: str = "Any",
    diet: str = "Any",
    cooking_time: str = "Any",
    difficulty: str = "Any",
    top_k: int = config.TOP_K_RECIPES,
) -> list[dict]:
    """Return the best-matching recipes for the query, ranked by ingredient overlap.

    Each result is the recipe's fields plus `matched` (ingredients the user has),
    `missing` (ingredients the user doesn't) and `score` (share of the recipe's
    ingredients the user already has). Recipes with zero overlap are excluded.
    """
    candidates = filter_recipes(recipes, cuisine, diet, cooking_time, difficulty)
    query_norm = _normalize(query)

    results = []
    for _, row in candidates.iterrows():
        ingredients = parse_ingredients(row["ingredients"])
        have, missing = match_ingredients(ingredients, query)
        name_matched = _phrase_in_query(row["recipe_name"], query_norm)

        # Include a recipe either on ingredient overlap, or because the user
        # named the dish directly (e.g. "I want to make palak paneer") even
        # if none of its ingredients were separately confirmed as on hand.
        if not have and not name_matched:
            continue

        result = row.to_dict()
        result["matched"] = have
        result["missing"] = missing
        result["match_count"] = len(have)
        result["name_matched"] = name_matched
        result["score"] = len(have) / len(ingredients) if ingredients else 0.0
        results.append(result)

    results.sort(key=lambda r: (r["name_matched"], r["match_count"], r["score"]), reverse=True)
    return results[:top_k]


def find_substitutions(
    missing_ingredients: list[str],
    substitutions: pd.DataFrame,
    top_k_per_ingredient: int = config.TOP_K_SUBSTITUTIONS,
) -> dict[str, list[dict]]:
    """Map each missing ingredient to matching rows from substitutions.csv, if any."""
    found: dict[str, list[dict]] = {}

    for ingredient in missing_ingredients:
        ing_norm = _normalize(ingredient)

        exact = substitutions[substitutions["missing_ingredient"].str.lower().str.strip() == ing_norm]
        if not exact.empty:
            found[ingredient] = exact.head(top_k_per_ingredient).to_dict("records")
            continue

        loose = substitutions[
            substitutions["missing_ingredient"]
            .str.lower()
            .str.strip()
            .apply(lambda s: s in ing_norm or ing_norm in s)
        ]
        if not loose.empty:
            found[ingredient] = loose.head(top_k_per_ingredient).to_dict("records")

    return found


# ---------------------------------------------------------------------------
# Semantic retrieval (Step 4) - embeddings + FAISS, not wired into the UI yet.
# ---------------------------------------------------------------------------


def _to_semantic_result(hit: dict, document_type: str) -> dict:
    """Reshape a VectorStore.search() hit into the retriever's public shape.

    Metadata already carries everything chunker.py attached (recipe_id,
    recipe_name, cuisine, diet, cooking_time, difficulty, ingredients, ... or
    substitution_id, missing_ingredient, alternative, method, notes) - it is
    passed through as-is rather than picked apart and re-duplicated here.
    """
    return {
        "document": hit["text"],
        "metadata": hit["metadata"],
        "score": hit["score"],
        "document_type": document_type,
    }


def retrieve(
    query: str,
    top_k: int = config.VECTOR_SEARCH_TOP_K,
    relevance_threshold: float = config.RELEVANCE_THRESHOLD,
    recipe_store: VectorStore | None = None,
    substitution_store: VectorStore | None = None,
) -> list[dict]:
    """Semantic retrieval over the recipe + substitution knowledge base.

    Embeds `query` with `embeddings.embed_query`, searches both the recipe
    and substitution FAISS stores (loaded fresh from `config.VECTOR_STORE_DIR`
    unless a store is passed in explicitly - handy for tests and for
    swapping in a specific store), merges the candidates, drops anything
    below `relevance_threshold`, and returns the best `top_k` overall -
    each result tagged with which knowledge type it came from.

    Score meaning: both stores are `faiss.IndexFlatIP` over L2-normalized
    embeddings (see vector_store.py / embeddings.py), so `score` is a
    **cosine similarity in [-1, 1]** - not an L2 distance, not a percentage.
    Higher means more relevant. See `config.RELEVANCE_THRESHOLD` for what
    counts as relevant and why 0.2 is a starting default rather than a
    calibrated one.

    Returns `[]` safely (never raises) when a store is empty or hasn't been
    built yet, or when nothing clears the relevance threshold - this
    retriever only returns knowledge, it never fabricates a fallback answer.

    This function does not generate an answer - it only returns retrieved
    knowledge for a later step (the local LLM in `chatbot.py`) to reason over.
    """
    if recipe_store is None:
        recipe_store = load_index(config.VECTOR_STORE_DIR / "recipes") or VectorStore()
    if substitution_store is None:
        substitution_store = load_index(config.VECTOR_STORE_DIR / "substitutions") or VectorStore()

    query_embedding = embed_query(query)

    candidates = [_to_semantic_result(hit, "recipe") for hit in recipe_store.search(query_embedding, top_k=top_k)]
    candidates += [
        _to_semantic_result(hit, "substitution")
        for hit in substitution_store.search(query_embedding, top_k=top_k)
    ]

    candidates.sort(key=lambda r: r["score"], reverse=True)
    relevant = [r for r in candidates if r["score"] >= relevance_threshold]
    return relevant[:top_k]


def display_name(result: dict) -> str:
    """Best-effort human-readable label for a retrieved result.

    Used by `format_debug_output` below, and reused by the UI (`app.py`)
    for its source-attribution list, so this stays the one place that
    knows how to label a retrieved recipe vs. substitution.
    """
    metadata = result["metadata"]
    if result["document_type"] == "recipe":
        return metadata.get("recipe_name", "?")
    return f"{metadata.get('missing_ingredient', '?')} -> {metadata.get('alternative', '?')}"


def format_debug_output(query: str, results: list[dict]) -> str:
    """Human-readable listing of retrieved documents - a dev/debug helper
    only (see item 7 of the Step 4 spec); not exposed in the UI.
    """
    lines = [f"Query: {query}", "", "Retrieved documents:", ""]
    if not results:
        lines.append("(none above the relevance threshold)")
    for i, result in enumerate(results, start=1):
        lines.append(f"{i}. {display_name(result)}")
        lines.append(f"   Type: {result['document_type']}")
        lines.append(f"   Score: {result['score']:.3f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_debug_retrieval(query: str, **kwargs) -> list[dict]:
    """Run `retrieve()` and print a human-readable summary of the results.

    A manual dev/debug entry point, not called from the UI. Also returns
    the results, so it doubles as a quick way to inspect them in a REPL.
    """
    results = retrieve(query, **kwargs)
    print(format_debug_output(query, results))
    return results
