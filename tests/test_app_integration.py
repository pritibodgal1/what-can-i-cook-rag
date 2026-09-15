"""Integration tests for app.py wired to the real RAG backend.

Uses Streamlit's `AppTest` (headless script execution + simulated widget
interaction) rather than a real server - fast, no network, and exercises
the actual render functions exactly as Streamlit would call them.

`src.retriever.retrieve` and `src.chatbot.generate_response` are mocked at
their real source location (`app.py` imports `retrieve` by name and
`chatbot` as a module, and AppTest re-executes the whole script - including
its imports - on every `.run()`, so patching the source picks up correctly
each time). No real embedding model or Ollama instance is needed.
"""

from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP_PATH = str(config.BASE_DIR / "app.py")


def _find_button(at, label):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button found with label {label!r}"
    return matches[0]


def _all_text(at) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(str(w.value) for w in at.warning) + " " + " ".join(str(e.value) for e in at.error)


def _recipe_result(recipe_id="R001", name="Paneer Bhurji", cuisine="Indian", diet="Vegetarian",
                    cooking_time=20, difficulty="Easy", ingredients=None, instructions="Cook it.", score=0.6):
    return {
        "document": f"Recipe: {name}",
        "metadata": {
            "recipe_id": recipe_id, "recipe_name": name, "cuisine": cuisine, "diet": diet,
            "cooking_time": cooking_time, "difficulty": difficulty,
            "ingredients": ingredients or ["paneer", "onion", "tomato", "capsicum"],
            "instructions": instructions,
        },
        "score": score,
        "document_type": "recipe",
    }


def _substitution_result(sub_id="S001", missing="paneer", alternative="Homemade Paneer",
                          method="Boil milk, add lemon juice.", notes="", score=0.5):
    return {
        "document": f"Missing Ingredient: {missing}",
        "metadata": {
            "substitution_id": sub_id, "missing_ingredient": missing,
            "alternative": alternative, "method": method, "notes": notes,
        },
        "score": score,
        "document_type": "substitution",
    }


def _run_search(at, query, cuisine=None, diet=None, cooking_time=None, difficulty=None):
    at.text_area(key="ingredient_query").set_value(query)
    if cuisine:
        at.selectbox(key="cuisine_filter").set_value(cuisine)
    if diet:
        at.selectbox(key="diet_filter").set_value(diet)
    if cooking_time:
        at.selectbox(key="time_filter").set_value(cooking_time)
    if difficulty:
        at.selectbox(key="difficulty_filter").set_value(difficulty)
    _find_button(at, "✨ Find Recipes").click()
    at.run()


# --- 1. successful search with mocked retrieval + mocked generation --------


def test_successful_search_shows_recipe_card_and_ai_answer():
    recipe = _recipe_result()
    with patch("src.retriever.retrieve", return_value=[recipe]) as mock_retrieve, \
         patch("src.chatbot.generate_response", return_value="Try Paneer Bhurji!") as mock_generate:
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "I have paneer, onion and tomato")

        assert at.exception == []
        mock_retrieve.assert_called_once_with("I have paneer, onion and tomato")
        assert mock_generate.called

        text = _all_text(at)
        assert "Paneer Bhurji" in text
        assert "Try Paneer Bhurji!" in text
        assert "% match" in text


def test_generate_response_receives_the_same_results_shown_on_screen():
    recipe = _recipe_result()
    with patch("src.retriever.retrieve", return_value=[recipe]), \
         patch("src.chatbot.generate_response", return_value="ok") as mock_generate:
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "I have paneer")

        call_args = mock_generate.call_args
        assert call_args.args[0] == "I have paneer"
        assert call_args.args[1] == [recipe]


# --- 2. no-results path -----------------------------------------------------


def test_no_results_shows_no_results_state_and_never_calls_generation():
    with patch("src.retriever.retrieve", return_value=[]) as mock_retrieve, \
         patch("src.chatbot.generate_response") as mock_generate:
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "xyz qwerty nonexistent")

        assert at.exception == []
        assert mock_retrieve.called
        mock_generate.assert_not_called()
        assert "No recipes found" in _all_text(at)


# --- 3. filter application ---------------------------------------------------


def test_cuisine_filter_excludes_non_matching_recipes():
    indian = _recipe_result(recipe_id="R001", name="Palak Paneer", cuisine="Indian")
    italian = _recipe_result(recipe_id="R002", name="Pasta Arrabbiata", cuisine="Italian")
    with patch("src.retriever.retrieve", return_value=[indian, italian]), \
         patch("src.chatbot.generate_response", return_value="ok") as mock_generate:
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "something with tomato", cuisine="Italian")

        text = _all_text(at)
        assert "Pasta Arrabbiata" in text
        assert "Palak Paneer" not in text
        # only the filtered-in recipe reaches the local LLM
        assert mock_generate.call_args.args[1] == [italian]


def test_difficulty_and_time_filters_apply_together():
    easy_quick = _recipe_result(recipe_id="R001", name="Quick Easy Dish", difficulty="Easy", cooking_time=10)
    hard_slow = _recipe_result(recipe_id="R002", name="Hard Slow Dish", difficulty="Medium", cooking_time=90)
    with patch("src.retriever.retrieve", return_value=[easy_quick, hard_slow]), \
         patch("src.chatbot.generate_response", return_value="ok"):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "something", cooking_time="Under 15 min", difficulty="Easy")

        text = _all_text(at)
        assert "Quick Easy Dish" in text
        assert "Hard Slow Dish" not in text


def test_substitution_results_pass_through_filters_unaffected():
    """Filters are recipe-specific fields; a substitution result has none of
    them, so it should never be excluded by cuisine/diet/time/difficulty.
    """
    sub = _substitution_result()
    with patch("src.retriever.retrieve", return_value=[sub]), \
         patch("src.chatbot.generate_response", return_value="ok") as mock_generate:
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "what can I substitute for paneer?", cuisine="Italian", diet="Vegan")

        assert mock_generate.call_args.args[1] == [sub]
        assert "No recipes found" not in _all_text(at)


# --- 4. Ollama error path ----------------------------------------------------


def test_ollama_unavailable_message_is_displayed():
    from src.chatbot import _config_error_message

    recipe = _recipe_result()
    error_message = _config_error_message()
    with patch("src.retriever.retrieve", return_value=[recipe]), \
         patch("src.chatbot.generate_response", return_value=error_message):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "I have paneer")

        assert at.exception == []
        assert "Ollama" in _all_text(at)


# --- 5. embedding/retrieval error path ---------------------------------------


def test_embedding_model_failure_shows_friendly_error_not_a_crash():
    with patch("src.retriever.retrieve", side_effect=OSError("DLL load failed")) as mock_retrieve, \
         patch("src.chatbot.generate_response") as mock_generate:
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "I have paneer")

        assert at.exception == []  # must not crash the app
        assert mock_retrieve.called
        mock_generate.assert_not_called()  # never reached if retrieval itself failed
        text = _all_text(at)
        # Friendly, non-technical wording (see app.py's RETRIEVAL_ERROR_MESSAGE) -
        # not "embedding model"/"semantic search"/any other implementation detail.
        assert "trouble preparing your recipe recommendations" in text.lower()
        assert "DLL load failed" not in text  # raw exception details not leaked to the user


# --- 6. example query --------------------------------------------------------


def test_example_button_populates_query_and_runs_search():
    recipe = _recipe_result(name="Vegetable Sandwich", ingredients=["potato", "onion"])
    with patch("src.retriever.retrieve", return_value=[recipe]) as mock_retrieve, \
         patch("src.chatbot.generate_response", return_value="ok"):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _find_button(at, "I have potatoes and onions").click()
        at.run()

        assert at.exception == []
        mock_retrieve.assert_called_once_with("I have potatoes and onions")
        assert at.text_area(key="ingredient_query").value == "I have potatoes and onions"
        assert "Vegetable Sandwich" in _all_text(at)


def test_example_with_time_filter_applies_it():
    quick = _recipe_result(name="Quick Dish", cooking_time=20)
    slow = _recipe_result(recipe_id="R002", name="Slow Dish", cooking_time=60)
    with patch("src.retriever.retrieve", return_value=[quick, slow]), \
         patch("src.chatbot.generate_response", return_value="ok"):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _find_button(at, "Give me something quick under 20 minutes").click()
        at.run()

        assert at.selectbox(key="time_filter").value == "Under 30 min"
        text = _all_text(at)
        assert "Quick Dish" in text
        assert "Slow Dish" not in text


# --- 7. recipe detail selection (View Recipe) -------------------------------


def test_view_recipe_shows_actual_retrieved_metadata_not_fabricated():
    recipe = _recipe_result(
        name="Paneer Bhurji", ingredients=["paneer", "onion", "capsicum"],
        instructions="Saute onion, add paneer, cook 5 minutes.", cooking_time=25, difficulty="Medium",
    )
    with patch("src.retriever.retrieve", return_value=[recipe]), \
         patch("src.chatbot.generate_response", return_value="ok"):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "paneer, onion")

        expander = at.expander[0]
        expander_text = " ".join(m.value for m in expander.markdown)
        assert "Saute onion, add paneer, cook 5 minutes." in expander_text
        assert "paneer, onion, capsicum" in expander_text
        assert "25 min" in expander_text
        assert "Medium" in expander_text


def test_missing_ingredient_substitution_shown_from_retrieved_data_only():
    """The recipe is missing "cream"; a retrieved substitution for "cream"
    should show up inside its expander - sourced only from what was
    actually retrieved, not a hard-coded lookup.
    """
    recipe = _recipe_result(ingredients=["paneer", "onion", "cream"])
    sub = _substitution_result(missing="cream", alternative="Milk + butter", method="Whisk together.")
    with patch("src.retriever.retrieve", return_value=[recipe, sub]), \
         patch("src.chatbot.generate_response", return_value="ok"):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "I have paneer and onion")  # cream is NOT mentioned -> missing

        expander_text = " ".join(m.value for m in at.expander[0].markdown)
        assert "cream" in expander_text.lower()
        assert "Milk + butter" in expander_text
        assert "Whisk together." in expander_text


def test_unmatched_substitution_shown_as_standalone_card():
    """A substitution with no corresponding missing ingredient in any
    displayed recipe (e.g. a pure "what can I substitute" query) should
    still be shown, not silently dropped.
    """
    sub = _substitution_result(missing="butter", alternative="Ghee", method="Use 3/4 the quantity.")
    with patch("src.retriever.retrieve", return_value=[sub]), \
         patch("src.chatbot.generate_response", return_value="ok"):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "what can I substitute for butter?")

        text = _all_text(at)
        assert "Ghee" in text
        assert "Use 3/4 the quantity." in text


# --- extra: empty query / clear filters -------------------------------------


def test_empty_query_shows_warning_and_never_calls_retrieve():
    with patch("src.retriever.retrieve") as mock_retrieve:
        at = AppTest.from_file(APP_PATH)
        at.run()
        _find_button(at, "✨ Find Recipes").click()
        at.run()

        assert at.exception == []
        mock_retrieve.assert_not_called()
        assert "Tell me what is in your kitchen" in _all_text(at)


def test_clear_filters_resets_everything():
    recipe = _recipe_result()
    with patch("src.retriever.retrieve", return_value=[recipe]), \
         patch("src.chatbot.generate_response", return_value="some answer"):
        at = AppTest.from_file(APP_PATH)
        at.run()
        _run_search(at, "I have paneer", cuisine="Indian")
        assert "Paneer Bhurji" in _all_text(at)

        _find_button(at, "Clear filters").click()
        at.run()

        assert at.text_area(key="ingredient_query").value == ""
        assert at.selectbox(key="cuisine_filter").value == "Any"
        assert "What's in your kitchen" in _all_text(at) or "what's in your kitchen" in _all_text(at).lower()
        assert "Paneer Bhurji" not in _all_text(at)


def test_no_stale_claude_wording_remains_in_ui():
    at = AppTest.from_file(APP_PATH)
    at.run()
    text = _all_text(at)
    assert "Claude" not in text
