"""Tests for the local-LLM (Ollama) generation layer.

Every test here mocks the Ollama client - Ollama does not need to be
installed or running for this suite to pass. `_get_client` is the one seam
`generate_response` uses to reach the local server, so patching it (via a
fake object with a `.chat(...)` method) lets us both inspect exactly what
was sent to the model and control what comes back.
"""

import sys
from types import SimpleNamespace

from src import chatbot, config

RECIPE_RESULT = {
    "document": "Recipe: Paneer Bhurji\n\nIngredients:\npaneer, onion, tomato, capsicum",
    "metadata": {
        "recipe_id": "R001", "recipe_name": "Paneer Bhurji", "cuisine": "Indian",
        "diet": "Vegetarian", "cooking_time": 20, "difficulty": "Easy",
        "ingredients": ["paneer", "onion", "tomato", "capsicum"],
        "instructions": "Saute onion and tomato, add crumbled paneer, cook 5 minutes.",
    },
    "score": 0.83,
    "document_type": "recipe",
}

SUBSTITUTION_RESULT = {
    "document": "Missing Ingredient: Paneer\nAlternative: Homemade Paneer",
    "metadata": {
        "substitution_id": "S001", "missing_ingredient": "Paneer",
        "alternative": "Homemade Paneer", "method": "Boil milk, add lemon juice, strain and press.",
        "notes": "Vinegar also works.",
    },
    "score": 0.71,
    "document_type": "substitution",
}


def _fake_client(reply_text="Try Paneer Bhurji - you have everything you need!"):
    """A minimal stand-in for ollama.Client, capturing call kwargs."""
    captured = {}

    class FakeClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content=reply_text))

    return FakeClient(), captured


# --- context / prompt building (no model calls involved) --------------------


def test_build_context_includes_recipe_metadata():
    context = chatbot.build_context([RECIPE_RESULT])
    assert "RECIPE SOURCE" in context
    assert "Paneer Bhurji" in context
    assert "Indian" in context and "Vegetarian" in context
    assert "20 minutes" in context
    assert "Easy" in context
    assert "paneer, onion, tomato, capsicum" in context


def test_build_context_includes_substitution_metadata():
    context = chatbot.build_context([SUBSTITUTION_RESULT])
    assert "SUBSTITUTION SOURCE" in context
    assert "Paneer" in context
    assert "Homemade Paneer" in context
    assert "Boil milk, add lemon juice" in context


def test_build_context_keeps_types_separate_and_both_present():
    context = chatbot.build_context([RECIPE_RESULT, SUBSTITUTION_RESULT])
    assert context.count("RECIPE SOURCE") == 1
    assert context.count("SUBSTITUTION SOURCE") == 1


def test_build_context_omits_internal_bookkeeping():
    context = chatbot.build_context([RECIPE_RESULT])
    assert "score" not in context.lower()
    assert "0.83" not in context
    assert "document_type" not in context.lower()


def test_build_prompt_clearly_separates_context_and_query():
    prompt = chatbot.build_prompt("I have paneer", "SOME CONTEXT HERE")
    assert "RETRIEVED CONTEXT" in prompt
    assert "USER QUESTION" in prompt
    assert "SOME CONTEXT HERE" in prompt
    assert "I have paneer" in prompt
    assert prompt.index("SOME CONTEXT HERE") < prompt.index("I have paneer")


# --- generate_response, mocked client ---------------------------------------


def test_ollama_receives_query_and_context(monkeypatch):
    fake, captured = _fake_client()
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    chatbot.generate_response("I have paneer, onion and tomato", [RECIPE_RESULT])

    user_message = captured["messages"][1]
    assert user_message["role"] == "user"
    assert "I have paneer, onion and tomato" in user_message["content"]
    assert "Paneer Bhurji" in user_message["content"]  # retrieved recipe context reached the model


def test_grounding_instructions_are_included_as_system_message(monkeypatch):
    fake, captured = _fake_client()
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    chatbot.generate_response("I have paneer", [RECIPE_RESULT])

    system_message = captured["messages"][0]
    assert system_message["role"] == "system"
    assert system_message["content"] == chatbot.SYSTEM_PROMPT
    assert "only" in system_message["content"].lower()
    assert "invent" in system_message["content"].lower()


def test_recipe_metadata_passed_correctly(monkeypatch):
    fake, captured = _fake_client()
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    chatbot.generate_response("what can I make with paneer?", [RECIPE_RESULT])

    sent = captured["messages"][1]["content"]
    for expected in ("Paneer Bhurji", "Indian", "Vegetarian", "20 minutes", "Easy"):
        assert expected in sent


def test_substitution_context_passed_correctly(monkeypatch):
    fake, captured = _fake_client()
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    chatbot.generate_response("what can I use instead of paneer?", [SUBSTITUTION_RESULT])

    sent = captured["messages"][1]["content"]
    assert "Homemade Paneer" in sent
    assert "Boil milk, add lemon juice" in sent


def test_uses_configured_model(monkeypatch):
    fake, captured = _fake_client()
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    chatbot.generate_response("I have paneer", [RECIPE_RESULT])

    assert captured["model"] == config.OLLAMA_MODEL


def test_local_llm_response_is_returned_correctly(monkeypatch):
    fake, _ = _fake_client(reply_text="Make Paneer Bhurji - a 20 minute Easy Indian dish.")
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    result = chatbot.generate_response("I have paneer", [RECIPE_RESULT])
    assert result == "Make Paneer Bhurji - a 20 minute Easy Indian dish."


def test_empty_response_content_returns_friendly_message(monkeypatch):
    fake = SimpleNamespace(chat=lambda **kw: SimpleNamespace(message=SimpleNamespace(content="")))
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    result = chatbot.generate_response("I have paneer", [RECIPE_RESULT])
    assert result == chatbot.EMPTY_RESPONSE_MESSAGE


def test_none_response_content_returns_friendly_message(monkeypatch):
    """The SDK's Message.content is Optional[str] - a None is possible, not just ''."""
    fake = SimpleNamespace(chat=lambda **kw: SimpleNamespace(message=SimpleNamespace(content=None)))
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    result = chatbot.generate_response("I have paneer", [RECIPE_RESULT])
    assert result == chatbot.EMPTY_RESPONSE_MESSAGE


# --- no-results / Ollama unavailable / errors --------------------------------


def test_no_results_does_not_call_ollama(monkeypatch):
    called = []
    monkeypatch.setattr(chatbot, "_get_client", lambda: called.append(True) or _fake_client()[0])

    result = chatbot.generate_response("anything", [])
    assert result == chatbot.NO_RESULTS_MESSAGE
    assert called == []  # _get_client was never even reached


def test_none_results_also_short_circuits(monkeypatch):
    assert chatbot.generate_response("anything", None) == chatbot.NO_RESULTS_MESSAGE


def test_ollama_not_running_is_handled_safely(monkeypatch):
    """Simulates the real failure mode: Ollama not installed/running raises
    a connection error (in real life, an httpx exception) from client.chat().
    """
    def raise_connection_error(**kwargs):
        raise ConnectionError("[Errno 111] Connection refused")

    fake = SimpleNamespace(chat=raise_connection_error)
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    result = chatbot.generate_response("I have paneer", [RECIPE_RESULT])
    assert result == chatbot._config_error_message()
    assert "Ollama" in result
    assert config.OLLAMA_MODEL in result  # tells the user which model to pull


def test_model_not_pulled_error_is_handled_safely(monkeypatch):
    """Simulates Ollama running but the configured model not yet pulled."""
    def raise_model_error(**kwargs):
        raise RuntimeError(f'model "{config.OLLAMA_MODEL}" not found, try pulling it first')

    fake = SimpleNamespace(chat=raise_model_error)
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    result = chatbot.generate_response("I have paneer", [RECIPE_RESULT])
    assert result == chatbot._config_error_message()


def test_generation_error_does_not_leak_internal_details(monkeypatch):
    def raise_error(**kwargs):
        raise RuntimeError("connection reset by peer at 10.0.0.5, internal trace xyz")

    fake = SimpleNamespace(chat=raise_error)
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    result = chatbot.generate_response("I have paneer", [RECIPE_RESULT])
    assert "10.0.0.5" not in result
    assert "internal trace" not in result


def test_generation_error_never_raises_out_of_generate_response(monkeypatch):
    class BoomError(Exception):
        pass

    def raise_error(**kwargs):
        raise BoomError("simulated failure")

    fake = SimpleNamespace(chat=raise_error)
    monkeypatch.setattr(chatbot, "_get_client", lambda: fake)

    # Must not raise - generate_response is documented to always return a string.
    result = chatbot.generate_response("I have paneer", [RECIPE_RESULT])
    assert isinstance(result, str)
    assert result == chatbot._config_error_message()


# --- client construction (real _get_client, cache cleared around the test) --


def test_get_client_constructs_with_configured_host(monkeypatch):
    captured = {}

    class FakeOllamaClient:
        def __init__(self, host=None, **kwargs):
            captured["host"] = host

    fake_ollama_module = SimpleNamespace(Client=FakeOllamaClient)
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama_module)
    chatbot._get_client.cache_clear()
    try:
        client = chatbot._get_client()
        assert isinstance(client, FakeOllamaClient)
        assert captured["host"] == config.OLLAMA_HOST
    finally:
        chatbot._get_client.cache_clear()


def test_get_client_is_cached(monkeypatch):
    fake_ollama_module = SimpleNamespace(Client=lambda **kw: object())
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama_module)
    chatbot._get_client.cache_clear()
    try:
        assert chatbot._get_client() is chatbot._get_client()
    finally:
        chatbot._get_client.cache_clear()


# --- migration-specific: Anthropic is fully gone -----------------------------


def test_no_anthropic_references_remain_in_source():
    import re

    pattern = re.compile(r"\banthropic\b", re.IGNORECASE)
    for py_file in (config.BASE_DIR / "src").glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert not pattern.search(content), f"Anthropic reference still present in {py_file}"


def test_requirements_no_longer_lists_anthropic_but_lists_ollama():
    content = (config.BASE_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "anthropic" not in content
    assert "ollama" in content


def test_env_example_has_no_anthropic_reference_and_shows_ollama_model():
    content = (config.BASE_DIR / ".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC" not in content.upper()
    assert "OLLAMA_MODEL=" in content


def test_env_is_gitignored():
    gitignore = (config.BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.splitlines()
