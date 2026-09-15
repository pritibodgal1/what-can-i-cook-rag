"""Tests for the semantic retrieval layer (Step 4).

`retrieve()` accepts `recipe_store`/`substitution_store` directly, so most
tests here build tiny hand-crafted `VectorStore`s and monkeypatch
`src.retriever.embed_query` instead of loading the real sentence-transformers
model - fast, deterministic, and independent of the Windows VC++ issue that
currently blocks torch on this machine (see tests/test_embeddings.py).

The real-model tests at the bottom are NOT removed or hidden - they're
skipped with a clear reason (same pattern as test_embeddings.py) when torch
can't load, and will run for real once the Redistributable is installed.
"""

import numpy as np
import pytest

from src import config
from src.retriever import format_debug_output, print_debug_retrieval, retrieve
from src.vector_store import VectorStore

RECIPE_DOCS = [
    {
        "id": "R001",
        "text": "Recipe: Paneer Bhurji\n\nIngredients:\npaneer, onion, tomato, capsicum",
        "metadata": {
            "recipe_id": "R001", "recipe_name": "Paneer Bhurji", "cuisine": "Indian",
            "diet": "Vegetarian", "cooking_time": 20, "difficulty": "Easy",
            "ingredients": ["paneer", "onion", "tomato", "capsicum"],
        },
    },
    {
        "id": "R002",
        "text": "Recipe: Fruit Salad\n\nIngredients:\napple, banana, grapes",
        "metadata": {
            "recipe_id": "R002", "recipe_name": "Fruit Salad", "cuisine": "Continental",
            "diet": "Vegan", "cooking_time": 10, "difficulty": "Easy",
            "ingredients": ["apple", "banana", "grapes"],
        },
    },
]

SUBSTITUTION_DOCS = [
    {
        "id": "S001",
        "text": "Missing Ingredient: Paneer\nAlternative: Homemade Paneer\n\nMethod:\nBoil milk, add lemon juice...",
        "metadata": {
            "substitution_id": "S001", "missing_ingredient": "Paneer",
            "alternative": "Homemade Paneer", "method": "Boil milk, add lemon juice, strain and press.",
            "notes": "",
        },
    },
]

# 3D hand-built embeddings: recipe 0 (Paneer Bhurji) and the substitution are
# both "close" to a paneer-flavoured query direction; recipe 1 (Fruit Salad)
# points somewhere unrelated.
RECIPE_EMBEDDINGS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
SUBSTITUTION_EMBEDDINGS = np.array([[0.95, 0.05, 0.0]], dtype="float32")
SUBSTITUTION_EMBEDDINGS /= np.linalg.norm(SUBSTITUTION_EMBEDDINGS, axis=1, keepdims=True)

PANEER_QUERY_VECTOR = np.array([1.0, 0.0, 0.0], dtype="float32")
UNRELATED_QUERY_VECTOR = np.array([0.0, 0.0, 1.0], dtype="float32")  # orthogonal to everything indexed


def _stores():
    recipe_store = VectorStore()
    recipe_store.add(RECIPE_EMBEDDINGS, RECIPE_DOCS)
    substitution_store = VectorStore()
    substitution_store.add(SUBSTITUTION_EMBEDDINGS, SUBSTITUTION_DOCS)
    return recipe_store, substitution_store


def _mock_embed_query(monkeypatch, vector):
    monkeypatch.setattr("src.retriever.embed_query", lambda query: vector)


# --- core behavior, all with mocked embeddings ------------------------------


def test_query_embedding_is_used(monkeypatch):
    """`retrieve()` calls embed_query with the query text - and uses the result."""
    recipe_store, substitution_store = _stores()
    calls = []

    def fake_embed_query(query):
        calls.append(query)
        return PANEER_QUERY_VECTOR

    monkeypatch.setattr("src.retriever.embed_query", fake_embed_query)
    retrieve("I have paneer, onion and tomato", recipe_store=recipe_store, substitution_store=substitution_store)
    assert calls == ["I have paneer, onion and tomato"]


def test_faiss_search_returns_results(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve("paneer curry", recipe_store=recipe_store, substitution_store=substitution_store)
    assert len(results) > 0


def test_metadata_is_preserved(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve("paneer curry", recipe_store=recipe_store, substitution_store=substitution_store)
    top = results[0]
    assert top["metadata"]["recipe_id"] == "R001" or top["metadata"]["substitution_id"] == "S001"
    assert "document" in top and top["document"].startswith(("Recipe:", "Missing Ingredient:"))


def test_document_type_is_preserved_for_both_kinds(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve(
        "paneer", top_k=10, relevance_threshold=-1.0,
        recipe_store=recipe_store, substitution_store=substitution_store,
    )
    types = {r["document_type"] for r in results}
    assert types == {"recipe", "substitution"}
    for r in results:
        if r["document_type"] == "recipe":
            assert "recipe_id" in r["metadata"]
        else:
            assert "substitution_id" in r["metadata"]


def test_top_k_limits_results(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve(
        "paneer", top_k=1, relevance_threshold=-1.0,
        recipe_store=recipe_store, substitution_store=substitution_store,
    )
    assert len(results) == 1
    # And it should be the single most relevant one out of all candidates.
    assert results[0]["metadata"].get("recipe_id") == "R001"


def test_default_top_k_matches_config(monkeypatch):
    import inspect

    assert inspect.signature(retrieve).parameters["top_k"].default == config.VECTOR_SEARCH_TOP_K
    assert config.VECTOR_SEARCH_TOP_K == 5


def test_results_are_sorted_by_score_descending(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve(
        "paneer", top_k=10, relevance_threshold=-1.0,
        recipe_store=recipe_store, substitution_store=substitution_store,
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_cosine_similarity_range(monkeypatch):
    """Scores must fall in [-1, 1] (cosine similarity), not e.g. [0, 100]."""
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve(
        "paneer", top_k=10, relevance_threshold=-1.0,
        recipe_store=recipe_store, substitution_store=substitution_store,
    )
    for r in results:
        assert -1.0 <= r["score"] <= 1.0 + 1e-6


def test_irrelevant_query_is_rejected_by_relevance_threshold(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, UNRELATED_QUERY_VECTOR)
    # Default threshold (0.2) should reject a near-orthogonal (~0 similarity) match.
    results = retrieve("completely unrelated gibberish", recipe_store=recipe_store, substitution_store=substitution_store)
    assert results == []


def test_relevance_threshold_is_configurable_per_call(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, UNRELATED_QUERY_VECTOR)
    # A very low threshold should let the same low-similarity match through.
    results = retrieve(
        "completely unrelated gibberish", relevance_threshold=-1.0,
        recipe_store=recipe_store, substitution_store=substitution_store,
    )
    assert len(results) > 0


def test_empty_stores_return_empty_list_safely(monkeypatch):
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve("anything", recipe_store=VectorStore(), substitution_store=VectorStore())
    assert results == []


def test_missing_persisted_index_is_handled_safely(monkeypatch, tmp_path):
    """No vector store on disk at all yet - should degrade to no results,
    never raise (this is genuinely today's state on a fresh checkout).
    """
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", tmp_path / "does_not_exist")
    results = retrieve("paneer curry")
    assert results == []


def test_persisted_index_can_be_loaded_and_searched(monkeypatch, tmp_path):
    """Build stores, persist them for real, then retrieve() with NO explicit
    store args - exercising the actual load-from-disk path, not just the
    injected-store shortcut the other tests use.
    """
    recipe_store, substitution_store = _stores()
    recipe_store.save(tmp_path / "recipes")
    substitution_store.save(tmp_path / "substitutions")

    monkeypatch.setattr(config, "VECTOR_STORE_DIR", tmp_path)
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)

    results = retrieve("paneer curry")
    assert len(results) > 0
    assert results[0]["metadata"]["recipe_id"] == "R001"


def test_format_debug_output_matches_expected_shape(monkeypatch):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = retrieve("paneer curry", recipe_store=recipe_store, substitution_store=substitution_store)
    text = format_debug_output("paneer curry", results)
    assert text.startswith("Query: paneer curry")
    assert "Retrieved documents:" in text
    assert "Type: recipe" in text or "Type: substitution" in text
    assert "Score:" in text


def test_format_debug_output_handles_no_results():
    text = format_debug_output("gibberish", [])
    assert "Query: gibberish" in text
    assert "none above the relevance threshold" in text


def test_print_debug_retrieval_prints_and_returns(monkeypatch, capsys):
    recipe_store, substitution_store = _stores()
    _mock_embed_query(monkeypatch, PANEER_QUERY_VECTOR)
    results = print_debug_retrieval("paneer curry", recipe_store=recipe_store, substitution_store=substitution_store)
    out = capsys.readouterr().out
    assert "Query: paneer curry" in out
    assert results and results[0]["document_type"] in ("recipe", "substitution")


# --- real-model tests: skipped (not removed) if torch can't load -----------

torch_ok = True
try:
    import sentence_transformers  # noqa: F401
except OSError:
    torch_ok = False


@pytest.mark.skipif(
    not torch_ok,
    reason="sentence-transformers/torch failed to import in this environment "
           "(commonly a missing Microsoft Visual C++ Redistributable on Windows)",
)
class TestRetrieveWithRealEmbeddings:
    """Same behaviors as above, but through the real embed_query - no mocking.

    Uses real recipe/substitution documents (via chunker.py) so this also
    doubles as an end-to-end smoke test of chunker -> embeddings -> FAISS ->
    retriever once the real model can load.
    """

    def _real_stores(self):
        from src.chunker import build_knowledge_base
        from src.vector_store import build_index

        kb = build_knowledge_base(verbose=False)
        return build_index(kb["recipe_documents"]), build_index(kb["substitution_documents"])

    def test_real_semantic_search_ranks_relevant_recipe_first(self):
        recipe_store, substitution_store = self._real_stores()
        results = retrieve(
            "a quick paneer dish with tomatoes and onions",
            recipe_store=recipe_store, substitution_store=substitution_store,
        )
        assert results
        assert any("paneer" in r["document"].lower() for r in results[:3])

    def test_real_substitution_retrieval_for_missing_paneer(self):
        recipe_store, substitution_store = self._real_stores()
        results = retrieve(
            "I don't have paneer, can I make it myself?",
            top_k=5, recipe_store=recipe_store, substitution_store=substitution_store,
        )
        assert any(r["document_type"] == "substitution" for r in results)

    def test_real_scores_are_valid_cosine_similarities(self):
        recipe_store, substitution_store = self._real_stores()
        results = retrieve(
            "What can I substitute for butter?",
            recipe_store=recipe_store, substitution_store=substitution_store,
        )
        for r in results:
            assert -1.0 <= r["score"] <= 1.0
