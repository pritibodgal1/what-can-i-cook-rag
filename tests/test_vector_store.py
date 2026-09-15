"""Tests for the FAISS-backed vector store.

`VectorStore` itself never touches the embedding model - it just stores and
searches vectors it's handed - so every test here uses small, hand-built
numpy arrays instead of real sentence-transformers output. That keeps the
whole file fast and independent of torch being installed/working. The two
tests for the `build_index`/`search` convenience wrappers (which *do* call
into `src.embeddings`) mock that module out rather than loading the real
model, per the "avoid tests that require downloading the model" guidance.
"""

import numpy as np
import pytest

from src import config
from src.document_loader import load_recipes, load_substitutions
from src.vector_store import VectorStore, build_index, build_knowledge_base_index, load_index, save_index, search

DOCS = [
    {"id": "r1", "text": "paneer curry", "metadata": {"name": "Paneer Curry"}},
    {"id": "r2", "text": "chicken curry", "metadata": {"name": "Chicken Curry"}},
    {"id": "r3", "text": "fruit salad", "metadata": {"name": "Fruit Salad"}},
]

# Hand-built, already-normalized 3D "embeddings": r1 and r2 are close to
# each other (both curries), r3 points in an unrelated direction.
EMBEDDINGS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.9, 0.436, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype="float32",
)
EMBEDDINGS = EMBEDDINGS / np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)


def _populated_store() -> VectorStore:
    store = VectorStore()
    store.add(EMBEDDINGS, DOCS)
    return store


def test_add_documents_increases_length():
    store = VectorStore()
    assert len(store) == 0
    store.add(EMBEDDINGS, DOCS)
    assert len(store) == 3


def test_add_mismatched_lengths_raises():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.add(EMBEDDINGS[:2], DOCS)  # 2 embeddings, 3 documents


def test_add_wrong_dimension_raises():
    store = _populated_store()
    with pytest.raises(ValueError):
        store.add(np.zeros((1, 2), dtype="float32"), [{"id": "x", "text": "t", "metadata": {}}])


def test_similarity_search_ranks_by_closeness():
    store = _populated_store()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=3)
    assert [r["id"] for r in results] == ["r1", "r2", "r3"]
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]


def test_search_top_k_is_configurable():
    store = _populated_store()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    assert len(store.search(query, top_k=1)) == 1
    assert len(store.search(query, top_k=2)) == 2


def test_search_top_k_larger_than_store_is_clamped():
    store = _populated_store()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=100)
    assert len(results) == 3  # not an error, just capped to what exists


def test_search_result_carries_original_text_and_metadata():
    store = _populated_store()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    top = store.search(query, top_k=1)[0]
    assert top["text"] == "paneer curry"
    assert top["metadata"] == {"name": "Paneer Curry"}


def test_empty_store_search_returns_empty_list_not_error():
    store = VectorStore()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    assert store.search(query, top_k=5) == []


def test_adding_zero_documents_is_a_safe_no_op():
    store = VectorStore()
    store.add(np.empty((0, 3), dtype="float32"), [])
    assert len(store) == 0
    assert store.index is None


def test_save_and_load_roundtrip(tmp_path):
    store = _populated_store()
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)
    assert loaded.ids == store.ids
    assert loaded.texts == store.texts
    assert loaded.metadata == store.metadata

    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    assert [r["id"] for r in loaded.search(query, top_k=3)] == [r["id"] for r in store.search(query, top_k=3)]


def test_load_missing_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "does_not_exist")


def test_load_index_convenience_returns_none_when_absent(tmp_path):
    assert load_index(tmp_path / "nothing_here") is None


def test_save_and_load_index_convenience_functions(tmp_path):
    store = _populated_store()
    save_index(store, tmp_path)
    loaded = load_index(tmp_path)
    assert loaded is not None
    assert len(loaded) == len(store)


def test_build_index_embeds_and_adds_documents(monkeypatch):
    """Mocks the embedding model - no real sentence-transformers/torch needed."""

    def fake_embed_texts(texts, model_name=None):
        # Deterministic fake: encode each text as (len, first-char-ord, 0)
        return np.array([[len(t), ord(t[0]), 0.0] for t in texts], dtype="float32")

    monkeypatch.setattr("src.vector_store.embed_texts", fake_embed_texts)

    store = build_index(DOCS)
    assert len(store) == 3


def test_search_convenience_function_embeds_query(monkeypatch):
    def fake_embed_texts(texts, model_name=None):
        return EMBEDDINGS

    def fake_embed_query(query, model_name=None):
        return np.array([1.0, 0.0, 0.0], dtype="float32")

    monkeypatch.setattr("src.vector_store.embed_texts", fake_embed_texts)
    monkeypatch.setattr("src.vector_store.embed_query", fake_embed_query)

    store = build_index(DOCS)
    results = search(store, "anything - the fake embedder ignores this", top_k=1)
    assert results[0]["id"] == "r1"


def test_build_knowledge_base_index_end_to_end(monkeypatch, tmp_path, capsys):
    """Full chunker -> embed -> FAISS -> persist pipeline, with a fake
    embedder standing in for the real model (see module docstring) and
    persistence redirected to a temp dir instead of the real data folder.
    """

    def fake_embed_texts(texts, model_name=None):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), 8)).astype("float32")

    monkeypatch.setattr("src.vector_store.embed_texts", fake_embed_texts)
    monkeypatch.setattr(config, "VECTOR_STORE_DIR", tmp_path)

    kb_index = build_knowledge_base_index(persist=True)

    assert set(kb_index.keys()) == {"recipes", "substitutions"}
    assert len(kb_index["recipes"]) == len(load_recipes())
    assert len(kb_index["substitutions"]) == len(load_substitutions())

    # Persistence actually happened, at the (redirected) configured location.
    assert (tmp_path / "recipes" / "documents.pkl").exists()
    assert (tmp_path / "recipes" / "index.faiss").exists()
    assert (tmp_path / "substitutions" / "documents.pkl").exists()

    reloaded = load_index(tmp_path / "recipes")
    assert len(reloaded) == len(kb_index["recipes"])
