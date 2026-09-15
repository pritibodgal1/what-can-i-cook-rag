"""FAISS vector index: build, save, load, search.

`VectorStore` wraps a `faiss.IndexFlatIP` (inner product over L2-normalized
vectors == cosine similarity) alongside the original document text and
metadata for each vector, so a search result can hand back everything the
retriever/UI will eventually need without a second lookup.

The module-level `build_index` / `save_index` / `load_index` / `search`
functions are thin wrappers around that class - kept as functions (matching
the original stub names) for callers that just want the simple path.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from src import config
from src.chunker import build_knowledge_base
from src.embeddings import embed_query, embed_texts


class VectorStore:
    """An in-memory FAISS index plus the text/metadata for each vector."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int | None = None
        self.ids: list = []
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    def __len__(self) -> int:
        return len(self.texts)

    def add(self, embeddings: np.ndarray, documents: list[dict]) -> None:
        """Add a batch of documents with their pre-computed embeddings.

        `embeddings` must have one row per entry in `documents`, each a
        `{"id", "text", "metadata"}` dict (the shape `chunker.py` produces).
        The FAISS index is created lazily on the first call, sized to
        whatever embedding dimension it's given.
        """
        if len(embeddings) != len(documents):
            raise ValueError(f"Got {len(embeddings)} embeddings for {len(documents)} documents.")
        if len(documents) == 0:
            return

        embeddings = np.asarray(embeddings, dtype="float32")

        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
        elif embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"this store's existing dimension {self.dimension}."
            )

        self.index.add(embeddings)
        for doc in documents:
            self.ids.append(doc["id"])
            self.texts.append(doc["text"])
            self.metadata.append(doc["metadata"])

    def search(self, query_embedding: np.ndarray, top_k: int = config.VECTOR_SEARCH_TOP_K) -> list[dict]:
        """Return the top_k documents most similar to a query embedding.

        Each result is `{"id", "text", "metadata", "score"}`, sorted by
        descending similarity. Returns `[]` for an empty store instead of
        raising - FAISS itself has nothing to search yet.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than top_k exist
                continue
            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "score": float(score),
            })
        return results

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and the id/text/metadata mapping to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))

        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(
                {"ids": self.ids, "texts": self.texts, "metadata": self.metadata, "dimension": self.dimension},
                f,
            )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Load a previously saved store. Raises FileNotFoundError if absent."""
        directory = Path(directory)
        documents_path = directory / "documents.pkl"
        if not documents_path.exists():
            raise FileNotFoundError(f"No vector store found at {directory}")

        store = cls()
        with open(documents_path, "rb") as f:
            data = pickle.load(f)
        store.ids = data["ids"]
        store.texts = data["texts"]
        store.metadata = data["metadata"]
        store.dimension = data["dimension"]

        index_path = directory / "index.faiss"
        if index_path.exists():
            store.index = faiss.read_index(str(index_path))

        return store


def build_index(documents: list[dict]) -> VectorStore:
    """Embed a list of documents and build a fresh FAISS index over them."""
    store = VectorStore()
    if documents:
        embeddings = embed_texts([doc["text"] for doc in documents])
        store.add(embeddings, documents)
    return store


def save_index(store: VectorStore, directory: Path | None = None) -> None:
    """Persist a store to `directory` (defaults to `config.VECTOR_STORE_DIR`)."""
    store.save(directory if directory is not None else config.VECTOR_STORE_DIR)


def load_index(directory: Path | None = None) -> VectorStore | None:
    """Load a previously saved store, or `None` if nothing is persisted yet.

    Returning `None` (rather than raising) lets a caller write
    `store = load_index() or build_index(documents)` to avoid rebuilding
    embeddings on every app start when a persisted store already exists.
    """
    directory = directory if directory is not None else config.VECTOR_STORE_DIR
    try:
        return VectorStore.load(directory)
    except FileNotFoundError:
        return None


def search(store: VectorStore, query: str, top_k: int = config.VECTOR_SEARCH_TOP_K) -> list[dict]:
    """Embed `query` and return the top_k most similar documents in `store`."""
    return store.search(embed_query(query), top_k)


def build_knowledge_base_index(persist: bool = True) -> dict[str, VectorStore]:
    """Full pipeline: chunker.build_knowledge_base() -> embed -> FAISS -> persist.

    Recipes and substitutions get separate indexes (mirroring how
    chunker.py already keeps them as separate document lists), returned as
    `{"recipes": VectorStore, "substitutions": VectorStore}`. Not wired into
    the UI or retriever yet - this is the standalone build step.
    """
    kb = build_knowledge_base(verbose=True)

    recipe_store = build_index(kb["recipe_documents"])
    substitution_store = build_index(kb["substitution_documents"])

    if persist:
        save_index(recipe_store, config.VECTOR_STORE_DIR / "recipes")
        save_index(substitution_store, config.VECTOR_STORE_DIR / "substitutions")

    return {"recipes": recipe_store, "substitutions": substitution_store}
