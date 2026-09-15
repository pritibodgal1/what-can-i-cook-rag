"""Local embeddings via Sentence Transformers.

Uses `config.EMBEDDING_MODEL` (overridable with the `EMBEDDING_MODEL` env
var), loaded once per process and reused - `get_model()` is cached so
repeated calls (and every `embed_texts`/`embed_query` call) don't reload
the model from disk. Vectors are L2-normalized so cosine similarity can be
computed as a plain inner product, which is what `vector_store.py`'s FAISS
index expects.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from src import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> "SentenceTransformer":
    """The actual cached load - always called with a concrete model name
    (never a default), so `lru_cache`'s key is stable regardless of how the
    public `get_model()` below was called. `lru_cache` keys on the literal
    arguments it's given, not on a resolved default, so caching this directly
    on a function with a default parameter would let `get_model()` and
    `get_model(config.EMBEDDING_MODEL)` - equivalent by value - miss each
    other's cache entry and load the model twice.

    The import is deliberately local to this function rather than at module
    level: sentence-transformers pulls in torch, which is a heavy, sometimes
    environment-fragile dependency (e.g. missing system DLLs). Keeping it
    lazy means importing `src.embeddings` - and anything that imports it,
    like `src.vector_store` - never requires torch to be installed/working
    unless an embedding is actually requested.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_model(model_name: str = config.EMBEDDING_MODEL) -> "SentenceTransformer":
    """Load and cache the local sentence-transformers model.

    Cached by model name, so swapping `EMBEDDING_MODEL` mid-process (e.g. in
    tests) loads a distinct cached instance rather than reusing a stale one.
    """
    return _load_model(model_name)


def embed_texts(texts: list[str], model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    """Embed a batch of documents into dense, L2-normalized vectors.

    Returns a float32 array of shape (len(texts), embedding_dim), ready to
    hand to FAISS (`vector_store.VectorStore.add`) without any further
    conversion.
    """
    if not texts:
        dim = get_model(model_name).get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype="float32")

    model = get_model(model_name)
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def embed_query(query: str, model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    """Embed a single query string into one dense, L2-normalized vector.

    Returns a float32 array of shape (embedding_dim,).
    """
    return embed_texts([query], model_name=model_name)[0]
