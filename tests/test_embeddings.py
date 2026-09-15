"""Tests for the sentence-transformers embedding layer.

These are the only tests in the suite that touch the real model (loading
sentence-transformers pulls in torch), so the whole file is skipped up
front if that import doesn't work in this environment - e.g. missing
system DLLs - rather than failing every other, unrelated test collected
alongside it. Once loaded, the model is cached (see `get_model`), so this
file only pays the load cost once no matter how many tests run.
"""

import numpy as np
import pytest

torch_ok = True
try:
    import sentence_transformers  # noqa: F401
except OSError:
    torch_ok = False

pytestmark = pytest.mark.skipif(
    not torch_ok,
    reason="sentence-transformers/torch failed to import in this environment "
           "(commonly a missing Microsoft Visual C++ Redistributable on Windows)",
)

from src import config
from src.embeddings import embed_query, embed_texts, get_model


def test_get_model_is_cached():
    model_a = get_model()
    model_b = get_model()
    assert model_a is model_b


def test_embed_texts_shape_and_dtype():
    vectors = embed_texts(["paneer butter masala", "chicken curry", "fruit salad"])
    dim = get_model().get_sentence_embedding_dimension()
    assert vectors.shape == (3, dim)
    assert vectors.dtype == np.float32


def test_embed_texts_empty_list_returns_empty_array_with_correct_dim():
    dim = get_model().get_sentence_embedding_dimension()
    vectors = embed_texts([])
    assert vectors.shape == (0, dim)


def test_embed_query_returns_single_1d_vector():
    dim = get_model().get_sentence_embedding_dimension()
    vector = embed_query("paneer, onion, tomato")
    assert vector.shape == (dim,)
    assert vector.dtype == np.float32


def test_embeddings_are_l2_normalized():
    vectors = embed_texts(["a quick paneer curry", "something completely different"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_texts_score_higher_than_dissimilar_ones():
    """Sanity check that the model captures real semantic similarity."""
    query = embed_query("paneer curry with tomatoes")
    close = embed_query("paneer butter masala with tomato gravy")
    far = embed_query("chocolate cake recipe")
    assert float(query @ close) > float(query @ far)


def test_uses_configured_model_name():
    assert config.EMBEDDING_MODEL  # non-empty, comes from config/env
    model = get_model(config.EMBEDDING_MODEL)
    assert model is get_model()
