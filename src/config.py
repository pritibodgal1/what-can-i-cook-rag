"""Central configuration. Paths and settings live here, secrets never do."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RECIPES_CSV = DATA_DIR / "recipes.csv"
SUBSTITUTIONS_CSV = DATA_DIR / "substitutions.csv"

# Where the persisted FAISS indexes + document/metadata mappings live.
# Generated at build time (see src/vector_store.py) - not checked into git.
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(DATA_DIR / "vector_store")))

# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Local LLM for grounded answer generation, served by Ollama (https://ollama.com)
# - free, runs entirely on your machine, no API key. Install Ollama, then
# pull the model once with: ollama pull <OLLAMA_MODEL>
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Retrieval settings ---
TOP_K_RECIPES = 5
TOP_K_SUBSTITUTIONS = 3

# Default top-k for raw vector similarity search (src/vector_store.py). Kept
# separate from TOP_K_RECIPES/TOP_K_SUBSTITUTIONS above, which belong to the
# existing keyword-overlap retriever and are untouched by this step.
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", "5"))

# Minimum score for a semantically-retrieved document to count as relevant
# (see src/retriever.py's `retrieve()`). This is a COSINE SIMILARITY, not an
# L2 distance and not a percentage: our FAISS indexes are `IndexFlatIP` over
# L2-normalized embeddings (see vector_store.VectorStore, embeddings.py), so
# inner product == cosine similarity, range [-1, 1], higher = more similar.
# A result below this threshold is dropped rather than passed forward.
#
# 0.2 is a reasonable starting point for all-MiniLM-L6-v2-style query-vs-
# document similarity, NOT an empirically-calibrated value - the real model
# can't run on this machine yet (missing Microsoft Visual C++ Redistributable;
# see src/embeddings.py / src/retriever.py), so re-check real retrieved
# scores once that's fixed and adjust if needed.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))

# --- App ---
APP_TITLE = "What Can I Cook? 🍳"
APP_TAGLINE = "Turn the ingredients in your kitchen into your next meal."
