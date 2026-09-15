"""Answer generation with a local LLM (via Ollama) - the last stage of the RAG flow:

    User query -> retriever.retrieve() -> retrieved documents
        -> build_context() -> local LLM (Ollama) -> grounded answer

100% free and local: no paid API, no API key, nothing leaves your machine.
Requires Ollama (https://ollama.com) installed and running, with the
configured model pulled once - see config.OLLAMA_MODEL.

This module never searches the vector store itself; it only turns
documents that `retriever.retrieve()` already found into a grounded
prompt and asks the local model to answer from them. Not wired into the
UI yet.

`generate_response()` always returns a plain, user-facing string and
never raises - the no-results, not-available, and generation-error cases
each return a friendly message instead of a model answer, so a caller can
always just display whatever comes back.
"""

from functools import lru_cache

from src import config

SYSTEM_PROMPT = """You are the recipe assistant for "What Can I Cook?", a cooking app.

You will be given RETRIEVED CONTEXT (recipe and/or substitution entries
pulled from a knowledge base) followed by a USER QUESTION. Follow these
rules strictly:

- Answer using ONLY the information in the retrieved context below. Do not
  invent recipes, ingredients, cooking times, difficulty levels, cuisines,
  or instructions that are not present in it.
- Only mention an ingredient substitution if that exact substitution
  appears in a SUBSTITUTION SOURCE entry in the context - never invent one.
- If the retrieved context does not contain enough information to answer
  the question, say so plainly rather than guessing.
- Prefer recipes whose ingredients best match what the user says they have.
- When a recipe needs ingredients the user hasn't mentioned having,
  mention what's missing.
- When you recommend a recipe, mention its cooking time, cuisine, and
  difficulty if that metadata is present in the context.
- Refer to each recipe or substitution by the name given in the context,
  so the user can tell which source you're drawing from.
- Do not claim an ingredient, recipe, or substitution is available if it
  does not appear in the context.
- Keep the answer practical and concise - a short recommendation, not an
  essay.
"""

NO_RESULTS_MESSAGE = (
    "I couldn't find a relevant recipe in my knowledge base for that request. "
    "Try adding some ingredients or changing your filters."
)
EMPTY_RESPONSE_MESSAGE = "The local model didn't return an answer for that request. Please try again."


def _config_error_message() -> str:
    """Built at call time (not frozen at import) so it always reflects the
    current `config.OLLAMA_MODEL`, including in tests that monkeypatch it.
    """
    return (
        "The local LLM (Ollama) isn't available right now. Make sure Ollama is "
        "installed and running (https://ollama.com), and that you've pulled the "
        f"model with: ollama pull {config.OLLAMA_MODEL}"
    )


@lru_cache(maxsize=1)
def _get_client():
    """Lazily construct and cache the Ollama client.

    Construction alone doesn't touch the network or validate anything -
    the actual connection is only attempted on `_call_ollama`'s
    `client.chat(...)` call below.
    """
    import ollama

    return ollama.Client(host=config.OLLAMA_HOST)


def build_context(retrieved_results: list[dict]) -> str:
    """Turn `retriever.retrieve()` results into plain-text context blocks.

    Only the fields the model actually needs are included - similarity
    scores, document ids and other internal book-keeping are left out.
    Reuses the metadata chunker.py already attached rather than
    re-deriving anything.
    """
    blocks = []
    for result in retrieved_results:
        metadata = result.get("metadata", {})

        if result.get("document_type") == "substitution":
            block = (
                "SUBSTITUTION SOURCE\n"
                f"Missing ingredient: {metadata.get('missing_ingredient', 'unknown')}\n"
                f"Alternative: {metadata.get('alternative', 'unknown')}\n"
                f"Method: {metadata.get('method', 'unknown')}"
            )
            if metadata.get("notes"):
                block += f"\nNotes: {metadata['notes']}"
        else:
            ingredients = metadata.get("ingredients", "unknown")
            if isinstance(ingredients, (list, tuple)):
                ingredients = ", ".join(ingredients)
            block = (
                "RECIPE SOURCE\n"
                f"Recipe: {metadata.get('recipe_name', 'unknown')}\n"
                f"Cuisine: {metadata.get('cuisine', 'unknown')}\n"
                f"Diet: {metadata.get('diet', 'unknown')}\n"
                f"Cooking time: {metadata.get('cooking_time', 'unknown')} minutes\n"
                f"Difficulty: {metadata.get('difficulty', 'unknown')}\n"
                f"Ingredients: {ingredients}\n"
                f"Instructions: {metadata.get('instructions', 'unknown')}"
            )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(query: str, context: str) -> str:
    """Combine the retrieved context and the user's question, clearly
    labeled and separated, into the single user-turn message sent to the model.
    """
    return f"RETRIEVED CONTEXT:\n\n{context}\n\n---\n\nUSER QUESTION:\n{query}"


def _call_ollama(system_prompt: str, user_content: str) -> str:
    """Make the actual Ollama chat call and return the extracted answer text.

    Raises on any failure (Ollama not running, model not pulled, network
    issue, etc.) - `generate_response` is the layer responsible for
    catching that and translating it into a safe message.
    """
    client = _get_client()
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return (response.message.content or "").strip()


def generate_response(query: str, retrieved_results: list[dict] | None) -> str:
    """The RAG generation step: retrieved documents -> grounded local-LLM answer.

    `retrieved_results` must already come from `retriever.retrieve()` -
    this function performs no retrieval of its own. Only the retrieved
    context and the user's question are sent to the model - nothing else -
    so the answer stays grounded. Returns a plain string in every case
    (never raises):

    - no retrieved_results -> NO_RESULTS_MESSAGE, Ollama is never called
    - Ollama unreachable / model not pulled / any generation error ->
      a friendly configuration message
    - model replies with no text -> EMPTY_RESPONSE_MESSAGE
    - otherwise -> the model's grounded answer
    """
    if not retrieved_results:
        return NO_RESULTS_MESSAGE

    context = build_context(retrieved_results)
    prompt = build_prompt(query, context)

    try:
        answer = _call_ollama(SYSTEM_PROMPT, prompt)
    except Exception as exc:
        # Server-side debug log only - never returned to the caller.
        print(f"[chatbot] Ollama call failed: {type(exc).__name__}: {exc}")
        return _config_error_message()

    return answer if answer else EMPTY_RESPONSE_MESSAGE


# A small, realistic stand-in for retriever.retrieve() output, used only by
# manual_test() below when the real vector store isn't available yet.
_SAMPLE_RETRIEVED_RESULTS = [
    {
        "document": (
            "Recipe: Paneer Bhurji\n\nCuisine: Indian\nDiet: Vegetarian\n"
            "Cooking Time: 20 minutes\nDifficulty: Easy\n\n"
            "Ingredients:\npaneer, onion, tomato, capsicum\n\n"
            "Instructions:\nHeat oil, saute onion and tomato, add crumbled "
            "paneer and spices, and cook for 5 minutes."
        ),
        "metadata": {
            "recipe_id": "R000", "recipe_name": "Paneer Bhurji", "cuisine": "Indian",
            "diet": "Vegetarian", "cooking_time": 20, "difficulty": "Easy",
            "ingredients": ["paneer", "onion", "tomato", "capsicum"],
            "instructions": "Heat oil, saute onion and tomato, add crumbled paneer and spices, and cook for 5 minutes.",
        },
        "score": 0.87,
        "document_type": "recipe",
    },
    {
        "document": (
            "Missing Ingredient: Paneer\nAlternative: Homemade Paneer\n\n"
            "Method:\nBoil milk, add lemon juice, strain and press."
        ),
        "metadata": {
            "substitution_id": "S000", "missing_ingredient": "Paneer",
            "alternative": "Homemade Paneer", "method": "Boil milk, add lemon juice, strain and press.",
            "notes": "",
        },
        "score": 0.75,
        "document_type": "substitution",
    },
]


def manual_test(
    query: str = "I have paneer, onion and tomato. What can I cook?",
    retrieved_results: list[dict] | None = None,
) -> str:
    """Manual, non-automated check of the full generation layer.

    Not part of the pytest suite - run it directly:

        python -c "from src.chatbot import manual_test; manual_test()"

    Makes a REAL call to your local Ollama if it's installed, running, and
    has the configured model pulled (`ollama pull <OLLAMA_MODEL>`) -
    otherwise returns the friendly "not available" message, same as
    generate_response always would.

    If `retrieved_results` isn't given, it tries a real `retriever.retrieve()`
    call first; if that comes back empty, or can't even run yet because the
    real embedding model is blocked by the Windows VC++ Redistributable
    issue (see src/embeddings.py), it falls back to a small hand-written
    sample so the local-LLM generation step can still be exercised
    end-to-end without bypassing that issue.
    """
    if retrieved_results is None:
        from src.retriever import retrieve

        try:
            retrieved_results = retrieve(query)
        except Exception as exc:
            print(f"(Retrieval unavailable ({type(exc).__name__}) - using a small mocked retrieval set instead.)")
            retrieved_results = _SAMPLE_RETRIEVED_RESULTS

        if not retrieved_results:
            print("(No real vector-store results available yet - using a small mocked retrieval set instead.)")
            retrieved_results = _SAMPLE_RETRIEVED_RESULTS

    print(f"Query: {query}")
    print(f"Retrieved {len(retrieved_results)} document(s).\n")

    answer = generate_response(query, retrieved_results)
    print("--- Response ---")
    print(answer)
    return answer
