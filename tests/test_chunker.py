"""Tests for turning cleaned rows into RAG-ready documents."""

from src.chunker import build_knowledge_base, recipes_to_documents, substitutions_to_documents
from src.document_loader import load_recipes, load_substitutions


def test_recipe_documents_created_for_every_recipe():
    recipes = load_recipes()
    docs = recipes_to_documents(recipes)
    assert len(docs) == len(recipes)


def test_recipe_document_metadata_has_required_keys():
    recipes = load_recipes()
    doc = recipes_to_documents(recipes)[0]
    for key in ("recipe_id", "recipe_name", "cuisine", "diet", "cooking_time", "difficulty", "ingredients"):
        assert key in doc["metadata"]


def test_recipe_document_ingredients_are_normalized():
    recipes = load_recipes()
    doc = recipes_to_documents(recipes)[0]
    ingredients = doc["metadata"]["ingredients"]
    assert isinstance(ingredients, list) and len(ingredients) > 0
    assert all(i == i.lower().strip() for i in ingredients)


def test_recipe_document_text_contains_key_sections():
    recipes = load_recipes()
    doc = recipes_to_documents(recipes)[0]
    for section in ("Recipe:", "Cuisine:", "Ingredients:", "Instructions:"):
        assert section in doc["text"]


def test_substitution_documents_created_for_every_substitution():
    subs = load_substitutions()
    docs = substitutions_to_documents(subs)
    assert len(docs) == len(subs)


def test_substitution_document_has_required_metadata():
    subs = load_substitutions()
    doc = substitutions_to_documents(subs)[0]
    for key in ("substitution_id", "missing_ingredient", "alternative", "method"):
        assert key in doc["metadata"]


def test_substitution_document_text_contains_key_sections():
    subs = load_substitutions()
    doc = substitutions_to_documents(subs)[0]
    assert "Missing Ingredient:" in doc["text"]
    assert "Alternative:" in doc["text"]
    assert "Method:" in doc["text"]


def test_build_knowledge_base_returns_expected_structure(capsys):
    kb = build_knowledge_base(verbose=False)
    assert set(kb.keys()) == {"recipe_documents", "substitution_documents"}
    assert len(kb["recipe_documents"]) == len(load_recipes())
    assert len(kb["substitution_documents"]) == len(load_substitutions())

    captured = capsys.readouterr()
    assert captured.out == ""  # verbose=False must not print anything


def test_build_knowledge_base_verbose_logs_progress(capsys):
    build_knowledge_base(verbose=True)
    out = capsys.readouterr().out
    assert "Loading recipe knowledge base" in out
    assert "Recipe documents created" in out
    assert "Loading substitution knowledge base" in out
    assert "Substitution documents created" in out
    assert "Knowledge base preparation complete" in out
