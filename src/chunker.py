"""Turn cleaned recipe/substitution rows into RAG-ready documents.

Recipes and substitutions are already small, self-contained records, so
each row becomes exactly one document - no arbitrary character-based
chunking. Every document is `{"id", "text", "metadata"}`: `text` is what
Step 3 will embed, `metadata` is what the retriever and UI read back.
"""

import pandas as pd

from src.document_loader import (
    load_recipes_with_stats,
    load_substitutions_with_stats,
    normalize_ingredients,
    parse_ingredients,
)


def _get(row: pd.Series, key: str, default: str = "") -> str:
    """Read an optional column, treating a missing/blank value as `default`."""
    value = row.get(key, default)
    return default if pd.isna(value) else value


def recipes_to_documents(recipes: pd.DataFrame) -> list[dict]:
    """Build one RAG document per recipe, with metadata for retrieval/UI."""
    documents = []
    for _, row in recipes.iterrows():
        ingredients_list = parse_ingredients(row["ingredients"])
        meal_type = _get(row, "meal_type")

        text = (
            f"Recipe: {row['recipe_name']}\n\n"
            f"Cuisine: {row['cuisine']}\n"
            f"Diet: {row['diet']}\n"
            f"Cooking Time: {row['cooking_time_minutes']} minutes\n"
            f"Difficulty: {row['difficulty']}\n"
            + (f"Meal Type: {meal_type}\n" if meal_type else "")
            + f"\nIngredients:\n{', '.join(ingredients_list)}\n\n"
            f"Instructions:\n{row['instructions']}"
        )

        documents.append({
            "id": row["recipe_id"],
            "text": text,
            "metadata": {
                "recipe_id": row["recipe_id"],
                "recipe_name": row["recipe_name"],
                "cuisine": row["cuisine"],
                "diet": row["diet"],
                "cooking_time": int(row["cooking_time_minutes"]),
                "difficulty": row["difficulty"],
                "ingredients": normalize_ingredients(row["ingredients"]),
                "meal_type": meal_type,
                "instructions": row["instructions"],
            },
        })
    return documents


def substitutions_to_documents(substitutions: pd.DataFrame) -> list[dict]:
    """Build one RAG document per substitution / make-at-home entry."""
    documents = []
    for _, row in substitutions.iterrows():
        notes = _get(row, "notes")

        text = (
            f"Missing Ingredient: {row['missing_ingredient']}\n"
            f"Alternative: {row['alternative']}\n\n"
            f"Method:\n{row['method']}"
            + (f"\n\nNotes:\n{notes}" if notes else "")
        )

        documents.append({
            "id": row["substitution_id"],
            "text": text,
            "metadata": {
                "substitution_id": row["substitution_id"],
                "missing_ingredient": row["missing_ingredient"],
                "alternative": row["alternative"],
                "method": row["method"],
                "notes": notes,
            },
        })
    return documents


def _log_section(label: str, stats: dict, doc_count: int, item_noun: str) -> None:
    print(f"{item_noun.capitalize()}s loaded: {stats['loaded']}")
    if stats["dropped_invalid"] or stats["dropped_duplicates"]:
        print(
            f"  Dropped {stats['dropped_invalid']} invalid row(s) and "
            f"{stats['dropped_duplicates']} duplicate id(s)"
        )
    print(f"Valid {item_noun}s: {stats['valid']}")
    print(f"{label} documents created: {doc_count}")


def build_knowledge_base(verbose: bool = True) -> dict:
    """Load, validate, clean and chunk both CSVs into RAG-ready documents.

    Returns `{"recipe_documents": [...], "substitution_documents": [...]}`,
    each a list of `{"id", "text", "metadata"}` dicts - ready to be handed
    to the embedding step (Step 3) without any further transformation.
    """
    if verbose:
        print("Loading recipe knowledge base...")
    recipes, recipe_stats = load_recipes_with_stats()
    recipe_documents = recipes_to_documents(recipes)
    if verbose:
        _log_section("Recipe", recipe_stats, len(recipe_documents), "recipe")
        print()
        print("Loading substitution knowledge base...")

    substitutions, sub_stats = load_substitutions_with_stats()
    substitution_documents = substitutions_to_documents(substitutions)
    if verbose:
        _log_section("Substitution", sub_stats, len(substitution_documents), "substitution")
        print()
        print("Knowledge base preparation complete.")

    return {
        "recipe_documents": recipe_documents,
        "substitution_documents": substitution_documents,
    }
