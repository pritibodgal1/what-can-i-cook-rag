"""Tests for the CSV loading/validation/cleaning layer."""

import os
import tempfile
from pathlib import Path

import pandas as pd

from src.document_loader import (
    REQUIRED_RECIPE_COLUMNS,
    _load_and_clean,
    load_recipes,
    load_recipes_with_stats,
    load_substitutions,
    load_substitutions_with_stats,
    normalize_ingredient,
    normalize_ingredients,
    parse_ingredients,
)


def test_recipes_load():
    df = load_recipes()
    assert len(df) > 0


def test_required_recipe_columns_present():
    df = load_recipes()
    for col in REQUIRED_RECIPE_COLUMNS:
        assert col in df.columns


def test_recipe_ids_are_unique():
    df = load_recipes()
    assert df["recipe_id"].is_unique


def test_cooking_time_is_numeric():
    df = load_recipes()
    assert pd.api.types.is_integer_dtype(df["cooking_time_minutes"])


def test_substitutions_load():
    df = load_substitutions()
    assert len(df) > 0
    assert "missing_ingredient" in df.columns


def test_parse_ingredients_splits_and_strips():
    assert parse_ingredients(" paneer, onion , tomato") == ["paneer", "onion", "tomato"]


def test_normalize_ingredient_cleans_case_whitespace_punctuation():
    assert normalize_ingredient("  Paneer  ") == "paneer"
    assert normalize_ingredient("Onion.") == "onion"
    assert normalize_ingredient("Red   Chilli  Powder") == "red chilli powder"


def test_normalize_ingredients_list():
    assert normalize_ingredients(" Paneer, Onion, Tomato ") == ["paneer", "onion", "tomato"]


def test_invalid_and_duplicate_rows_are_dropped_not_crashed():
    """A CSV with a fully blank row, a row missing a required field, and a
    duplicate id should load down to just the one genuinely valid row -
    without raising.
    """
    bad_csv = pd.DataFrame([
        {"recipe_id": "X1", "recipe_name": "Good Recipe", "ingredients": "a, b",
         "instructions": "do it", "cuisine": "Indian", "diet": "Vegan",
         "cooking_time_minutes": 10, "difficulty": "Easy"},
        {"recipe_id": "X1", "recipe_name": "Duplicate Id", "ingredients": "a, b",
         "instructions": "do it", "cuisine": "Indian", "diet": "Vegan",
         "cooking_time_minutes": 10, "difficulty": "Easy"},
        {"recipe_id": "X2", "recipe_name": "", "ingredients": "c, d",
         "instructions": "do it", "cuisine": "Indian", "diet": "Vegan",
         "cooking_time_minutes": 20, "difficulty": "Easy"},
        {"recipe_id": None, "recipe_name": None, "ingredients": None,
         "instructions": None, "cuisine": None, "diet": None,
         "cooking_time_minutes": None, "difficulty": None},
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        bad_csv.to_csv(f.name, index=False)
        path = Path(f.name)

    try:
        cleaned, stats = _load_and_clean(path, REQUIRED_RECIPE_COLUMNS, "recipe_id")
        assert stats["loaded"] == 4
        assert stats["dropped_invalid"] == 1  # the row with a blank recipe_name
        assert stats["dropped_duplicates"] == 1  # the second "X1"
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["recipe_name"] == "Good Recipe"
    finally:
        os.unlink(path)


def test_missing_required_column_raises_clearly():
    incomplete = pd.DataFrame([{"recipe_id": "X1", "recipe_name": "No Ingredients Column"}])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        incomplete.to_csv(f.name, index=False)
        path = Path(f.name)

    try:
        try:
            _load_and_clean(path, REQUIRED_RECIPE_COLUMNS, "recipe_id")
            assert False, "expected ValueError for missing required columns"
        except ValueError as e:
            assert "missing required column" in str(e).lower()
    finally:
        os.unlink(path)


def test_with_stats_matches_plain_loader():
    df, stats = load_recipes_with_stats()
    assert len(df) == stats["valid"]
    assert df.equals(load_recipes())

    sdf, sstats = load_substitutions_with_stats()
    assert len(sdf) == sstats["valid"]
    assert sdf.equals(load_substitutions())
