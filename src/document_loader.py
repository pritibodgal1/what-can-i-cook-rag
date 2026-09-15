"""Load, validate and clean the CSV knowledge bases into pandas DataFrames.

`load_recipes()` / `load_substitutions()` are the existing public API the UI
and retriever already depend on - their signature and the shape of what they
return is unchanged, so nothing downstream needs to change. What's new here
is that they now validate required columns, drop unusable rows, dedupe by
id, and strip stray whitespace before handing the DataFrame back.

For callers that want visibility into *what* got cleaned (the ingestion
pipeline in chunker.py does), `load_recipes_with_stats()` and
`load_substitutions_with_stats()` return the same DataFrame plus a small
stats dict. Both loaders share one cleaning routine (`_load_and_clean`) so
there's a single place that knows how to validate a knowledge-base CSV.
"""

from pathlib import Path

import pandas as pd

from src import config

REQUIRED_RECIPE_COLUMNS = [
    "recipe_id", "recipe_name", "ingredients", "instructions",
    "cuisine", "diet", "cooking_time_minutes", "difficulty",
]
REQUIRED_SUBSTITUTION_COLUMNS = ["substitution_id", "missing_ingredient", "alternative", "method"]


def parse_ingredients(ingredients: str) -> list[str]:
    """Split a comma separated ingredient cell into a clean list."""
    return [item.strip() for item in str(ingredients).split(",") if item.strip()]


def normalize_ingredient(name: str) -> str:
    """Normalize one ingredient name for consistent matching.

    Lowercases, collapses internal whitespace, and trims stray leading/
    trailing punctuation. Deliberately leaves everything else alone (e.g.
    descriptive words like "fresh") - this is for matching, not for
    stripping information a later step might still want.
    """
    text = " ".join(str(name).split()).lower()
    return text.strip(" .,-")


def normalize_ingredients(raw: str) -> list[str]:
    """Parse a raw ingredients cell and normalize every item in it."""
    return [normalize_ingredient(item) for item in parse_ingredients(raw) if normalize_ingredient(item)]


def _is_blank(series: pd.Series) -> pd.Series:
    """True where a cell is missing or empty/whitespace once stringified."""
    as_text = series.astype(str).str.strip().str.lower()
    return series.isna() | as_text.isin(["", "nan", "none", "null"])


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace on every text column, leaving real nulls as nulls."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: str(v).strip() if pd.notna(v) else v)
    return df


def _load_and_clean(csv_path: Path, required_columns: list[str], id_column: str) -> tuple[pd.DataFrame, dict]:
    """Load a knowledge-base CSV, validate it, and drop unusable rows.

    Cleaning steps, in order (chosen so a genuinely-missing value is never
    mistaken for the valid text "nan" once columns get stringified):
    1. drop rows that are entirely empty
    2. drop rows missing a value in any required column
    3. strip whitespace on the surviving rows
    4. drop duplicate ids, keeping the first occurrence

    Returns the cleaned DataFrame plus a stats dict: loaded / valid /
    dropped_invalid / dropped_duplicates.
    """
    df = pd.read_csv(csv_path)
    loaded = len(df)

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path.name} is missing required column(s): {missing_cols}")

    df = df.dropna(how="all")

    blank_mask = pd.concat([_is_blank(df[c]) for c in required_columns], axis=1).any(axis=1)
    dropped_invalid = int(blank_mask.sum())
    df = df[~blank_mask]

    df = _strip_strings(df)

    dup_mask = df[id_column].duplicated(keep="first")
    dropped_duplicates = int(dup_mask.sum())
    df = df[~dup_mask]

    stats = {
        "loaded": loaded,
        "valid": len(df),
        "dropped_invalid": dropped_invalid,
        "dropped_duplicates": dropped_duplicates,
    }
    return df.reset_index(drop=True), stats


def load_recipes_with_stats() -> tuple[pd.DataFrame, dict]:
    """Load, validate and clean recipes.csv; also return cleaning stats."""
    df, stats = _load_and_clean(config.RECIPES_CSV, REQUIRED_RECIPE_COLUMNS, "recipe_id")

    numeric_time = pd.to_numeric(df["cooking_time_minutes"], errors="coerce")
    invalid_time = int(numeric_time.isna().sum())
    if invalid_time:
        df = df[numeric_time.notna()]
        numeric_time = numeric_time[numeric_time.notna()]
        stats["dropped_invalid"] += invalid_time
        stats["valid"] = len(df)
    df = df.copy()
    df["cooking_time_minutes"] = numeric_time.astype(int)

    return df.reset_index(drop=True), stats


def load_substitutions_with_stats() -> tuple[pd.DataFrame, dict]:
    """Load, validate and clean substitutions.csv; also return cleaning stats."""
    return _load_and_clean(config.SUBSTITUTIONS_CSV, REQUIRED_SUBSTITUTION_COLUMNS, "substitution_id")


def load_recipes() -> pd.DataFrame:
    """Return the cleaned recipe knowledge base."""
    df, _ = load_recipes_with_stats()
    return df


def load_substitutions() -> pd.DataFrame:
    """Return the cleaned ingredient substitution knowledge base."""
    df, _ = load_substitutions_with_stats()
    return df
