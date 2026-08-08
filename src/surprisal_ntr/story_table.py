"""
Build the story-level table from the annotation file.

Handles both layouts the project has used:

  wide (current)  one row per story, with ``mean_<stem>``, ``ann1_<stem>``,
                  ``ann2_<stem>`` and ``n_annotators`` columns.
  long (legacy)   one row per annotator, with bare ``<stem>`` columns; collapsed
                  by averaging within story.

The layout is detected from the columns present, so the same script works
against either file without a flag. Stories that nobody has rated are kept with
NaN ratings (see ``data.min_annotators``): the computational measures are
defined for them, and any rating-dependent analysis drops them downstream.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def detect_layout(df: pd.DataFrame, dimensions: Dict[str, str]) -> str:
    """Return "wide", "long" or "unknown" based on which columns are present."""
    stems = list(dimensions.values())
    if any(f"ann1_{s}" in df.columns or f"mean_{s}" in df.columns for s in stems):
        return "wide"
    if any(s in df.columns for s in stems):
        return "long"
    return "unknown"


def _ratings_wide(df: pd.DataFrame, dimensions: Dict[str, str]) -> pd.DataFrame:
    """
    Extract the six dimensions from a wide file.

    Prefers the precomputed ``mean_<stem>`` column; falls back to averaging
    ``ann1_``/``ann2_`` when it is absent.
    """
    out = pd.DataFrame(index=df.index)
    for name, stem in dimensions.items():
        mean_col = f"mean_{stem}"
        if mean_col in df.columns:
            out[name] = pd.to_numeric(df[mean_col], errors="coerce")
        else:
            cols = [c for c in (f"ann1_{stem}", f"ann2_{stem}") if c in df.columns]
            if cols:
                vals = df[cols].apply(pd.to_numeric, errors="coerce")
                out[name] = vals.mean(axis=1)
            else:
                out[name] = np.nan
    return out


def _n_annotators_wide(df: pd.DataFrame, dimensions: Dict[str, str]) -> pd.Series:
    """Number of annotators per story, from the column if present else inferred."""
    if "n_annotators" in df.columns:
        return pd.to_numeric(df["n_annotators"], errors="coerce").fillna(0).astype(int)
    stems = list(dimensions.values())
    n = pd.Series(0, index=df.index, dtype=int)
    for prefix in ("ann1_", "ann2_"):
        cols = [f"{prefix}{s}" for s in stems if f"{prefix}{s}" in df.columns]
        if cols:
            n += df[cols].notna().any(axis=1).astype(int)
    return n


def build_story_table(
    annotations: pd.DataFrame,
    data_cfg: dict,
) -> Tuple[pd.DataFrame, str]:
    """
    Collapse the annotation file to one row per story.

    Args:
        annotations: raw annotation dataframe.
        data_cfg: the ``data`` section of config.yaml.

    Returns:
        (story table indexed by id, detected layout)
        Columns: text, cond, n_annot, the six dimensions, composites, overall, llm.
    """
    id_col = data_cfg.get("id_col", "id")
    text_col = data_cfg.get("text_col", "text")
    cond_col = data_cfg.get("condition_col", "condition")
    dimensions: Dict[str, str] = data_cfg["dimensions"]
    dims: List[str] = list(dimensions.keys())

    for required in (id_col, text_col):
        if required not in annotations.columns:
            raise ValueError(
                f"Annotation file is missing '{required}'. "
                f"Available columns: {list(annotations.columns)}"
            )

    layout = detect_layout(annotations, dimensions)
    if layout == "unknown":
        raise ValueError(
            "Could not find the rating columns in either wide (mean_/ann1_/ann2_) "
            "or long (bare stem) layout. Check data.dimensions in config.yaml "
            f"against the file's columns: {list(annotations.columns)}"
        )

    df = annotations.copy()

    if layout == "wide":
        if df[id_col].duplicated().any():
            raise ValueError("Wide layout expects unique ids, but duplicates were found.")
        story = pd.DataFrame(index=df[id_col])
        story.index.name = id_col
        story["text"] = df[text_col].astype(str).values
        if cond_col in df.columns:
            story["cond"] = df[cond_col].astype(str).str.strip().str.lower().values
        story["n_annot"] = _n_annotators_wide(df, dimensions).values
        ratings = _ratings_wide(df, dimensions)
        for d in dims:
            story[d] = ratings[d].values

    else:  # long
        for d, stem in dimensions.items():
            if stem in df.columns:
                df[d] = pd.to_numeric(df[stem], errors="coerce")
        agg = {"text": (text_col, "first")}
        if cond_col in df.columns:
            agg["cond"] = (cond_col, "first")
        if "annotator" in df.columns:
            agg["n_annot"] = ("annotator", "nunique")
        for d in dims:
            if d in df.columns:
                agg[d] = (d, "mean")
        story = df.groupby(id_col).agg(**agg)
        if "cond" in story.columns:
            story["cond"] = story["cond"].astype(str).str.strip().str.lower()
        if "n_annot" not in story.columns:
            story["n_annot"] = story[dims].notna().any(axis=1).astype(int)

    # Map condition labels and ordinal coding.
    cond_map = {k.lower(): v for k, v in data_cfg.get("condition_map", {}).items()}
    if "cond" in story.columns:
        story["cond"] = story["cond"].map(cond_map).fillna(story["cond"])
        llmness = data_cfg.get("llmness", {})
        story["llm"] = story["cond"].map(llmness)

    # Composites and overall, computed only where components exist.
    for comp_name, members in (data_cfg.get("composites") or {}).items():
        present = [m for m in members if m in story.columns]
        if len(present) == len(members):
            story[comp_name] = story[present].mean(axis=1)
    present_dims = [d for d in dims if d in story.columns]
    if present_dims:
        story["overall"] = story[present_dims].mean(axis=1)

    # Optional filter on annotation coverage.
    min_ann = int(data_cfg.get("min_annotators", 0) or 0)
    if min_ann > 0:
        before = len(story)
        story = story[story["n_annot"] >= min_ann]
        print(f"Filtered to stories with >= {min_ann} annotators: {before} -> {len(story)}")

    # Order columns predictably.
    front = [c for c in ["cond", "llm", "n_annot", "text"] if c in story.columns]
    rest = [c for c in story.columns if c not in front]
    story = story[front + rest]

    return story, layout


def summarise(story: pd.DataFrame) -> None:
    """Print a short console summary of the story table."""
    print(f"Stories: {len(story)}")
    if "cond" in story.columns:
        print("By condition:")
        print(story["cond"].value_counts().to_string())
    if "n_annot" in story.columns:
        counts = story["n_annot"].value_counts().sort_index()
        print("By number of annotators:")
        print(counts.to_string())
        rated = int((story["n_annot"] > 0).sum())
        print(f"Rated stories: {rated} | unrated (ratings will be NaN): {len(story) - rated}")
