"""
Loading stories and writing outputs.

The input is any CSV with one column of story ids, one of story text and
(optionally) one of condition labels. Annotation files that carry one row per
annotator are de-duplicated to one row per story.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd


CONDITION_MAP = {"hh": "HH", "ha": "H-LLM", "aa": "LLM-LLM",
                 "human-human": "HH", "human-ai": "H-LLM", "ai-ai": "LLM-LLM"}


def load_stories(
    path: str,
    id_col: str = "id",
    text_col: str = "text",
    condition_col: Optional[str] = "condition",
    normalise_conditions: bool = True,
) -> pd.DataFrame:
    """
    Load one row per story.

    Args:
        path: CSV path.
        id_col / text_col / condition_col: column names in the source file.
        normalise_conditions: map raw labels (hh/ha/aa) to HH/H-LLM/LLM-LLM.

    Returns:
        DataFrame indexed by story id with columns ['text', 'cond'].
    """
    df = pd.read_csv(path)

    missing = [c for c in (id_col, text_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}. "
                         f"Available: {list(df.columns)}")

    rename_map = {id_col: "story_id", text_col: "text"}
    if condition_col and condition_col in df.columns:
        rename_map[condition_col] = "cond"

    out = df.rename(columns=rename_map)
    # Deduplicate / collapse to one row per story if needed
    out = out.groupby("story_id", as_index=True).first()

    if "cond" in out.columns and normalise_conditions:
        out["cond"] = (out["cond"].astype(str).str.strip().str.lower()
                       .map(CONDITION_MAP).fillna(out["cond"]))

    out["text"] = out["text"].astype(str)
    print(f"Loaded {len(out)} stories from {path}")
    if "cond" in out.columns:
        print(out["cond"].value_counts().to_string())
    return out


def save_embeddings(embeddings: Dict[str, np.ndarray], path: Path) -> None:
    """Save per-story sentence embeddings to a compressed .npz keyed by story id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in embeddings.items()})
    print(f"Saved embeddings -> {path}")


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    """Load per-story sentence embeddings from .npz."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def save_matrices(matrices: Dict[str, np.ndarray], path: Path) -> None:
    """Save per-story matrices (similarity, distance or recurrence) to .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(k): v for k, v in matrices.items()})
    print(f"Saved matrices -> {path}")


def save_metadata(meta: dict, path: Path) -> None:
    """Write a JSON run-metadata sidecar (model, parameters, versions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved metadata -> {path}")


def sentence_index(story_sentences: Dict[str, list]) -> pd.DataFrame:
    """Long table of every sentence, for traceability and for use in R."""
    rows = []
    for sid, sents in story_sentences.items():
        for i, s in enumerate(sents):
            rows.append({"story_id": sid, "sent_idx": i,
                         "n_words": len(s.split()), "sentence": s})
    return pd.DataFrame(rows)
