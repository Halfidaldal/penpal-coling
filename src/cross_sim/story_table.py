"""
Turn-level output -> story-level table.

Produces the same shape as data/interim/stories/ai-ai_stories_full_text_filtered.csv
so scripts/00_build_story_table.py can pick the new condition up without
special-casing, with the model-pair columns added on top.

Two things are carried over deliberately from the EMNLP cleaning step so the
new stories stay comparable with the existing 80:

- the "This is the story of" primer is stripped from the opener before any
  text is concatenated, exactly as clean_ai_ai_data did;
- the author_1 / author_2 column labels are counterbalanced, swapped for a
  seeded ~50% of stories, with starter / starter_side recording which column
  now holds the opener. This removes the positional confound that would
  otherwise make author_1 synonymous with "went first".

The model columns follow the swap: model_author_1 / model_author_2 track the
columns, while model_starter / model_responder are swap-invariant and are the
ones to analyse by.
"""

from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

STORY_PREFIX = "This is the story of"


def _normalize_story_text(value: Any) -> str:
    """Collapse embedded newlines so story text is built as a single line."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _concat(series: pd.Series) -> str:
    return " ".join(t for t in series.map(_normalize_story_text).tolist() if t)


def _concat_with_period(series: pd.Series) -> str:
    return " ".join(f"{t}." for t in series.map(_normalize_story_text).tolist() if t)


def strip_primer(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the story primer from the opener's first turn."""
    df = df.copy()
    for col in ("agent_1", "agent_2"):
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(f"{STORY_PREFIX}\n", "", regex=False)
            .str.replace(STORY_PREFIX, "", regex=False)
            .str.strip()
        )
    return df


def build_story_table(
    df_turns: pd.DataFrame,
    condition: str = "aa_cross",
    max_turns: int = 10,
    seed: int = 42,
    swap_probability: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse turn-level simulation output into one row per story.

    Args:
        df_turns: output of simulate_cross_model_dataset.
        condition: condition code written to every row.
        max_turns: turns beyond this are dropped.
        seed: seed for the author-column counterbalancing.
        swap_probability: probability a story has its author columns swapped.

    Returns:
        (story_table, id_map). id_map is story_id -> conversation_id, kept
        separately so a blinded annotation export can drop the model columns
        and still be traced back.
    """
    if df_turns.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = strip_primer(df_turns)
    df = df[df["turn"] <= max_turns].copy()
    # A resumed run reads the checkpoint back from CSV, where is_self_pair has
    # become the string "True"/"False" and bool("False") is True.
    df["is_self_pair"] = df["model_starter"] == df["model_responder"]

    # Opaque ids, so nothing downstream can read the model pairing off the id.
    story_ids = df["story_id"].dropna().unique()
    id_map = {sid: f"conv_{uuid4().hex}" for sid in story_ids}
    df["conversation_id"] = df["story_id"].map(id_map)
    df["interaction_count"] = df["turn"]

    df = df.sort_values(["story_id", "turn"], kind="stable")

    records: List[Dict[str, Any]] = []
    for story_id, grp in df.groupby("story_id", sort=True):
        first = grp.iloc[0]
        records.append({
            "conversation_id": first["conversation_id"],
            "story_id": story_id,
            # agent_1 is the starter, agent_2 the responder; the swap below
            # may reassign these to the other column.
            "full_author_1": _concat(grp["agent_1"]),
            "full_author_2": _concat(grp["agent_2"]),
            "full_author_1_dot": _concat_with_period(grp["agent_1"]),
            "full_author_2_dot": _concat_with_period(grp["agent_2"]),
            "condition": condition,
            "timestamp": first["timestamp"],
            "interaction_count": 1,
            "turn": 1,
            "starter": "author_1",
            "author_1_type": "ai",
            "author_2_type": "ai",
            "starter_type": "ai",
            "model_author_1": first["model_starter"],
            "model_author_2": first["model_responder"],
            "model_starter": first["model_starter"],
            "model_responder": first["model_responder"],
            "pair_id": first["pair_id"],
            "dyad_id": first["dyad_id"],
            "is_self_pair": bool(first["is_self_pair"]),
            "n_turns": len(grp),
        })

    table = pd.DataFrame(records)

    # full_story is chronological: starter then responder, turn by turn, and
    # is built before the swap so it never depends on the column labels.
    table["full_story"] = [
        _build_chronological_story(df[df["story_id"] == sid])
        for sid in table["story_id"]
    ]

    table = counterbalance_authors(table, seed=seed, swap_probability=swap_probability)
    table["starter_side"] = table["starter"]

    # model_id keeps the column the existing ai-ai file uses populated; for a
    # cross-model story a single model id is meaningless, so it carries the
    # ordered pair.
    table["model_id"] = table["pair_id"]

    column_order = [
        "conversation_id", "full_author_1", "full_author_2", "condition",
        "timestamp", "interaction_count", "starter", "starter_side",
        "starter_type", "author_1_type", "author_2_type", "model_id", "turn",
        "full_author_1_dot", "full_author_2_dot", "full_story",
        "model_author_1", "model_author_2", "model_starter", "model_responder",
        "pair_id", "dyad_id", "is_self_pair", "n_turns",
    ]
    table = table[column_order]

    id_frame = pd.DataFrame(
        sorted(id_map.items()), columns=["story_id", "conversation_id"]
    )
    return table, id_frame


def _build_chronological_story(grp: pd.DataFrame) -> str:
    parts = []
    for _, row in grp.sort_values("turn", kind="stable").iterrows():
        for col in ("agent_1", "agent_2"):
            text = _normalize_story_text(row[col])
            if text:
                parts.append(text)
    return " ".join(parts)


def counterbalance_authors(
    table: pd.DataFrame,
    seed: int = 42,
    swap_probability: float = 0.5,
) -> pd.DataFrame:
    """
    Swap the author_1 / author_2 labels for a seeded subset of stories.

    Mirrors randomize_author_assignment in the EMNLP pipeline. The text, model
    labels and starter flag all move together, so the row stays internally
    consistent; full_story is untouched because it is already chronological.
    """
    table = table.copy()
    rng = np.random.default_rng(seed)
    swap = rng.random(len(table)) < swap_probability

    pairs = [
        ("full_author_1", "full_author_2"),
        ("full_author_1_dot", "full_author_2_dot"),
        ("model_author_1", "model_author_2"),
    ]
    for left, right in pairs:
        table.loc[swap, [left, right]] = table.loc[swap, [right, left]].values

    table.loc[swap, "starter"] = table.loc[swap, "starter"].map(
        {"author_1": "author_2", "author_2": "author_1"}
    )

    print(f"Author counterbalancing: swapped {int(swap.sum())}/{len(table)} stories")
    return table
