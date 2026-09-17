#!/usr/bin/env python3
"""
Script 00: Build the full story-level text table.

Combines the per-condition interim story files into one CSV with the three
columns the embedding step expects: story id, full story text, condition.

Unlike the annotation set (which covers 147 stories: 91 ha, 36 hh, 20 aa), this
includes *every* story, so the LLM-LLM condition contributes all 80 rather than
the 20 that were annotated. Total: 216 stories (100 ha, 36 hh, 80 aa).

The cross-model LLM-LLM condition (`aa_cross`) is included when
scripts/00a_simulate_cross_model.py has been run; its file is skipped if absent.

Input:  data/interim/stories/<condition>_stories_full_text_filtered.csv
Output: data/interim/stories/full_stories_all.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CONDITIONS = {
    "human-ai": "ha",
    "human-human": "hh",
    "ai-ai": "aa",
    # Cross-model LLM-LLM, produced by scripts/00a_simulate_cross_model.py.
    # Absent until that script has been run, so a missing file is not an error.
    "ai-ai-cross": "aa_cross",
}

# Conditions whose interim file may legitimately not exist yet.
OPTIONAL_CONDITIONS = {"ai-ai-cross"}

# Data-quality exclusions carried over from the EMNLP pipeline so the two papers
# work from the same story set. Both were already dropped before these interim
# files were written, so this filter currently removes nothing -- it is kept as a
# guard in case the interim files are ever regenerated from an earlier stage.
EXCLUDED_STORY_IDS = {
    "conv_ed575a06c11d42358e3eeb7826d2f959",
    "conv_a79338efb1384551affc0d7597822b0f",
}


def main():
    p = argparse.ArgumentParser(description="Build the combined story table.")
    p.add_argument("--indir", default=str(ROOT / "data" / "interim" / "stories"))
    p.add_argument("--output", default=str(ROOT / "data" / "interim" / "stories" / "full_stories_all.csv"))
    args = p.parse_args()

    indir = Path(args.indir)
    frames = []
    for condition, short in CONDITIONS.items():
        path = indir / f"{condition}_stories_full_text_filtered.csv"
        if not path.exists() and condition in OPTIONAL_CONDITIONS:
            print(f"{condition:12s}   -  not generated yet, skipping")
            continue
        df = pd.read_csv(path)[["conversation_id", "full_story"]]
        df["condition"] = short
        print(f"{condition:12s} {len(df):3d} stories  <- {path.name}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined["conversation_id"].isin(EXCLUDED_STORY_IDS)]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print(f"\nWrote {len(combined)} stories -> {out}")
    print(combined["condition"].value_counts().to_string())


if __name__ == "__main__":
    main()
