#!/usr/bin/env python3
"""
Step 00b: build the canonical sentence index.

Run this BEFORE the embedding (01), surprisal (03) and endpoint (04) steps. Each
of those consumes ``paths.sentence_index`` rather than re-segmenting, so that
sent_idx means the same thing everywhere and per-sentence measures from different
pipelines can be joined on (story_id, sent_idx).

This is not cosmetic. When the embedding pipeline used spaCy with a 3-word
minimum and the surprisal pipeline used a regex with none, only 27% of stories
segmented identically and in 72% of stories index i referred to different text --
regex does not split after a closing quotation mark, which is everywhere in
dialogue.

Output (path from config.yaml → paths.sentence_index):
    sentence_index.csv    story_id, sent_idx, n_words, sentence

Usage:
    python scripts/00b_build_sentence_index.py
    python scripts/00b_build_sentence_index.py --config configs/other.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from penpal_config import load_config, resolve_path
import penpal_segmentation as seg
from surprisal_ntr import story_table as st


def parse_args():
    p = argparse.ArgumentParser(description="Build the canonical sentence index.")
    p.add_argument("--config", default=None,
                   help="path to a config file (default: <repo root>/config.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Config: {cfg['_config_path']}\n")

    seg_cfg = cfg.get("segmentation", {}) or {}

    ann_path = resolve_path(cfg, "paths.annotations")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    print(f"Reading annotations: {ann_path}")

    story, layout = st.build_story_table(pd.read_csv(ann_path), cfg["data"])
    print(f"Detected annotation layout: {layout} | {len(story)} stories")

    print(f"\nSegmenting with '{seg_cfg.get('segmenter', 'spacy')}' "
          f"(model={seg_cfg.get('spacy_model')}, "
          f"min_words={seg_cfg.get('min_words', 1)})...")
    index = seg.build_sentence_index(story["text"].to_dict(), seg_cfg)

    print()
    seg.summarise(index, conditions=story["cond"] if "cond" in story.columns else None)

    empty = [sid for sid in story.index if sid not in set(index["story_id"])]
    if empty:
        print(f"\n[WARNING] {len(empty)} story/ies produced no sentences: {empty[:5]}")

    out = resolve_path(cfg, "paths.sentence_index")
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out, index=False)
    print(f"\nSaved canonical sentence index -> {out} ({len(index):,} sentences)")
    print("\nEvery downstream step must now consume this file. Re-run this script "
          "if the annotations or segmentation settings change.")
    print("\n✅ Step 00b complete.")


if __name__ == "__main__":
    main()
