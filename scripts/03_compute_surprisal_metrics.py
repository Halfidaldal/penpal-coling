#!/usr/bin/env python3
"""
Step 03: stylometrics and LM-surprisal Novelty / Transience / Resonance.

Thin driver. Every hyperparameter lives in config.yaml; the computation lives in
src/surprisal_ntr/. The only command-line argument points at an alternative
config file, so several presets can be kept side by side.

Outputs (paths come from config.yaml → paths):
    penpal_measures_all.csv        story-level table: ratings + stylometry + NTR
    surprisal_window_level.csv     one row per window (primary modelling unit)
    surprisal_run_metadata.json    model, parameters, versions, timestamp

Usage:
    python scripts/03_compute_surprisal_metrics.py
    python scripts/03_compute_surprisal_metrics.py --config configs/gemma_w28.yaml
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from penpal_config import load_config, get, resolve_path, resolve_token
from surprisal_ntr import story_table as st
from surprisal_ntr import stylometry, ntr

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute stylometrics and surprisal NTR (configured via config.yaml)."
    )
    p.add_argument("--config", default=None,
                   help="path to a config file (default: <repo root>/config.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Config: {cfg['_config_path']}\n")

    data_cfg = cfg["data"]
    style_cfg = cfg.get("stylometry", {}) or {}
    surp_cfg = cfg.get("surprisal", {}) or {}
    progress = bool(get(cfg, "surprisal.progress", True))

    # ---- 1. Story table -------------------------------------------------
    ann_path = resolve_path(cfg, "paths.annotations")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    print(f"Reading annotations: {ann_path}")

    annotations = pd.read_csv(ann_path)
    story, layout = st.build_story_table(annotations, data_cfg)
    print(f"Detected annotation layout: {layout}")
    st.summarise(story)

    # ---- 2. Stylometry --------------------------------------------------
    if style_cfg.get("enabled", True):
        print("\n--- Stylometric features ---")
        story = story.join(stylometry.compute(story, style_cfg, progress=progress))
    else:
        print("\nStylometry disabled in config; skipping.")

    # ---- 3. Surprisal NTR -----------------------------------------------
    lm_bundle = None
    window_df = pd.DataFrame()
    if surp_cfg.get("enabled", True):
        print("\n--- LM surprisal NTR ---")
        from surprisal_ntr import lm as lm_mod

        lm_bundle = lm_mod.load_model(surp_cfg, hf_token=resolve_token(cfg))
        print(f"Window: {surp_cfg['window_words']} words | "
              f"max context: {lm_bundle.max_context} tokens")

        summary_df, window_df = ntr.compute_corpus_ntr(
            story, lm_bundle, surp_cfg, progress=progress
        )
        story = story.join(summary_df)

        gap = ntr.identity_gap(story)
        print(f"\nIdentity check  max |resonance - (novelty - transience)| = {gap:.2e}")
        if gap > 1e-6:
            print("  WARNING: the three story means disagree. Set "
                  "surprisal.common_index_range: true in config.yaml.")
    else:
        print("\nSurprisal disabled in config; skipping.")

    # ---- 4. Export ------------------------------------------------------
    story_out = resolve_path(cfg, "paths.story_metrics")
    story_out.parent.mkdir(parents=True, exist_ok=True)
    export = story.drop(columns=["text"], errors="ignore")
    export.to_csv(story_out)
    print(f"\nSaved story metrics -> {story_out} "
          f"({export.shape[0]} stories x {export.shape[1]} columns)")

    if not window_df.empty and surp_cfg.get("save_window_level", True):
        win_out = resolve_path(cfg, "paths.window_metrics")
        win_out.parent.mkdir(parents=True, exist_ok=True)
        window_df.to_csv(win_out, index=False)
        print(f"Saved window metrics -> {win_out} ({len(window_df)} windows, "
              f"mean {window_df.groupby('id').size().mean():.1f} per story)")

    meta_out = resolve_path(cfg, "paths.run_metadata")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "config_path": cfg["_config_path"],
        "annotations": str(ann_path),
        "annotation_layout": layout,
        "n_stories": int(len(story)),
        "n_rated_stories": int((story["n_annot"] > 0).sum()) if "n_annot" in story else None,
        "condition_counts": story["cond"].value_counts().to_dict() if "cond" in story else None,
        "stylometry": style_cfg,
        "surprisal": {k: v for k, v in surp_cfg.items() if k != "hf_token"},
        "model_dtype": lm_bundle.dtype if lm_bundle else None,
        "model_device": lm_bundle.device if lm_bundle else None,
        "max_context": lm_bundle.max_context if lm_bundle else None,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved run metadata -> {meta_out}")

    # ---- 5. Console sanity check ---------------------------------------
    if "cond" in story.columns and "s_novelty" in story.columns:
        print("\nCondition means (sanity check only):")
        print(story.groupby("cond")[["s_novelty", "s_transience", "s_resonance"]]
              .agg(["mean", "count"]).round(3).to_string())

    print("\n✅ Step 03 complete.")


if __name__ == "__main__":
    main()
