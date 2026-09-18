#!/usr/bin/env python3
"""
Step 04: endpoint predictability.

How the surprisal of each story's ENDING falls as the story accumulates. Unlike
the window measures in step 03 this is a global measure: every point conditions
on all preceding text against one fixed target, so it asks whether a story
builds toward its ending or arrives at it abruptly.

Thin driver. Hyperparameters live in config.yaml (``endpoint`` section); the
computation lives in src/endpoint_predictability/.

Outputs (paths from config.yaml → paths):
    endpoint_metrics.csv        one row per story
    endpoint_trajectories.csv   long format: id, step, s_target, drop, cumulative_drop
    endpoint_run_metadata.json  model, parameters, versions, timestamp

Usage:
    python scripts/04_compute_endpoint_predictability.py
    python scripts/04_compute_endpoint_predictability.py --config configs/other.yaml
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

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute endpoint predictability (configured via config.yaml)."
    )
    p.add_argument("--config", default=None,
                   help="path to a config file (default: <repo root>/config.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"Config: {cfg['_config_path']}\n")

    ep_cfg = cfg.get("endpoint", {}) or {}
    if not ep_cfg.get("enabled", True):
        print("endpoint.enabled is false in config; nothing to do.")
        return

    progress = bool(ep_cfg.get("progress", True))

    # ---- 1. Story table -------------------------------------------------
    ann_path = resolve_path(cfg, "paths.annotations")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    print(f"Reading annotations: {ann_path}")

    story, layout = st.build_story_table(pd.read_csv(ann_path), cfg["data"])
    print(f"Detected annotation layout: {layout}")
    st.summarise(story)

    # ---- 2. Model -------------------------------------------------------
    from surprisal_ntr import lm as lm_mod
    from endpoint_predictability import endpoint as ep

    lm_bundle = lm_mod.load_model(ep_cfg, hf_token=resolve_token(cfg, "endpoint.hf_token"))
    print(f"Target: {ep_cfg.get('target_mode', 'last_sentences')} "
          f"(n={ep_cfg.get('n_target_sentences', 2)}, "
          f"frac={ep_cfg.get('target_fraction', 0.1)}) | "
          f"max context: {lm_bundle.max_context} tokens")

    # ---- 3. Compute -----------------------------------------------------
    # Use the canonical sentence units so this pipeline segments identically to
    # the embedding and surprisal pipelines.
    sentence_index = None
    if ep_cfg.get("use_sentence_index", True):
        import penpal_segmentation as seg
        idx_path = resolve_path(cfg, "paths.sentence_index")
        if idx_path.exists():
            sentence_index = seg.load_sentence_index(idx_path)
            print(f"Units: canonical sentences from {idx_path} "
                  f"({len(sentence_index):,} total)")
        else:
            raise FileNotFoundError(
                f"{idx_path} not found. Run scripts/00b_build_sentence_index.py "
                f"first, or set endpoint.use_sentence_index: false to re-segment."
            )

    print("\n--- Endpoint predictability ---")
    summary_df, traj_df = ep.compute_corpus_endpoint(
        story, lm_bundle, ep_cfg, sentence_index=sentence_index, progress=progress
    )

    metrics = summary_df.copy()
    for col in ("cond", "llm", "n_annot"):
        if col in story.columns:
            metrics[col] = story[col]
    front = [c for c in ("cond", "llm", "n_annot") if c in metrics.columns]
    metrics = metrics[front + [c for c in metrics.columns if c not in front]]
    metrics.index.name = story.index.name

    # ---- 4. Export ------------------------------------------------------
    out = resolve_path(cfg, "paths.endpoint_metrics")
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out)
    print(f"\nSaved endpoint metrics -> {out} "
          f"({metrics.shape[0]} stories x {metrics.shape[1]} columns)")

    if ep_cfg.get("save_trajectories", True) and not traj_df.empty:
        traj_out = resolve_path(cfg, "paths.endpoint_trajectories")
        traj_out.parent.mkdir(parents=True, exist_ok=True)
        traj_df.to_csv(traj_out, index=False)
        print(f"Saved trajectories -> {traj_out} ({len(traj_df)} rows)")

    meta_out = resolve_path(cfg, "paths.endpoint_metadata")
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config_path": cfg["_config_path"],
            "annotations": str(ann_path),
            "annotation_layout": layout,
            "n_stories": int(len(metrics)),
            "n_usable": int(metrics["ep_total_drop"].notna().sum()),
            "condition_counts": story["cond"].value_counts().to_dict()
            if "cond" in story.columns else None,
            "endpoint": {k: v for k, v in ep_cfg.items() if k != "hf_token"},
            "model_dtype": lm_bundle.dtype,
            "model_device": lm_bundle.device,
            "max_context": lm_bundle.max_context,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        }, f, indent=2, default=str)
    print(f"Saved run metadata -> {meta_out}")

    # ---- 5. Console sanity check ---------------------------------------
    n_nan = int(metrics["ep_total_drop"].isna().sum())
    if n_nan:
        print(f"\n[note] {n_nan} story/ies had fewer than "
              f"{ep_cfg.get('min_body_sentences', 5)} body sentences -> NaN metrics")
    if "cond" in metrics.columns:
        show = ["ep_total_drop", "ep_linearity_r2", "ep_auc", "ep_tail_ratio"]
        print("\nCondition means (sanity check only):")
        print(metrics.groupby("cond")[show].mean().round(3).to_string())

    print("\n✅ Step 04 complete.")


if __name__ == "__main__":
    main()
