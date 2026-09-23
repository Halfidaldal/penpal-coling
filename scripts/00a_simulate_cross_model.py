#!/usr/bin/env python3
"""
Script 00a: Simulate the cross-model LLM-LLM condition.

The EMNLP LLM-LLM condition paired every model with itself: 4 models x 20
stories = 80. This script generalises that to the full model x model grid, so
the two authors of a story can be different models and the opener is a factor
rather than a constant.

With the four configured models the grid has 16 ordered cells -- 4 self-pairs
on the diagonal and 12 cross-model cells off it. At the default 7 stories per
cell that is 112 stories, 1120 turns.

Generation itself is unchanged from the EMNLP simulation: same prompt for both
sides, same per-provider context handling, echo stripping, and symmetric 2..5
word truncation of every output before it is saved and passed on.

Outputs (paths from config.yaml, cross_simulation.paths):
  raw turn-level CSV     one row per turn, written incrementally
  story-level CSV        one row per story, in the interim/stories format
  id map CSV             story_id -> conversation_id
  run metadata JSON      settings, grid, and per-cell counts

Usage:
    # full grid, 7 stories per cell (112 stories)
    python scripts/00a_simulate_cross_model.py

    # see the grid and the cost of the run without calling any API
    python scripts/00a_simulate_cross_model.py --dry-run

    # cross-model cells only (12 cells), or the EMNLP same-model design
    python scripts/00a_simulate_cross_model.py --pairing cross
    python scripts/00a_simulate_cross_model.py --pairing same

    # target a total instead of a per-cell count
    python scripts/00a_simulate_cross_model.py --n-stories 100

    # pick up an interrupted run from its checkpoint
    python scripts/00a_simulate_cross_model.py --resume

API keys are read from the environment or a .env file: OPENAI_API_KEY,
ANTHROPIC_API_KEY, HF_API_KEY, OPENROUTER_API_KEY. Cells whose models have no key are
skipped with a warning.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from cross_sim.pairing import allocate_stories, build_pair_grid, describe_grid
from cross_sim.simulate import simulate_cross_model_dataset
from cross_sim.story_table import build_story_table
from penpal_config import load_config


def _short(path: Path) -> str:
    """Repo-relative if it is inside the repo, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    p = argparse.ArgumentParser(description="Simulate the cross-model LLM-LLM condition.")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument(
        "--pairing",
        choices=["all", "cross", "same"],
        default=None,
        help="'all' (default) is the full M x M grid, 'cross' the off-diagonal "
             "cells only, 'same' the diagonal (the EMNLP design)",
    )
    p.add_argument("--stories-per-pair", type=int, default=None, help="stories per cell")
    p.add_argument(
        "--n-stories",
        type=int,
        default=None,
        help="target total instead of a per-cell count; the remainder after "
             "dividing by the number of cells goes to the first cells, so the "
             "design is no longer balanced",
    )
    p.add_argument("--n-turns", type=int, default=None, help="turns per story")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--delay", type=float, default=None, help="seconds between stories")
    p.add_argument("--models", nargs="+", default=None, help="restrict to these model ids")
    p.add_argument("--seed", type=int, default=None, help="seed for truncation and counterbalancing")
    p.add_argument("--resume", action="store_true", help="skip stories already in the checkpoint")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    cfg = load_config(args.config)
    sim = cfg.get("cross_simulation") or {}
    if not sim:
        sys.exit("No 'cross_simulation' section in config.yaml")

    paths = sim.get("paths", {})
    pairing = args.pairing or sim.get("pairing", "all")
    stories_per_pair = args.stories_per_pair or sim.get("stories_per_pair", 7)
    n_turns = args.n_turns or sim.get("n_turns_per_story", 10)
    temperature = args.temperature if args.temperature is not None else sim.get("temperature", 1.0)
    max_tokens = args.max_tokens or sim.get("max_tokens", 40)
    delay = args.delay if args.delay is not None else sim.get("delay_between_stories", 2.0)
    seed = args.seed if args.seed is not None else sim.get("seed", 42)
    condition = sim.get("condition", "aa_cross")
    max_story_turns = sim.get("max_turns", n_turns)

    model_configs = sim.get("models") or []
    if not model_configs:
        sys.exit("No models configured under cross_simulation.models")
    if args.models:
        model_configs = [m for m in model_configs if m["id"] in args.models]
        if not model_configs:
            sys.exit(f"No configured models match {args.models}")

    model_ids = [m["id"] for m in model_configs]
    pairs = build_pair_grid(model_ids, mode=pairing)
    if not pairs:
        sys.exit(
            f"Pairing mode '{pairing}' yields no cells for {len(model_ids)} model(s). "
            f"Cross-model pairing needs at least two."
        )
    allocation = allocate_stories(
        pairs, stories_per_pair=stories_per_pair, n_stories=args.n_stories
    )
    total_stories = sum(allocation.values())

    print("=" * 72)
    print("Script 00a: cross-model LLM-LLM simulation")
    print("=" * 72)
    print(f"  models ({len(model_ids)}): {', '.join(model_ids)}")
    print(f"  pairing mode: {pairing}")
    print(f"  turns per story: {n_turns}   temperature: {temperature}   max tokens: {max_tokens}")
    print(f"  seed: {seed}   delay between stories: {delay}s")
    print(f"  condition code: {condition}")
    print("\nGrid:")
    print(describe_grid(allocation))
    print(f"\n  total generation calls: {total_stories * n_turns * 2}")

    print("\nAPI keys:")
    for env_key in dict.fromkeys(m["env_key"] for m in model_configs):
        mark = "set" if os.environ.get(env_key) else "MISSING"
        print(f"  {env_key:24s} {mark}")

    raw_path = ROOT / paths.get("raw_turns", "data/interim/stories/aa_cross_turns.csv")
    story_path = ROOT / paths.get(
        "story_table", "data/interim/stories/ai-ai-cross_stories_full_text_filtered.csv"
    )
    id_map_path = ROOT / paths.get("id_map", "data/interim/stories/aa_cross_id_map.csv")
    meta_path = ROOT / paths.get("run_metadata", "data/interim/stories/aa_cross_run_metadata.json")

    print("\nOutputs:")
    for label, path in [
        ("raw turns", raw_path), ("story table", story_path),
        ("id map", id_map_path), ("run metadata", meta_path),
    ]:
        print(f"  {label:14s} {_short(path)}")

    if args.dry_run:
        print("\n[dry run: nothing generated]")
        return

    for path in (raw_path, story_path, id_map_path, meta_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    df_existing = pd.DataFrame()
    if raw_path.exists():
        if args.resume:
            df_existing = pd.read_csv(raw_path)
            completed_ids = set(df_existing["story_id"].unique())
        else:
            sys.exit(
                f"\n{_short(raw_path)} already exists. Pass --resume to "
                f"continue that run, or move the file aside to start over."
            )

    # Seeds the per-turn truncation draw and, separately, the author swap.
    random.seed(seed)

    print("\n" + "=" * 72)
    df_new = simulate_cross_model_dataset(
        model_configs=model_configs,
        allocation=allocation,
        n_turns_per_story=n_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        delay_between_stories=delay,
        checkpoint_path=str(raw_path),
        completed_story_ids=completed_ids,
    )

    df_turns = pd.concat([df_existing, df_new], ignore_index=True) if len(df_existing) else df_new
    if df_turns.empty:
        sys.exit("\nNo stories generated.")

    table, id_frame = build_story_table(
        df_turns, condition=condition, max_turns=max_story_turns, seed=seed
    )
    table.to_csv(story_path, index=False)
    id_frame.to_csv(id_map_path, index=False)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "pairing_mode": pairing,
        "n_models": len(model_ids),
        "model_ids": model_ids,
        "n_cells": len(allocation),
        "stories_planned": total_stories,
        "stories_generated": int(table["conversation_id"].nunique()),
        "turns_per_story": n_turns,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "cells": [
            {
                "starter": pair.starter,
                "responder": pair.responder,
                "is_self_pair": pair.is_self_pair,
                "stories_planned": n,
                "stories_generated": int((table["pair_id"] == pair.pair_id).sum()),
            }
            for pair, n in allocation.items()
        ],
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  stories: {table['conversation_id'].nunique()} / {total_stories} planned")
    print(f"  turns:   {len(df_turns)}")
    print("\n  by cell:")
    print(table["pair_id"].value_counts().sort_index().to_string())
    print(f"\n  wrote {_short(story_path)}")
    print(f"  wrote {_short(id_map_path)}")
    print(f"  wrote {_short(meta_path)}")
    print("\nNext: add the condition to CONDITIONS in scripts/00_build_story_table.py")
    print("and re-run the 00 -> 01 -> 02 -> 03 pipeline.")


if __name__ == "__main__":
    main()
