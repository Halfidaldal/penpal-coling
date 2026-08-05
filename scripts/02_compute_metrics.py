#!/usr/bin/env python
"""
Step 02: build similarity / distance / recurrence matrices and extract metrics.

CPU-only and fast, so it can be re-run freely with different thresholds without
touching the embeddings.

RQA is computed twice, under both standard thresholding strategies:
  - fixed-RR  : per-story epsilon giving a constant recurrence rate (prefix rr_)
                Use these for comparing *structure* across conditions.
  - fixed-eps : one corpus-wide epsilon (prefix eps_)
                Here RR itself is informative: how self-similar is this story.
Both are exported; see README for which to use when.

Outputs (into --outdir):
    story_metrics.csv          one row per story — the main deliverable
    lag_profiles.csv           long format: story_id, lag, mean_sim, n_pairs
    similarity_matrices.npz    per-story cosine similarity matrices
    distance_matrices.npz      per-story cosine distance matrices
    recurrence_matrices_rr.npz per-story binary recurrence (fixed-RR)
    recurrence_matrices_eps.npz per-story binary recurrence (fixed-eps)
    metrics_metadata.json      parameters and versions

Usage:
    python scripts/02_compute_metrics.py --outdir ../output --target-rr 0.05
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from story_recurrence.matrices import (
    cosine_matrices, recurrence_matrix, corpus_epsilon,
)
from story_recurrence.rqa import rqa_metrics
from story_recurrence.trajectory import trajectory_metrics
from story_recurrence.decay import decay_metrics
from story_recurrence.io_utils import load_embeddings, save_matrices, save_metadata


def parse_args():
    p = argparse.ArgumentParser(description="Compute story-structure metrics.")
    p.add_argument("--outdir", default="output",
                   help="directory holding step-01 outputs; results written here too")
    p.add_argument("--theiler", type=int, default=0,
                   help="exclude cells with |i-j| <= theiler. 0 = drop only the "
                        "main diagonal; 1 = also drop adjacent sentences")
    p.add_argument("--target-rr", type=float, default=0.05,
                   help="target recurrence rate for fixed-RR thresholding")
    p.add_argument("--l-min", type=int, default=2, help="min diagonal line length")
    p.add_argument("--v-min", type=int, default=2, help="min vertical line length")
    p.add_argument("--edge-frac", type=float, default=0.2,
                   help="opening/closing fraction used for circularity")
    p.add_argument("--decay-min-pairs", type=int, default=5,
                   help="min sentence pairs required to keep a lag")
    p.add_argument("--decay-max-lag-frac", type=float, default=0.5,
                   help="cap the largest lag at this fraction of story length")
    p.add_argument("--save-matrices", action="store_true", default=True,
                   help="export per-story matrices as .npz")
    p.add_argument("--no-save-matrices", dest="save_matrices", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)

    emb_path = outdir / "sentence_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found — run 01_compute_embeddings.py first.")

    embeddings = load_embeddings(emb_path)
    print(f"Loaded embeddings for {len(embeddings)} stories")

    stories_path = outdir / "stories.csv"
    stories = (pd.read_csv(stories_path, index_col=0)
               if stories_path.exists() else None)

    # ---- similarity / distance -----------------------------------------
    sims, dists = {}, {}
    for sid, E in tqdm(embeddings.items(), desc="Similarity matrices"):
        S, D = cosine_matrices(E)
        sims[sid], dists[sid] = S, D

    # One corpus-wide epsilon for the fixed-eps variant.
    eps_global = corpus_epsilon(dists, theiler=args.theiler, target_rr=args.target_rr)
    print(f"Corpus-wide epsilon (fixed-eps, target RR={args.target_rr}): {eps_global:.4f}")

    # ---- per-story metrics ---------------------------------------------
    rows, profiles = [], []
    rec_rr, rec_eps = {}, {}

    for sid, E in tqdm(embeddings.items(), desc="Metrics"):
        S, D = sims[sid], dists[sid]
        n = E.shape[0]
        row = {"story_id": sid, "n_sentences": n}

        # RQA under both thresholding strategies.
        R_rr, eps_rr = recurrence_matrix(
            D, theiler=args.theiler, mode="fixed_rr", target_rr=args.target_rr)
        R_eps, _ = recurrence_matrix(
            D, theiler=args.theiler, mode="fixed_eps", epsilon=eps_global)
        rec_rr[sid], rec_eps[sid] = R_rr, R_eps

        row["eps_story"] = eps_rr          # the per-story threshold itself is a measure
        row.update(rqa_metrics(R_rr, theiler=args.theiler, l_min=args.l_min,
                               v_min=args.v_min, prefix="rr_"))
        row.update(rqa_metrics(R_eps, theiler=args.theiler, l_min=args.l_min,
                               v_min=args.v_min, prefix="eps_"))

        # Trajectory geometry / embedding drift.
        row.update(trajectory_metrics(E, S, edge_frac=args.edge_frac))

        # Semantic decay.
        dmet, prof = decay_metrics(
            S, min_pairs=args.decay_min_pairs, max_lag_frac=args.decay_max_lag_frac)
        row.update(dmet)
        if prof is not None:
            lags, msim, cnt = prof
            for k, s, c in zip(lags, msim, cnt):
                profiles.append({"story_id": sid, "lag": int(k),
                                 "mean_sim": float(s), "n_pairs": int(c)})

        rows.append(row)

    metrics = pd.DataFrame(rows).set_index("story_id")

    # Attach condition / length columns for convenience in R.
    if stories is not None:
        stories.index.name = "story_id"
        for col in ("cond", "n_words"):
            if col in stories.columns:
                metrics[col] = stories[col].reindex(metrics.index)
        front = [c for c in ("cond", "n_sentences", "n_words") if c in metrics.columns]
        metrics = metrics[front + [c for c in metrics.columns if c not in front]]

    metrics.to_csv(outdir / "story_metrics.csv")
    print(f"\nSaved metrics -> {outdir / 'story_metrics.csv'}  "
          f"({metrics.shape[0]} stories x {metrics.shape[1]} columns)")

    pd.DataFrame(profiles).to_csv(outdir / "lag_profiles.csv", index=False)
    print(f"Saved lag profiles -> {outdir / 'lag_profiles.csv'} ({len(profiles)} rows)")

    if args.save_matrices:
        save_matrices(sims, outdir / "similarity_matrices.npz")
        save_matrices(dists, outdir / "distance_matrices.npz")
        save_matrices({k: v.astype(np.uint8) for k, v in rec_rr.items()},
                      outdir / "recurrence_matrices_rr.npz")
        save_matrices({k: v.astype(np.uint8) for k, v in rec_eps.items()},
                      outdir / "recurrence_matrices_eps.npz")

    save_metadata(
        {
            "step": "02_compute_metrics",
            "timestamp": datetime.now().isoformat(),
            "n_stories": int(metrics.shape[0]),
            "theiler": args.theiler,
            "target_rr": args.target_rr,
            "epsilon_fixed_corpus": eps_global,
            "l_min": args.l_min,
            "v_min": args.v_min,
            "edge_frac": args.edge_frac,
            "decay_min_pairs": args.decay_min_pairs,
            "decay_max_lag_frac": args.decay_max_lag_frac,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        },
        outdir / "metrics_metadata.json",
    )

    # ---- data-quality diagnostics --------------------------------------
    # These catch the two failure modes that would otherwise pass silently into
    # the R analysis: a target RR too low to form any diagonal lines, and
    # degenerate exponential decay fits.
    n_tot = len(metrics)
    print("\n--- diagnostics ---")
    for col, note in [
        ("rr_L", "stories with no diagonal line >= l_min (raise --target-rr if high)"),
        ("rr_ENTR", "same, for diagonal-length entropy"),
        ("decay_lambda", "stories where the exponential fit did not converge"),
    ]:
        if col in metrics.columns:
            n_nan = int(metrics[col].isna().sum())
            print(f"  {col:<14} NaN in {n_nan:>3}/{n_tot} stories  — {note}")
    if "decay_fit_at_bound" in metrics.columns:
        n_bound = int((metrics["decay_fit_at_bound"] == 1).sum())
        print(f"  {'decay fits':<14} on a parameter bound in {n_bound:>3}/{n_tot} "
              f"stories — treat these as degenerate (screen before modelling)")
    if "rr_RR" in metrics.columns:
        print(f"  achieved fixed-RR: mean {metrics['rr_RR'].mean():.4f} "
              f"(target {args.target_rr})")
    if "eps_RR" in metrics.columns:
        print(f"  fixed-eps RR:      mean {metrics['eps_RR'].mean():.4f}, "
              f"range [{metrics['eps_RR'].min():.4f}, {metrics['eps_RR'].max():.4f}]")

    # Quick console summary so a bad run is obvious immediately.
    if "cond" in metrics.columns:
        show = [c for c in ["rr_DET", "rr_LAM", "eps_RR", "straightness",
                            "radius_gyration", "decay_lambda", "decay_C"]
                if c in metrics.columns]
        print("\nCondition means (sanity check only — no inference here):")
        print(metrics.groupby("cond")[show].mean().round(3).to_string())

    print("\n✅ Step 02 complete.")


if __name__ == "__main__":
    main()
