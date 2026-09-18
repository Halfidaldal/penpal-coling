# /// script
# dependencies = [
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "torch>=2.1",
#   "transformers>=4.40",
#   "accelerate>=0.30",
#   "tqdm>=4.65",
#   "huggingface-hub>=0.23.0",
# ]
# ///
"""
Hugging Face Job: endpoint predictability on a managed GPU (e.g. a100-large).

Self-contained by necessity — HF Jobs run a single file with no access to this
repository's src/, so the computation from src/endpoint_predictability/endpoint.py
is inlined here. Keep the two in sync when the metric definitions change.

Inputs
    Reads input.csv from the HF Dataset repo given by TARGET_HF_REPO.

Outputs (uploaded back to TARGET_HF_REPO)
    endpoint_metrics.csv        one row per story
    endpoint_trajectories.csv   long format, one row per reveal step
    endpoint_run_metadata.json  model, parameters, versions

Environment
    HF_TOKEN            (secret) write token
    TARGET_HF_REPO      dataset repo holding input.csv and receiving outputs
    MODEL_NAME          causal LM (default google/gemma-4-31b)
    TARGET_MODE         last_sentences | last_fraction
    N_TARGET_SENTENCES  ending length in sentences (default 2)
    TARGET_FRACTION     ending length as a fraction (default 0.1)
    MIN_BODY_SENTENCES  below this a story yields NaN (default 5)
    TAIL_FRACTION       tail share for the tail-drop ratio (default 0.2)
    ID_COL / TEXT_COL / CONDITION_COL
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- environment
HF_TOKEN = os.environ.get("HF_TOKEN")
TARGET_HF_REPO = os.environ.get("TARGET_HF_REPO")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-31b")

TARGET_MODE = os.environ.get("TARGET_MODE", "last_sentences")
N_TARGET_SENTENCES = int(os.environ.get("N_TARGET_SENTENCES", "2"))
TARGET_FRACTION = float(os.environ.get("TARGET_FRACTION", "0.1"))
MIN_BODY_SENTENCES = int(os.environ.get("MIN_BODY_SENTENCES", "5"))
TAIL_FRACTION = float(os.environ.get("TAIL_FRACTION", "0.2"))

ID_COL = os.environ.get("ID_COL", "id")
TEXT_COL = os.environ.get("TEXT_COL", "text")
CONDITION_COL = os.environ.get("CONDITION_COL", "condition")

CONDITION_MAP = {"hh": "HH", "ha": "H-LLM", "aa": "LLM-LLM"}
SENT_REGEX = re.compile(r"(?<=[.!?])\s+")


# -------------------------------------------------------------------- helpers
def split_sentences(text, min_words=1):
    """Split into sentences. min_words=1 keeps every sentence: each one is only
    appended to a growing context, so filtering would remove story text the
    model conditions on."""
    text = "" if text is None else str(text).strip()
    if not text:
        return []
    return [s.strip() for s in SENT_REGEX.split(text) if len(s.split()) >= min_words]


def split_body_target(sentences):
    n = len(sentences)
    if n < 2:
        return [], ""
    if TARGET_MODE == "last_fraction":
        k = max(1, int(round(TARGET_FRACTION * n)))
    else:
        k = max(1, N_TARGET_SENTENCES)
    k = min(k, n - 1)
    return sentences[:n - k], " ".join(sentences[n - k:])


@torch.no_grad()
def token_surprisal(model, ids, device):
    ids_dev = ids.to(device)
    logits = model(ids_dev.unsqueeze(0)).logits[0]
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    return -logp.gather(1, ids_dev[1:].unsqueeze(1)).squeeze(1) / math.log(2)


def mean_surprisal(model, device, bos_id, max_ctx, target, context=None):
    prefix = torch.tensor([bos_id], device=device)
    if context is not None and len(context) > 0:
        budget = max_ctx - len(target) - 1
        if budget > 0:
            prefix = torch.cat([prefix, context[-budget:].to(device)])
    full = torch.cat([prefix, target.to(device)])
    return float(token_surprisal(model, full, device)[-len(target):].mean().item())


def shape_stats(surprisals, tail_fraction):
    """Shape statistics of one endpoint-predictability trajectory (S_0 .. S_n)."""
    s0 = float(surprisals[0])
    s_final = float(surprisals[-1])
    drops = -np.diff(surprisals)
    n = drops.size
    cumulative = s0 - surprisals[1:]
    total_drop = s0 - s_final

    out = {
        "ep_s0": s0,
        "ep_s_final": s_final,
        "ep_total_drop": total_drop,
        "ep_n_body_sentences": float(n),
        "ep_mean_delta": float(np.mean(drops)),
        "ep_var_delta": float(np.var(drops)),
        "ep_max_delta": float(np.max(drops)),
        "ep_max_delta_pos": float((np.argmax(drops) + 1) / n),
        "ep_frac_negative_drops": float(np.mean(drops < 0)),
    }

    x_norm = np.arange(1, n + 1, dtype=float) / n
    if n >= 2:
        slope, intercept = np.polyfit(x_norm, cumulative, 1)
        pred = slope * x_norm + intercept
        ss_res = float(np.sum((cumulative - pred) ** 2))
        ss_tot = float(np.sum((cumulative - cumulative.mean()) ** 2))
        out["ep_slope_norm"] = float(slope)
        out["ep_linearity_r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        out["ep_slope_norm"] = float("nan")
        out["ep_linearity_r2"] = float("nan")

    if abs(total_drop) > 1e-9:
        frac = cumulative / total_drop
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        out["ep_auc"] = float(trapz(frac, x_norm))
    else:
        out["ep_auc"] = float("nan")

    tail_size = max(1, int(round(tail_fraction * n)))
    tail = drops[-tail_size:]
    rest = drops[:-tail_size]
    mean_tail = float(np.mean(tail))
    mean_rest = float(np.mean(rest)) if rest.size else float("nan")
    out["ep_mean_delta_tail"] = mean_tail
    out["ep_mean_delta_rest"] = mean_rest
    # Share of total gain earned in the tail: bounded and defined even when the
    # rest of the story contributes nothing. Prefer this over the legacy ratio.
    out["ep_tail_share"] = float(np.sum(tail) / total_drop) if abs(total_drop) > 1e-9 \
        else float("nan")
    out["ep_tail_ratio"] = (mean_tail / mean_rest) if (rest.size and abs(mean_rest) > 1e-9) \
        else float("nan")
    return out


NAN_KEYS = [
    "ep_s0", "ep_s_final", "ep_total_drop", "ep_n_body_sentences", "ep_mean_delta",
    "ep_var_delta", "ep_max_delta", "ep_max_delta_pos", "ep_frac_negative_drops",
    "ep_slope_norm", "ep_linearity_r2", "ep_auc", "ep_mean_delta_tail",
    "ep_mean_delta_rest", "ep_tail_share", "ep_tail_ratio",
]


def main():
    if not HF_TOKEN or not TARGET_HF_REPO:
        print("[ERROR] HF_TOKEN and TARGET_HF_REPO must both be set.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # 1. Input --------------------------------------------------------------
    print(f"Downloading input.csv from {TARGET_HF_REPO}...")
    csv_path = hf_hub_download(
        repo_id=TARGET_HF_REPO, filename="input.csv",
        repo_type="dataset", token=HF_TOKEN,
    )
    df = pd.read_csv(csv_path)
    if df[ID_COL].duplicated().any():
        df = df.groupby(ID_COL, as_index=False).first()
    stories = df.set_index(ID_COL)
    stories[TEXT_COL] = stories[TEXT_COL].astype(str)
    if CONDITION_COL in stories.columns:
        stories["cond"] = (stories[CONDITION_COL].astype(str).str.strip().str.lower()
                           .map(CONDITION_MAP).fillna(stories[CONDITION_COL]))
    print(f"Loaded {len(stories)} stories")
    if "cond" in stories.columns:
        print(stories["cond"].value_counts().to_string())

    # 2. Model --------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"\nLoading '{MODEL_NAME}' on {device} ({torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    if not hasattr(model, "hf_device_map") and device != "cpu":
        model = model.to(device)
    model.eval()

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    max_ctx = getattr(model.config, "n_positions",
                      getattr(model.config, "max_position_embeddings", 1024))
    print(f"BOS id {bos_id} | max context {max_ctx}")

    def encode(text):
        return tokenizer(" " + text.strip(), add_special_tokens=False,
                         return_tensors="pt").input_ids[0]

    # 3. Compute ------------------------------------------------------------
    summaries, traj = {}, []
    for sid, text in tqdm(stories[TEXT_COL].items(), total=len(stories),
                          desc=f"Endpoint ({MODEL_NAME})"):
        sentences = split_sentences(text)
        body, target = split_body_target(sentences)

        if not target or len(body) < MIN_BODY_SENTENCES:
            summary = {k: float("nan") for k in NAN_KEYS}
            summary["ep_n_sentences_total"] = float(len(sentences))
            summary["ep_n_target_sentences"] = float(len(sentences) - len(body)) if sentences else 0.0
            summaries[sid] = summary
            continue

        target_ids = encode(target)
        body_ids = [encode(s) for s in body]

        surprisals = [mean_surprisal(model, device, bos_id, max_ctx, target_ids, None)]
        context = torch.empty(0, dtype=torch.long)
        for ids in body_ids:
            context = torch.cat([context, ids])
            surprisals.append(
                mean_surprisal(model, device, bos_id, max_ctx, target_ids, context)
            )

        surprisals = np.asarray(surprisals, dtype=float)
        summary = shape_stats(surprisals, TAIL_FRACTION)
        summary["ep_n_sentences_total"] = float(len(sentences))
        summary["ep_n_target_sentences"] = float(len(sentences) - len(body))
        summaries[sid] = summary

        for i, s in enumerate(surprisals):
            traj.append({
                "id": sid,
                "cond": stories.loc[sid, "cond"] if "cond" in stories.columns else None,
                "step": i,
                "n_body_revealed": i,
                "pos_norm": i / len(body),
                "s_target": float(s),
                "drop": float(surprisals[i - 1] - s) if i > 0 else float("nan"),
                "cumulative_drop": float(surprisals[0] - s),
            })

    metrics = pd.DataFrame(summaries).T
    metrics.index.name = ID_COL
    for col in ("cond",):
        if col in stories.columns:
            metrics[col] = stories[col]
    traj_df = pd.DataFrame(traj)

    # 4. Save & upload ------------------------------------------------------
    outdir = Path("out_artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outdir / "endpoint_metrics.csv")
    traj_df.to_csv(outdir / "endpoint_trajectories.csv", index=False)

    with open(outdir / "endpoint_run_metadata.json", "w") as f:
        json.dump({
            "step": "04_compute_endpoint_predictability_hf_job",
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "dtype": str(torch_dtype),
            "device": device,
            "max_context": int(max_ctx),
            "target_mode": TARGET_MODE,
            "n_target_sentences": N_TARGET_SENTENCES,
            "target_fraction": TARGET_FRACTION,
            "min_body_sentences": MIN_BODY_SENTENCES,
            "tail_fraction": TAIL_FRACTION,
            "n_stories": int(len(metrics)),
            "n_usable": int(metrics["ep_total_drop"].notna().sum()),
            "n_trajectory_rows": int(len(traj_df)),
        }, f, indent=2, default=str)

    print(f"\nUsable stories: {int(metrics['ep_total_drop'].notna().sum())}/{len(metrics)}")
    if "cond" in metrics.columns:
        print(metrics.groupby("cond")[["ep_total_drop", "ep_linearity_r2", "ep_auc"]]
              .mean().round(3).to_string())

    print("\nUploading artifacts...")
    api.upload_folder(
        folder_path=str(outdir),
        repo_id=TARGET_HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"✅ HF Job finished! Artifacts uploaded to {TARGET_HF_REPO}")


if __name__ == "__main__":
    main()
