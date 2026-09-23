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
Hugging Face Job: surprisal Novelty / Transience / Resonance on a managed GPU.

Units come from the CANONICAL sentence index uploaded alongside the input, so
this job never segments text itself. That is deliberate: segmentation drift
between pipelines previously misaligned sent_idx in 72% of stories, and a remote
copy of the splitter would reintroduce exactly that risk. With UNIT=window the
job derives fixed-width word windows instead, which needs no segmentation.

Definitions (identical to src/surprisal_ntr/ntr.py):

    Novelty_t    = s(T_t     | C_t)      - s(T_t     | BOS)
    Transience_t = s(T_{t+1} | C_t, T_t) - s(T_{t+1} | C_t)
    Resonance_t  = Novelty_t - Transience_t

Inputs (from the HF Dataset repo named by TARGET_HF_REPO)
    input.csv           one row per story: id, text, condition
    sentence_index.csv  canonical units: story_id, sent_idx, n_words, sentence
                        (required when UNIT=sentence)

Outputs (uploaded back to TARGET_HF_REPO)
    surprisal_story_level.csv   one row per story
    surprisal_unit_level.csv    one row per unit; carries sent_idx when
                                UNIT=sentence, for joining to the RQA measures
    surprisal_run_metadata.json

Environment
    HF_TOKEN            (secret) write token
    TARGET_HF_REPO      dataset repo holding the inputs and receiving outputs
    MODEL_NAME          causal LM (default google/gemma-4-31b)
    UNIT                sentence | window   (default sentence)
    WINDOW_WORDS        window length when UNIT=window (default 14)
    MIN_WINDOW_WORDS    drop trailing short windows (default 3)
    COMMON_INDEX_RANGE  1 to aggregate all three over the interior range
    ID_COL / TEXT_COL / CONDITION_COL
"""

import json
import math
import os
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

UNIT = os.environ.get("UNIT", "sentence")
WINDOW_WORDS = int(os.environ.get("WINDOW_WORDS", "14"))
MIN_WINDOW_WORDS = int(os.environ.get("MIN_WINDOW_WORDS", "3"))
COMMON_INDEX_RANGE = os.environ.get("COMMON_INDEX_RANGE", "1") not in ("0", "false", "False")

ID_COL = os.environ.get("ID_COL", "id")
TEXT_COL = os.environ.get("TEXT_COL", "text")
CONDITION_COL = os.environ.get("CONDITION_COL", "condition")

CONDITION_MAP = {
    "hh": "HH", "ha": "H-LLM", "aa": "LLM-LLM", "aa_cross": "LLM-LLM-cross",
}


# -------------------------------------------------------------------- helpers
def make_windows(text, window_words, min_window_words):
    words = str(text).split()
    return [
        " ".join(words[i:i + window_words])
        for i in range(0, len(words), window_words)
        if len(words[i:i + window_words]) >= min_window_words
    ]


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


def story_ntr(units, encode, model, device, bos_id, max_ctx):
    """Novelty / transience / resonance for one story's ordered units."""
    n = len(units)
    if n < 2:
        return ({"s_novelty": np.nan, "s_transience": np.nan, "s_resonance": np.nan,
                 "n_units_total": n, "n_units_used": 0}, [])

    ids = [encode(u) for u in units]
    base = [mean_surprisal(model, device, bos_id, max_ctx, ids[i]) for i in range(n)]
    s_ctx = [base[0]] + [
        mean_surprisal(model, device, bos_id, max_ctx, ids[i], torch.cat(ids[:i]))
        for i in range(1, n)
    ]

    nov = np.full(n, np.nan)
    tra = np.full(n, np.nan)
    for i in range(1, n):
        nov[i] = s_ctx[i] - base[i]
    for i in range(n - 1):
        ctx_before = torch.cat(ids[:i]) if i > 0 else None
        tra[i] = s_ctx[i + 1] - mean_surprisal(
            model, device, bos_id, max_ctx, ids[i + 1], ctx_before
        )
    res = nov - tra

    records = [{
        "unit_idx": i,
        "n_unit_words": len(units[i].split()),
        "s_base": float(base[i]),
        "s_ctx": float(s_ctx[i]),
        "s_novelty": float(nov[i]),
        "s_transience": float(tra[i]),
        "s_resonance": float(res[i]),
        "unit_text": units[i],
    } for i in range(n)]

    lo, hi = (1, n - 1) if COMMON_INDEX_RANGE else (0, n)
    if hi <= lo:
        summary = {"s_novelty": np.nan, "s_transience": np.nan, "s_resonance": np.nan,
                   "n_units_total": n, "n_units_used": 0}
    else:
        summary = {
            "s_novelty": float(np.nanmean(nov[lo:hi])),
            "s_transience": float(np.nanmean(tra[lo:hi])),
            "s_resonance": float(np.nanmean(res[lo:hi])),
            "n_units_total": n,
            "n_units_used": int(hi - lo),
        }
    return summary, records


def main():
    if not HF_TOKEN or not TARGET_HF_REPO:
        print("[ERROR] HF_TOKEN and TARGET_HF_REPO must both be set.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # 1. Inputs -------------------------------------------------------------
    print(f"Downloading input.csv from {TARGET_HF_REPO}...")
    df = pd.read_csv(hf_hub_download(
        repo_id=TARGET_HF_REPO, filename="input.csv",
        repo_type="dataset", token=HF_TOKEN))
    if df[ID_COL].duplicated().any():
        df = df.groupby(ID_COL, as_index=False).first()
    stories = df.set_index(ID_COL)
    stories[TEXT_COL] = stories[TEXT_COL].astype(str)
    if CONDITION_COL in stories.columns:
        stories["cond"] = (stories[CONDITION_COL].astype(str).str.strip().str.lower()
                           .map(CONDITION_MAP).fillna(stories[CONDITION_COL]))
    print(f"Loaded {len(stories)} stories")

    units_by_story = {}
    if UNIT == "sentence":
        print("Downloading canonical sentence_index.csv...")
        idx = pd.read_csv(hf_hub_download(
            repo_id=TARGET_HF_REPO, filename="sentence_index.csv",
            repo_type="dataset", token=HF_TOKEN))
        idx["sentence"] = idx["sentence"].astype(str)
        idx = idx.sort_values(["story_id", "sent_idx"])
        for sid, grp in idx.groupby("story_id", sort=False):
            units_by_story[sid] = grp["sentence"].tolist()
        missing = [s for s in stories.index if s not in units_by_story]
        if missing:
            print(f"[WARNING] {len(missing)} story/ies absent from the index: {missing[:5]}")
        units_by_story = {s: units_by_story.get(s, []) for s in stories.index}
        print(f"Units: {len(idx):,} canonical sentences "
              f"(mean {len(idx) / max(1, len(stories)):.1f} per story)")
    elif UNIT == "window":
        units_by_story = {s: make_windows(t, WINDOW_WORDS, MIN_WINDOW_WORDS)
                          for s, t in stories[TEXT_COL].items()}
        total = sum(len(v) for v in units_by_story.values())
        print(f"Units: {total:,} windows of {WINDOW_WORDS} words")
    else:
        print(f"[ERROR] Unknown UNIT={UNIT!r} (expected 'sentence' or 'window')")
        sys.exit(1)

    # 2. Model --------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"\nLoading '{MODEL_NAME}' on {device} ({torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True, token=HF_TOKEN)
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
    summaries, unit_rows = {}, []
    for sid, units in tqdm(units_by_story.items(), total=len(units_by_story),
                           desc=f"Surprisal [{UNIT}] ({MODEL_NAME})"):
        summary, records = story_ntr(units, encode, model, device, bos_id, max_ctx)
        summary["unit"] = UNIT
        summaries[sid] = summary
        for rec in records:
            rec["id"] = sid
            rec["unit"] = UNIT
            if UNIT == "sentence":
                rec["sent_idx"] = rec["unit_idx"]
            if "cond" in stories.columns:
                rec["cond"] = stories.loc[sid, "cond"]
            unit_rows.append(rec)

    story_df = pd.DataFrame(summaries).T
    story_df.index.name = ID_COL
    if "cond" in stories.columns:
        story_df["cond"] = stories["cond"]
    unit_df = pd.DataFrame(unit_rows)

    gap = (story_df["s_resonance"]
           - (story_df["s_novelty"] - story_df["s_transience"])).abs().max()
    print(f"\nIdentity check  max |resonance - (novelty - transience)| = {gap:.2e}")

    # 4. Save & upload ------------------------------------------------------
    outdir = Path("out_artifacts")
    outdir.mkdir(parents=True, exist_ok=True)
    story_df.to_csv(outdir / "surprisal_story_level.csv")
    unit_df.to_csv(outdir / "surprisal_unit_level.csv", index=False)

    with open(outdir / "surprisal_run_metadata.json", "w") as f:
        json.dump({
            "step": "03_compute_surprisal_metrics_hf_job",
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "dtype": str(torch_dtype),
            "device": device,
            "max_context": int(max_ctx),
            "unit": UNIT,
            "window_words": WINDOW_WORDS if UNIT == "window" else None,
            "common_index_range": COMMON_INDEX_RANGE,
            "n_stories": int(len(story_df)),
            "n_units": int(len(unit_df)),
            "identity_gap": float(gap),
        }, f, indent=2, default=str)

    if "cond" in story_df.columns:
        print(story_df.groupby("cond")[["s_novelty", "s_transience", "s_resonance"]]
              .mean().round(3).to_string())

    print("\nUploading artifacts...")
    api.upload_folder(folder_path=str(outdir), repo_id=TARGET_HF_REPO,
                      repo_type="dataset", token=HF_TOKEN)
    print(f"✅ HF Job finished! Artifacts uploaded to {TARGET_HF_REPO}")


if __name__ == "__main__":
    main()
