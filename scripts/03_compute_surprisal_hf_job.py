# /// script
# dependencies = [
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "scipy>=1.10",
#   "torch>=2.1",
#   "transformers>=4.40.0",
#   "accelerate>=0.30",
#   "tqdm>=4.65",
#   "huggingface-hub>=0.23.0",
#   "scikit-learn>=1.0",
# ]
# ///

"""
Hugging Face Job Script for computing LM Surprisal & NTR on managed GPU.

Runs remotely on HF Jobs (e.g., flavor="a10g-small" or "a100-large").
Inputs:
  Reads input CSV from HF Dataset (TARGET_HF_REPO/input.csv).
Outputs:
  Uploads penpal_measures_all.csv, surprisal_window_level.csv,
  and surprisal_run_metadata.json back to TARGET_HF_REPO on Hugging Face.
"""

import gc
import json
import math
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Environment / Secret parameters
HF_TOKEN = os.environ.get("HF_TOKEN")
TARGET_HF_REPO = os.environ.get("TARGET_HF_REPO")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-31b")
WINDOW_WORDS = int(os.environ.get("WINDOW_WORDS", "14"))
MIN_WINDOW_WORDS = int(os.environ.get("MIN_WINDOW_WORDS", "3"))
STYLOMETRY_ENABLED = os.environ.get("STYLOMETRY_ENABLED", "true").lower() in ("true", "1", "yes")
COMMON_INDEX_RANGE = os.environ.get("COMMON_INDEX_RANGE", "true").lower() in ("true", "1", "yes")
SAVE_RAW_TERMS = os.environ.get("SAVE_RAW_TERMS", "true").lower() in ("true", "1", "yes")
ID_COL = os.environ.get("ID_COL", "id")
TEXT_COL = os.environ.get("TEXT_COL", "text")
CONDITION_COL = os.environ.get("CONDITION_COL", "condition")

DIMENSIONS = {
    "consistency": "coherence_element_consistency",
    "coherence": "coherence_logical_progression",
    "originality": "creativity_originality",
    "surprisingness": "creativity_surprisingness",
    "enjoyment": "likeability_enjoyability",
    "quality": "likeability_quality",
}

COMPOSITES = {
    "coherence_comp": ["consistency", "coherence"],
    "creativity_comp": ["originality", "surprisingness"],
    "attract_comp": ["enjoyment", "quality"],
}

CONDITION_MAP = {
    "hh": "HH",
    "ha": "H-LLM",
    "aa": "LLM-LLM",
    "human-human": "HH",
    "human-ai": "H-LLM",
    "ai-ai": "LLM-LLM",
}

LLMNESS = {"HH": 0, "H-LLM": 1, "LLM-LLM": 2}

WORD_REGEX = re.compile(r"[A-Za-z']+")
SENT_REGEX = re.compile(r"(?<=[.!?])\s+")


# --- Story Table Builder ---

def detect_layout(df: pd.DataFrame, dimensions: Dict[str, str]) -> str:
    stems = list(dimensions.values())
    if any(f"ann1_{s}" in df.columns or f"mean_{s}" in df.columns for s in stems):
        return "wide"
    if any(s in df.columns for s in stems):
        return "long"
    return "unknown"


def _ratings_wide(df: pd.DataFrame, dimensions: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, stem in dimensions.items():
        mean_col = f"mean_{stem}"
        if mean_col in df.columns:
            out[name] = pd.to_numeric(df[mean_col], errors="coerce")
        else:
            cols = [c for c in (f"ann1_{stem}", f"ann2_{stem}") if c in df.columns]
            if cols:
                vals = df[cols].apply(pd.to_numeric, errors="coerce")
                out[name] = vals.mean(axis=1)
            else:
                out[name] = np.nan
    return out


def _n_annotators_wide(df: pd.DataFrame, dimensions: Dict[str, str]) -> pd.Series:
    if "n_annotators" in df.columns:
        return pd.to_numeric(df["n_annotators"], errors="coerce").fillna(0).astype(int)
    stems = list(dimensions.values())
    n = pd.Series(0, index=df.index, dtype=int)
    for prefix in ("ann1_", "ann2_"):
        cols = [f"{prefix}{s}" for s in stems if f"{prefix}{s}" in df.columns]
        if cols:
            n += df[cols].notna().any(axis=1).astype(int)
    return n


def build_story_table(annotations: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    id_col = ID_COL if ID_COL in annotations.columns else ("conversation_id" if "conversation_id" in annotations.columns else "id")
    text_col = TEXT_COL if TEXT_COL in annotations.columns else ("full_story" if "full_story" in annotations.columns else "text")
    cond_col = CONDITION_COL if CONDITION_COL in annotations.columns else "condition"
    dims = list(DIMENSIONS.keys())

    for required in (id_col, text_col):
        if required not in annotations.columns:
            raise ValueError(
                f"Annotation file is missing '{required}'. "
                f"Available columns: {list(annotations.columns)}"
            )

    layout = detect_layout(annotations, DIMENSIONS)
    df = annotations.copy()

    if layout == "wide":
        df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        story = pd.DataFrame(index=df[id_col])
        story.index.name = "id"
        story["text"] = df[text_col].astype(str).values
        if cond_col in df.columns:
            story["cond"] = df[cond_col].astype(str).str.strip().str.lower().values
        story["n_annot"] = _n_annotators_wide(df, DIMENSIONS).values
        ratings = _ratings_wide(df, DIMENSIONS)
        for d in dims:
            story[d] = ratings[d].values
    elif layout == "long":
        for d, stem in DIMENSIONS.items():
            if stem in df.columns:
                df[d] = pd.to_numeric(df[stem], errors="coerce")
        agg = {"text": (text_col, "first")}
        if cond_col in df.columns:
            agg["cond"] = (cond_col, "first")
        if "annotator" in df.columns:
            agg["n_annot"] = ("annotator", "nunique")
        for d in dims:
            if d in df.columns:
                agg[d] = (d, "mean")
        story = df.groupby(id_col).agg(**agg)
        story.index.name = "id"
        if "cond" in story.columns:
            story["cond"] = story["cond"].astype(str).str.strip().str.lower()
        if "n_annot" not in story.columns:
            story["n_annot"] = story[dims].notna().any(axis=1).astype(int)
    else:
        # Minimal fallback for story text without rating columns
        story = pd.DataFrame(index=df[id_col])
        story.index.name = "id"
        story["text"] = df[text_col].astype(str).values
        if cond_col in df.columns:
            story["cond"] = df[cond_col].astype(str).str.strip().str.lower().values
        story["n_annot"] = 0

    if "cond" in story.columns:
        story["cond"] = story["cond"].map(CONDITION_MAP).fillna(story["cond"])
        story["llm"] = story["cond"].map(LLMNESS)

    for comp_name, members in COMPOSITES.items():
        present = [m for m in members if m in story.columns]
        if len(present) == len(members):
            story[comp_name] = story[present].mean(axis=1)

    present_dims = [d for d in dims if d in story.columns]
    if present_dims:
        story["overall"] = story[present_dims].mean(axis=1)

    front = [c for c in ["cond", "llm", "n_annot", "text"] if c in story.columns]
    rest = [c for c in story.columns if c not in front]
    story = story[front + rest]

    return story, layout


# --- Stylometry ---

def count_syllables(word: str) -> int:
    w = word.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    return max(1, n - (1 if w.endswith("e") else 0))


def story_features(text: str) -> Dict[str, float]:
    words = WORD_REGEX.findall(str(text))
    sentences = [s for s in SENT_REGEX.split(str(text).strip()) if s.split()]

    n_words = len(words)
    n_sents = max(1, len(sentences))
    words_low = [w.lower() for w in words]
    vocab = set(words_low)
    freq = Counter(words_low)
    hapax_count = sum(1 for w, c in freq.items() if c == 1)

    out = {
        "n_words": float(n_words),
        "n_sentences": float(len(sentences)),
        "mean_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mean_sentence_length": n_words / n_sents,
        "ttr": len(vocab) / n_words if n_words else 0.0,
        "hapax_ratio": hapax_count / n_words if n_words else 0.0,
        "comma_density": (str(text).count(",") / n_words) * 1000.0 if n_words else 0.0,
        "flesch_approx": (
            206.835
            - 1.015 * (n_words / n_sents)
            - 84.6 * (sum(count_syllables(w) for w in words) / n_words if n_words else 0.0)
        ),
    }

    try:
        if len(sentences) >= 2:
            sim = cosine_similarity(TfidfVectorizer().fit_transform(sentences))
            out["adjacent_sentence_similarity"] = float(
                np.mean([sim[i, i + 1] for i in range(len(sentences) - 1)])
            )
        else:
            out["adjacent_sentence_similarity"] = np.nan
    except Exception:
        out["adjacent_sentence_similarity"] = np.nan

    return out


def semantic_distinctiveness(texts: List[str], ngram_range=(1, 2), min_df: int = 2) -> np.ndarray:
    try:
        matrix = TfidfVectorizer(ngram_range=tuple(ngram_range), min_df=min_df).fit_transform(list(texts))
        sim = cosine_similarity(matrix)
        np.fill_diagonal(sim, 0.0)
        return 1.0 - sim.max(axis=1)
    except Exception:
        return np.full(len(texts), np.nan)


def compute_stylometry(story: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame({i: story_features(t) for i, t in story["text"].items()}).T
    feats.index.name = story.index.name
    feats["semantic_distinctiveness"] = semantic_distinctiveness(story["text"].tolist())
    return feats


# --- LM & Surprisal NTR ---

@dataclass
class LMBundle:
    tokenizer: object
    model: object
    device: str
    bos_id: int
    max_context: int
    model_name: str
    dtype: str

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(" " + text.strip(), add_special_tokens=False, return_tensors="pt").input_ids[0]


def load_lm_model(model_name: str, hf_token: Optional[str] = None) -> LMBundle:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading Causal LM '{model_name}' on {device} ({torch_dtype})...")
    auth = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **auth)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **auth,
        )
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=True, **auth
        ).to(device)

    if not hasattr(model, "hf_device_map") and device != "cpu":
        model = model.to(device)
    model.eval()

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = 1  # Fallback

    max_context = getattr(
        model.config, "n_positions",
        getattr(model.config, "max_position_embeddings", 1024),
    )

    return LMBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        bos_id=int(bos_id),
        max_context=int(max_context),
        model_name=model_name,
        dtype=str(torch_dtype),
    )


@torch.no_grad()
def token_surprisal(lm: LMBundle, ids: torch.Tensor) -> torch.Tensor:
    ids_dev = ids.to(lm.device)
    logits = lm.model(ids_dev.unsqueeze(0)).logits[0]
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    return -logp.gather(1, ids_dev[1:].unsqueeze(1)).squeeze(1) / math.log(2)


def mean_surprisal(lm: LMBundle, target: torch.Tensor, context: Optional[torch.Tensor] = None) -> float:
    prefix = torch.tensor([lm.bos_id], device=lm.device)
    if context is not None and len(context) > 0:
        budget = lm.max_context - len(target) - 1
        if budget > 0:
            prefix = torch.cat([prefix, context[-budget:].to(lm.device)])
    full = torch.cat([prefix, target.to(lm.device)])
    return float(token_surprisal(lm, full)[-len(target):].mean().item())


def make_windows(text: str, window_words: int, min_window_words: int = 3) -> List[str]:
    words = str(text).split()
    return [
        " ".join(words[i : i + window_words])
        for i in range(0, len(words), window_words)
        if len(words[i : i + window_words]) >= min_window_words
    ]


def compute_story_ntr(
    text: str,
    lm: LMBundle,
    window_words: int,
    min_window_words: int = 3,
    common_index_range: bool = True,
    save_raw_terms: bool = True,
) -> Tuple[Dict[str, float], List[dict]]:
    units = make_windows(text, window_words, min_window_words)
    n = len(units)

    nan_summary = {
        "s_novelty": np.nan, "s_transience": np.nan, "s_resonance": np.nan,
        "n_windows_total": n, "n_windows_used": 0,
    }
    if n < 2:
        return nan_summary, []

    ids = [lm.encode(u) for u in units]

    base = [mean_surprisal(lm, ids[i]) for i in range(n)]
    s_ctx = [base[0]] + [
        mean_surprisal(lm, ids[i], torch.cat(ids[:i])) for i in range(1, n)
    ]

    nov = np.full(n, np.nan)
    tra = np.full(n, np.nan)

    for i in range(1, n):
        nov[i] = s_ctx[i] - base[i]

    for i in range(n - 1):
        ctx_before = torch.cat(ids[:i]) if i > 0 else None
        tra[i] = s_ctx[i + 1] - mean_surprisal(lm, ids[i + 1], ctx_before)

    res = nov - tra

    records = []
    for i in range(n):
        rec = {
            "window_idx": i,
            "n_window_words": len(units[i].split()),
            "s_novelty": float(nov[i]),
            "s_transience": float(tra[i]),
            "s_resonance": float(res[i]),
            "window_text": units[i],
        }
        if save_raw_terms:
            rec["s_base"] = float(base[i])
            rec["s_ctx"] = float(s_ctx[i])
        records.append(rec)

    if common_index_range:
        lo, hi = 1, n - 1
    else:
        lo, hi = 0, n

    if hi <= lo:
        summary = dict(nan_summary)
    else:
        summary = {
            "s_novelty": float(np.nanmean(nov[lo:hi])),
            "s_transience": float(np.nanmean(tra[lo:hi])),
            "s_resonance": float(np.nanmean(res[lo:hi])),
            "n_windows_total": n,
            "n_windows_used": int(hi - lo),
        }
    return summary, records


def compute_corpus_ntr(story: pd.DataFrame, lm: LMBundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summaries, windows = {}, []
    for sid, text in tqdm(story["text"].items(), total=len(story), desc=f"Surprisal ({lm.model_name})"):
        summary, records = compute_story_ntr(
            text=text,
            lm=lm,
            window_words=WINDOW_WORDS,
            min_window_words=MIN_WINDOW_WORDS,
            common_index_range=COMMON_INDEX_RANGE,
            save_raw_terms=SAVE_RAW_TERMS,
        )
        summaries[sid] = summary
        for rec in records:
            rec["id"] = sid
            if "cond" in story.columns:
                rec["cond"] = story.loc[sid, "cond"]
            windows.append(rec)

    summary_df = pd.DataFrame(summaries).T
    window_df = pd.DataFrame(windows)
    return summary_df, window_df


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN secret environment variable is missing.")
    if not TARGET_HF_REPO:
        raise ValueError("TARGET_HF_REPO environment variable is missing.")

    print(f"=== Hugging Face LM Surprisal & NTR Job ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Target HF Dataset Repo: {TARGET_HF_REPO}")
    print(f"Window Words: {WINDOW_WORDS}")
    print(f"Device CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    api = HfApi(token=HF_TOKEN)

    # 1. Download input CSV from dataset repo
    print(f"Downloading input data from {TARGET_HF_REPO}...")
    input_file_path = hf_hub_download(
        repo_id=TARGET_HF_REPO,
        filename="input.csv",
        repo_type="dataset",
        token=HF_TOKEN,
    )

    df_raw = pd.read_csv(input_file_path)
    story, layout = build_story_table(df_raw)
    print(f"Detected annotation layout: {layout}")
    print(f"Loaded {len(story)} stories for surprisal analysis.")

    # 2. Stylometry
    if STYLOMETRY_ENABLED:
        print("\n--- Computing Stylometric features ---")
        style_df = compute_stylometry(story)
        story = story.join(style_df)

    # 3. LM Surprisal & NTR
    print("\n--- Computing LM Surprisal & NTR ---")
    lm_bundle = load_lm_model(MODEL_NAME, hf_token=HF_TOKEN)
    summary_df, window_df = compute_corpus_ntr(story, lm_bundle)
    story = story.join(summary_df)

    # Identity gap check
    if "s_resonance" in story.columns and "s_novelty" in story.columns and "s_transience" in story.columns:
        gap = (story["s_resonance"] - (story["s_novelty"] - story["s_transience"])).abs().max(skipna=True)
        print(f"\nIdentity check max |resonance - (novelty - transience)| = {gap:.2e}")

    # 4. Save artifacts
    outdir = Path("out_artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    story_export = story.drop(columns=["text"], errors="ignore")
    story_out_path = outdir / "penpal_measures_all.csv"
    story_export.to_csv(story_out_path)
    print(f"\nSaved story metrics -> {story_out_path} ({story_export.shape[0]} stories x {story_export.shape[1]} cols)")

    if not window_df.empty:
        win_out_path = outdir / "surprisal_window_level.csv"
        window_df.to_csv(win_out_path, index=False)
        print(f"Saved window metrics -> {win_out_path} ({len(window_df)} windows)")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "step": "03_compute_surprisal_hf_job",
        "n_stories": len(story),
        "n_rated_stories": int((story["n_annot"] > 0).sum()) if "n_annot" in story else None,
        "condition_counts": story["cond"].value_counts().to_dict() if "cond" in story else None,
        "stylometry_enabled": STYLOMETRY_ENABLED,
        "surprisal": {
            "model_name": MODEL_NAME,
            "window_words": WINDOW_WORDS,
            "min_window_words": MIN_WINDOW_WORDS,
            "common_index_range": COMMON_INDEX_RANGE,
            "save_raw_terms": SAVE_RAW_TERMS,
        },
        "model_dtype": lm_bundle.dtype,
        "model_device": lm_bundle.device,
        "max_context": lm_bundle.max_context,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__},
    }
    meta_out_path = outdir / "surprisal_run_metadata.json"
    with open(meta_out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved run metadata -> {meta_out_path}")

    # 5. Upload back to HF Repo
    print(f"\nUploading generated artifacts back to {TARGET_HF_REPO}...")
    for artifact_file in [story_out_path, win_out_path, meta_out_path]:
        if artifact_file.exists():
            print(f" Uploading {artifact_file.name}...")
            api.upload_file(
                path_or_fileobj=str(artifact_file),
                path_in_repo=artifact_file.name,
                repo_id=TARGET_HF_REPO,
                repo_type="dataset",
                token=HF_TOKEN,
            )

    print(f"\n✅ Remote HF Surprisal Job completed successfully!")


if __name__ == "__main__":
    main()
