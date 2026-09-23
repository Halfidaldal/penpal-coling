# /// script
# dependencies = [
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "scipy>=1.10",
#   "torch>=2.1",
#   "transformers>=4.40.0,<4.45.0",
#   "sentence-transformers>=3.0",
#   "accelerate>=0.30",
#   "spacy>=3.7",
#   "click",
#   "tqdm>=4.65",
#   "huggingface-hub>=0.23.0",
# ]
# ///

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# [Existing imports and config patch omitted in replacement chunk target]

"""
Hugging Face Job Script for computing QZhou sentence embeddings on managed GPU.

Runs remotely on HF Jobs (e.g. flavor="a10g-small").
Inputs:
  Reads input CSV from HF Dataset (TARGET_HF_REPO/input.csv).
Outputs:
  Uploads sentence_embeddings.npz, sentence_index.csv, stories.csv,
  and embedding_metadata.json back to TARGET_HF_REPO on Hugging Face.
"""

import os
import sys
import gc
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download

# Patch PretrainedConfig to provide a fallback for rope_theta if missing in transformers
from transformers import PretrainedConfig
_orig_getattr = PretrainedConfig.__getattribute__
def _patched_getattr(self, name):
    try:
        return _orig_getattr(self, name)
    except AttributeError as e:
        if name == "rope_theta":
            return getattr(self, "rope_scaling", {}).get("rope_theta", 1000000.0) if isinstance(getattr(self, "rope_scaling", None), dict) else 1000000.0
        raise e
PretrainedConfig.__getattribute__ = _patched_getattr

# Environment / Secret parameters
HF_TOKEN = os.environ.get("HF_TOKEN")
TARGET_HF_REPO = os.environ.get("TARGET_HF_REPO")
MODEL_NAME = os.environ.get("MODEL_NAME", "Kingsoft-LLM/QZhou-Embedding")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
ID_COL = os.environ.get("ID_COL", "id")
TEXT_COL = os.environ.get("TEXT_COL", "text")
CONDITION_COL = os.environ.get("CONDITION_COL", "condition")
SEGMENTER = os.environ.get("SEGMENTER", "spacy")
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_md")
MIN_WORDS = int(os.environ.get("MIN_WORDS", "3"))

CONDITION_MAP = {
    "hh": "HH", "ha": "H-LLM", "aa": "LLM-LLM", "aa_cross": "LLM-LLM-cross",
    "human-human": "HH", "human-ai": "H-LLM", "ai-ai": "LLM-LLM", "ai-ai-cross": "LLM-LLM-cross",
}

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def get_spacy_nlp(model_name: str):
    import spacy
    try:
        nlp = spacy.load(model_name, exclude=["ner", "lemmatizer", "textcat"])
    except OSError:
        print(f"Downloading spaCy model {model_name}...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name, exclude=["ner", "lemmatizer", "textcat"])
    nlp.max_length = 2_000_000
    return nlp


def split_sentences(text: str, segmenter: str, spacy_model: str, min_words: int) -> list:
    text = "" if text is None else str(text).strip()
    if not text:
        return []

    if segmenter == "spacy":
        try:
            nlp = get_spacy_nlp(spacy_model)
            sents = [s.text.strip() for s in nlp(text).sents]
        except Exception as exc:
            print(f"[WARNING] spaCy segmentation failed ({exc}); falling back to regex.")
            sents = _SENT_RE.split(text)
    else:
        sents = _SENT_RE.split(text)

    return [s.strip() for s in sents if len(s.split()) >= min_words]


def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN secret environment variable is missing.")
    if not TARGET_HF_REPO:
        raise ValueError("TARGET_HF_REPO environment variable is missing.")

    print(f"=== Hugging Face Embedding Job ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Target HF Dataset Repo: {TARGET_HF_REPO}")
    print(f"Device CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    api = HfApi(token=HF_TOKEN)

    # 1. Download input CSV from target repo
    print(f"Downloading input data from {TARGET_HF_REPO}...")
    input_file_path = hf_hub_download(
        repo_id=TARGET_HF_REPO,
        filename="input.csv",
        repo_type="dataset",
        token=HF_TOKEN,
    )

    df_raw = pd.read_csv(input_file_path)
    id_col = ID_COL if ID_COL in df_raw.columns else ("conversation_id" if "conversation_id" in df_raw.columns else "id")
    text_col = TEXT_COL if TEXT_COL in df_raw.columns else ("full_story" if "full_story" in df_raw.columns else "text")

    print(f"Loaded {len(df_raw)} rows (id_col='{id_col}', text_col='{text_col}').")

    # Rename & deduplicate
    rename_map = {id_col: "story_id", text_col: "text"}
    if CONDITION_COL and CONDITION_COL in df_raw.columns:
        rename_map[CONDITION_COL] = "cond"

    stories = df_raw.rename(columns=rename_map).groupby("story_id", as_index=True).first()
    if "cond" in stories.columns:
        stories["cond"] = (
            stories["cond"].astype(str).str.strip().str.lower()
            .map(CONDITION_MAP).fillna(stories["cond"])
        )
    stories["text"] = stories["text"].astype(str)

    # 2. Segment sentences
    print(f"Segmenting corpus with '{SEGMENTER}' (min {MIN_WORDS} words)...")
    story_sentences = {}
    for sid, text in stories["text"].to_dict().items():
        sents = split_sentences(text, segmenter=SEGMENTER, spacy_model=SPACY_MODEL, min_words=MIN_WORDS)
        if sents:
            story_sentences[sid] = sents

    counts = {sid: len(s) for sid, s in story_sentences.items()}
    n_sent = np.array(list(counts.values()))
    print(f"Sentences per story: mean {n_sent.mean():.1f}, total {int(n_sent.sum())}")

    # 3. Load AutoTokenizer and AutoModel
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading Transformers model: {MODEL_NAME} (device={device}, dtype={torch_dtype})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    except Exception as exc:
        print(f"[INFO] Fast tokenizer instantiation failed ({exc}); falling back to use_fast=False...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
    model.eval()

    # Flatten sentences
    flat_sentences = []
    sentence_map = []
    for sid, sents in story_sentences.items():
        for i, s in enumerate(sents):
            flat_sentences.append(s)
            sentence_map.append({"story_id": sid, "sent_idx": i, "n_words": len(s.split()), "sentence": s})

    print(f"Embedding {len(flat_sentences)} sentences (batch_size={BATCH_SIZE})...")
    chunks = []
    for i in tqdm(range(0, len(flat_sentences), BATCH_SIZE), desc="Embedding"):
        batch_sentences = flat_sentences[i : i + BATCH_SIZE]
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = mean_pooling(outputs, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            chunks.append(emb.cpu().numpy())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    matrix = np.vstack(chunks).astype(np.float32)

    # Unit-norm check
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    dim = matrix.shape[1]
    print(f"Final embedding matrix shape: {matrix.shape}")

    # Reconstruct per-story array dictionary
    embeddings = {}
    cursor = 0
    for sid, sents in story_sentences.items():
        n = len(sents)
        embeddings[str(sid)] = matrix[cursor : cursor + n]
        cursor += n

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 4. Save artifacts to temporary directory
    outdir = Path("out_artifacts")
    outdir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(outdir / "sentence_embeddings.npz", **embeddings)

    idx_df = pd.DataFrame(sentence_map)
    idx_df.to_csv(outdir / "sentence_index.csv", index=False)

    meta_df = stories.copy()
    meta_df["n_sentences"] = meta_df.index.map(counts).fillna(0).astype(int)
    meta_df["n_words"] = meta_df["text"].str.split().str.len()
    meta_df.drop(columns=["text"]).to_csv(outdir / "stories.csv")

    meta_dict = {
        "step": "01_compute_embeddings_hf_job",
        "timestamp": datetime.now().isoformat(),
        "n_stories": len(embeddings),
        "n_sentences": len(flat_sentences),
        "embedding_model": MODEL_NAME,
        "embedding_dim": dim,
        "dtype": str(torch_dtype),
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "segmenter": SEGMENTER,
        "spacy_model": SPACY_MODEL if SEGMENTER == "spacy" else None,
        "min_sentence_words": MIN_WORDS,
        "normalised": True,
    }
    with open(outdir / "embedding_metadata.json", "w") as f:
        json.dump(meta_dict, f, indent=2)

    print("Artifacts saved locally. Uploading to HF Hub Dataset...")

    # 5. Upload artifacts to HF Dataset repository
    api.upload_folder(
        folder_path=str(outdir),
        repo_id=TARGET_HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    print(f"✅ HF Job finished! All artifacts uploaded to {TARGET_HF_REPO}")


if __name__ == "__main__":
    main()
