#!/usr/bin/env python
"""
Step 01: segment stories into sentences and embed them.

This is the only GPU-bound, slow step. It is deliberately separate from step 02
so that thresholds, Theiler windows and fit settings can be re-tuned without
re-embedding anything.

Outputs (into --outdir):
    sentence_embeddings.npz   per-story arrays, shape (n_sentences, dim)
    sentence_index.csv        every sentence with its story id and position
    stories.csv               one row per story: id, condition, counts
    embedding_metadata.json   model, dtype, parameters, versions

Usage (annotated subset, 147 stories):
    python scripts/01_compute_embeddings.py \
        --input data/interim/annotations/penpal_annotations_final.csv \
        --outdir data/interim/embeddings \
        --batch-size 16

Usage (full corpus, 216 stories -- run scripts/00_build_story_table.py first):
    python scripts/01_compute_embeddings.py \
        --input data/interim/stories/full_stories_all.csv \
        --id-col conversation_id --text-col full_story \
        --outdir data/interim/embeddings \
        --batch-size 16
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from story_recurrence.segmentation import segment_corpus
from story_recurrence.embedding import embed_sentences, DEFAULT_MODEL, get_device
from story_recurrence.io_utils import (
    load_stories, save_embeddings, save_metadata, sentence_index,
)


def parse_args():
    p = argparse.ArgumentParser(description="Segment and embed stories.")
    p.add_argument("--input", default="data/interim/annotations/penpal_annotations_final.csv", help="CSV with story id, text, condition")
    p.add_argument("--outdir", default="data/interim/embeddings", help="output directory")
    p.add_argument("--id-col", default="id")
    p.add_argument("--text-col", default="text")
    p.add_argument("--condition-col", default="condition")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF embedding model id")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device", default=None, help="cuda / cpu / mps (auto if unset)")
    p.add_argument("--segmenter", default="spacy", choices=["spacy", "regex"])
    p.add_argument("--spacy-model", default="en_core_web_md")
    p.add_argument("--min-sentence-words", type=int, default=3,
                   help="drop sentences shorter than this before embedding")
    p.add_argument("--instruction", default=None,
                   help="optional prompt/instruction for the encoder; leave unset "
                        "for symmetric within-document similarity")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stories = load_stories(
        args.input, id_col=args.id_col, text_col=args.text_col,
        condition_col=args.condition_col,
    )

    print(f"\nSegmenting with '{args.segmenter}' "
          f"(min {args.min_sentence_words} words per sentence)...")
    story_sentences = segment_corpus(
        stories["text"].to_dict(),
        segmenter=args.segmenter,
        spacy_model=args.spacy_model,
        min_words=args.min_sentence_words,
    )

    counts = {sid: len(s) for sid, s in story_sentences.items()}
    n_sent = np.array(list(counts.values()))
    print(f"Sentences per story: mean {n_sent.mean():.1f}, "
          f"min {n_sent.min()}, max {n_sent.max()}, total {int(n_sent.sum())}")

    too_short = [sid for sid, c in counts.items() if c < 3]
    if too_short:
        print(f"[WARNING] {len(too_short)} story/ies have <3 sentences and will "
              f"yield NaN metrics: {too_short[:10]}")

    # Drop empty stories so the encoder never sees a zero-length input.
    story_sentences = {k: v for k, v in story_sentences.items() if v}

    device = get_device(args.device)
    embeddings, dim = embed_sentences(
        story_sentences,
        model_name=args.model,
        batch_size=args.batch_size,
        device=device,
        dtype=args.dtype,
        instruction=args.instruction,
    )

    save_embeddings(embeddings, outdir / "sentence_embeddings.npz")

    idx = sentence_index(story_sentences)
    idx.to_csv(outdir / "sentence_index.csv", index=False)
    print(f"Saved sentence index -> {outdir / 'sentence_index.csv'}")

    meta_df = stories.copy()
    meta_df["n_sentences"] = meta_df.index.map(counts).fillna(0).astype(int)
    meta_df["n_words"] = meta_df["text"].str.split().str.len()
    meta_df.drop(columns=["text"]).to_csv(outdir / "stories.csv")
    print(f"Saved story table -> {outdir / 'stories.csv'}")

    import torch, sentence_transformers
    save_metadata(
        {
            "step": "01_compute_embeddings",
            "timestamp": datetime.now().isoformat(),
            "input": str(Path(args.input).resolve()),
            "n_stories": len(embeddings),
            "n_sentences": int(sum(len(v) for v in story_sentences.values())),
            "embedding_model": args.model,
            "embedding_dim": dim,
            "dtype": args.dtype,
            "device": str(device),
            "batch_size": args.batch_size,
            "instruction": args.instruction,
            "segmenter": args.segmenter,
            "spacy_model": args.spacy_model if args.segmenter == "spacy" else None,
            "min_sentence_words": args.min_sentence_words,
            "normalised": True,
            "versions": {
                "torch": torch.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        outdir / "embedding_metadata.json",
    )

    print("\n✅ Step 01 complete.")


if __name__ == "__main__":
    main()
