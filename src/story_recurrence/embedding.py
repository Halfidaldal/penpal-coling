"""
Sentence embedding with QZhou-Embedding (or any SentenceTransformer model).

QZhou-Embedding (Kingsoft-LLM/QZhou-Embedding) is a decoder-based embedding
model, so it needs ``padding_side="left"`` and ``trust_remote_code=True``. Both
are set here rather than inherited from the repo-wide helper in
``src/nes/embeddings.py``, which gates left-padding on an experiment name that
does not match this project's experiment labels.

All embeddings are L2-normalised at encode time. Every downstream measure
assumes unit-norm vectors, which makes the cosine similarity matrix a plain
Gram matrix (E @ E.T) and keeps the geometry measures interpretable.
"""

from typing import Dict, List, Optional, Tuple
import gc

import numpy as np
import torch
from tqdm import tqdm


DEFAULT_MODEL = "Kingsoft-LLM/QZhou-Embedding"

# Models that need left padding (decoder-based embedders).
_LEFT_PAD_HINTS = ("qzhou", "qwen", "e5-mistral", "gte-qwen", "sfr-embedding")


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available torch device, or the one requested."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _needs_left_padding(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(hint in lowered for hint in _LEFT_PAD_HINTS)


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: Optional[torch.device] = None,
    dtype: str = "auto",
):
    """
    Load a SentenceTransformer encoder.

    Args:
        model_name: HF model id.
        device: torch device; auto-detected when None.
        dtype: "auto" (bfloat16 on CUDA, float32 elsewhere), or an explicit
            torch dtype name such as "float16" / "bfloat16" / "float32".
    """
    from sentence_transformers import SentenceTransformer

    device = device or get_device()

    if dtype == "auto":
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    tokenizer_kwargs = {"trust_remote_code": True}
    if _needs_left_padding(model_name):
        tokenizer_kwargs["padding_side"] = "left"

    print(f"Loading embedding model: {model_name}  (device={device}, dtype={torch_dtype})")
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device=str(device),
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    model.eval()
    return model


def _encode(model, batch: List[str], instruction: Optional[str]) -> np.ndarray:
    """Encode one batch, passing an instruction/prompt only if the model takes one."""
    kwargs = dict(show_progress_bar=False, normalize_embeddings=True)
    if instruction:
        try:
            return model.encode(batch, prompt=instruction, **kwargs)
        except TypeError:
            pass
    return model.encode(batch, **kwargs)


def embed_sentences(
    story_sentences: Dict[str, List[str]],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    dtype: str = "auto",
    instruction: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Embed every sentence of every story.

    All sentences across the corpus are encoded in one flat pass (so batches are
    full even when individual stories are short), then split back per story.

    Args:
        story_sentences: {story_id: [sentence, ...]}.
        model_name: HF model id.
        batch_size: sentences per forward pass.
        instruction: optional prompt/instruction prefix. Leave None for
            symmetric within-document similarity, which is what we want here.

    Returns:
        ({story_id: array of shape (n_sentences, dim)}, embedding_dim)
    """
    device = device or get_device()
    model = load_model(model_name, device=device, dtype=dtype)

    # Flatten, remembering which story each sentence came from.
    ids: List[str] = []
    flat: List[str] = []
    for sid, sents in story_sentences.items():
        for s in sents:
            ids.append(sid)
            flat.append(s)

    if not flat:
        raise ValueError("No sentences to embed — check segmentation settings.")

    print(f"Embedding {len(flat)} sentences from {len(story_sentences)} stories "
          f"(batch_size={batch_size})...")

    chunks = []
    for i in tqdm(range(0, len(flat), batch_size), desc="Embedding"):
        chunks.append(_encode(model, flat[i:i + batch_size], instruction))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    matrix = np.vstack(chunks).astype(np.float32)

    # Guard against any model that ignores normalize_embeddings.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    dim = matrix.shape[1]
    print(f"Embedding matrix: {matrix.shape}")

    out: Dict[str, np.ndarray] = {}
    cursor = 0
    for sid, sents in story_sentences.items():
        n = len(sents)
        out[sid] = matrix[cursor:cursor + n]
        cursor += n

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out, dim
