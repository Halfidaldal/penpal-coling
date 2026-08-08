"""
Language-model loading and surprisal primitives.

Everything here is model-agnostic: any HuggingFace causal LM works. Because the
measures are model-relative, absolute magnitudes depend on the model chosen, so
only relative differences and correlations should be interpreted.
"""

from dataclasses import dataclass
from typing import Optional
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LMBundle:
    """A loaded model plus everything the surprisal functions need."""
    tokenizer: object
    model: object
    device: str
    bos_id: int
    max_context: int
    model_name: str
    dtype: str

    def encode(self, text: str) -> torch.Tensor:
        """Tokenise a window. The leading space keeps BPE consistent mid-text."""
        return self.tokenizer(
            " " + text.strip(), add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]


def resolve_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested and requested != "auto":
        return getattr(torch, requested)
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(cfg: dict, hf_token: Optional[str] = None) -> LMBundle:
    """
    Load tokenizer and model according to the ``surprisal`` config section.

    Args:
        cfg: the ``surprisal`` section of config.yaml.
        hf_token: optional HF access token (prefer the HF_TOKEN env var).
    """
    model_name = cfg["model_name"]
    device = resolve_device(cfg.get("device", "auto"))
    torch_dtype = resolve_dtype(cfg.get("dtype", "auto"), device)

    print(f"Loading LM '{model_name}' on {device} ({torch_dtype})")
    auth = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **auth)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            **auth,
        )
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, **auth
        ).to(device)

    if not hasattr(model, "hf_device_map") and device != "cpu":
        model = model.to(device)
    model.eval()

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError(f"{model_name} has neither a BOS nor an EOS token id.")

    max_context = cfg.get("max_context")
    if not max_context:
        max_context = getattr(
            model.config, "n_positions",
            getattr(model.config, "max_position_embeddings", 1024),
        )

    return LMBundle(
        tokenizer=tokenizer, model=model, device=device, bos_id=int(bos_id),
        max_context=int(max_context), model_name=model_name, dtype=str(torch_dtype),
    )


@torch.no_grad()
def token_surprisal(lm: LMBundle, ids: torch.Tensor) -> torch.Tensor:
    """Per-token surprisal in bits for a sequence of token ids."""
    ids_dev = ids.to(lm.device)
    logits = lm.model(ids_dev.unsqueeze(0)).logits[0]
    logp = torch.log_softmax(logits[:-1].float(), dim=-1)
    return -logp.gather(1, ids_dev[1:].unsqueeze(1)).squeeze(1) / math.log(2)


def mean_surprisal(
    lm: LMBundle,
    target: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> float:
    """
    Mean token surprisal (bits) of ``target``, optionally conditioned on
    ``context``. Context is truncated from the left to fit the model window.
    """
    prefix = torch.tensor([lm.bos_id], device=lm.device)
    if context is not None and len(context) > 0:
        budget = lm.max_context - len(target) - 1
        if budget > 0:
            prefix = torch.cat([prefix, context[-budget:].to(lm.device)])
    full = torch.cat([prefix, target.to(lm.device)])
    return float(token_surprisal(lm, full)[-len(target):].mean().item())
