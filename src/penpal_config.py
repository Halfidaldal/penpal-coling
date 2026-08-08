"""
Shared configuration loading.

Scripts stay thin: they read config.yaml from the repository root (or a path
given with --config) and hand sections to the functions in src/. Any package
under src/ can use this loader.
"""

from pathlib import Path
from typing import Any, Optional
import os

import yaml


def repo_root() -> Path:
    """Repository root, i.e. the parent of src/."""
    return Path(__file__).resolve().parent.parent


def load_config(path: Optional[str] = None) -> dict:
    """
    Load config.yaml.

    Args:
        path: explicit path; defaults to <repo_root>/config.yaml.
    """
    cfg_path = Path(path) if path else repo_root() / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Fetch a nested value with a dotted path, e.g. get(cfg, "surprisal.window_words").
    Returns `default` if any level is missing or None.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def resolve_path(cfg: dict, dotted_key: str, default: Optional[str] = None) -> Path:
    """
    Resolve a path from config against the repository root.

    Absolute paths in the config are respected as-is.
    """
    raw = get(cfg, dotted_key, default)
    if raw is None:
        raise KeyError(f"No path configured at '{dotted_key}'")
    p = Path(raw)
    return p if p.is_absolute() else (repo_root() / p)


def resolve_token(cfg: dict, dotted_key: str = "surprisal.hf_token",
                  env_var: str = "HF_TOKEN") -> Optional[str]:
    """
    Resolve a Hugging Face token, preferring the environment variable.

    Keeping the token in the environment rather than in config.yaml avoids
    committing a credential to the repository.
    """
    env = os.environ.get(env_var)
    if env:
        return env
    return get(cfg, dotted_key, None)
