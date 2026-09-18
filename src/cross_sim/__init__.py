"""
Cross-model AI-AI story simulation.

The EMNLP pipeline simulated the LLM-LLM condition with each model paired
against itself. This package generalises that to a full model x model grid,
so a story's two authors can be different models and "who starts" becomes an
experimental factor rather than a fixed property of the design.

Everything about a single exchange -- prompts, context handling per provider,
echo stripping, symmetric word truncation -- is carried over unchanged from
PenPal-EMNLP src/nes/simulation.py so the new stories stay comparable with
the existing 80 same-model ones.
"""

from .pairing import ModelPair, build_pair_grid, allocate_stories
from .prompts import SystemPrompts
from .providers import BaseProvider, get_provider
from .simulate import simulate_single_story, simulate_cross_model_dataset
from .story_table import build_story_table

__all__ = [
    "ModelPair",
    "build_pair_grid",
    "allocate_stories",
    "SystemPrompts",
    "BaseProvider",
    "get_provider",
    "simulate_single_story",
    "simulate_cross_model_dataset",
    "build_story_table",
]
