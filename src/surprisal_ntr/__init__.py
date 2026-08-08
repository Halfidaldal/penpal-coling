"""
surprisal_ntr — stylometry and LM-surprisal Novelty / Transience / Resonance.

Modules:
  story_table  build the one-row-per-story table from the annotation file
  stylometry   transparent surface features (no LM required)
  lm           model loading and surprisal primitives
  ntr          window-based novelty / transience / resonance

Submodules are imported lazily so that ``story_table`` and ``stylometry`` can be
used without torch installed; only ``lm`` and ``ntr`` require it.
"""

import importlib
from typing import TYPE_CHECKING

__version__ = "0.2.0"

__all__ = ["story_table", "stylometry", "lm", "ntr"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)


if TYPE_CHECKING:
    from . import story_table, stylometry, lm, ntr
