"""
Story-recurrence: full-story structural measures from sentence embeddings.

Three families, all computed on the completed story with no turn structure:

  1. Recurrence Quantification Analysis  (rqa.py)          — does the story echo itself
  2. Trajectory geometry / embedding drift (trajectory.py) — what shape is the story
  3. Semantic decay rate (decay.py)                        — how fast does it forget

Pipeline: segmentation -> embedding -> matrices -> metrics -> export.

Submodules are imported lazily so that the metrics step (matrices, rqa,
trajectory, decay) runs in a numpy-only environment. Only `embedding` requires
torch / sentence-transformers.
"""

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["segmentation", "embedding", "matrices", "rqa",
           "trajectory", "decay", "io_utils"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)


if TYPE_CHECKING:  # for editors / type checkers only
    from . import segmentation, embedding, matrices, rqa, trajectory, decay, io_utils
