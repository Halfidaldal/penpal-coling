"""
endpoint_predictability — how the surprisal of a story's ending falls as the
story accumulates.

A global counterpart to the local window measures in ``surprisal_ntr``: every
point conditions on all preceding text against one fixed target, so this asks
whether a story builds toward its ending or arrives at it abruptly.

Reuses ``surprisal_ntr.lm`` for model loading and mean token surprisal, so both
families are expressed in the same bits and are directly comparable.
"""

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["endpoint"]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)


if TYPE_CHECKING:
    from . import endpoint
