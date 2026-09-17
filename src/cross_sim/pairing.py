"""
The model x model design grid.

With M models there are M^2 ordered (starter, responder) cells: M self-pairs on
the diagonal and M*(M-1) cross-model cells off it. For the four models used in
PenPal that is 16 cells -- 4 self, 12 cross.

Order matters off the diagonal. gpt-4.1 opening to claude and claude opening to
gpt-4.1 are different cells, because the opener writes into an empty story while
the responder always writes into someone else's prose. On the diagonal the
distinction is vacuous (both sides are the same model), but the cell is kept so
the grid stays a clean M x M factorial and the same-model condition is
regenerated under identical conditions rather than borrowed from EMNLP.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal

PairingMode = Literal["all", "cross", "same"]


@dataclass(frozen=True)
class ModelPair:
    """One ordered cell of the grid."""

    starter: str  # model id that writes turn 1
    responder: str  # model id that replies

    @property
    def is_self_pair(self) -> bool:
        return self.starter == self.responder

    @property
    def pair_id(self) -> str:
        """Order-sensitive label, e.g. 'gpt-4.1>claude-sonnet-4-5'."""
        return f"{self.starter}>{self.responder}"

    @property
    def dyad_id(self) -> str:
        """Order-insensitive label, so the two directions of a cross pair share it."""
        return "+".join(sorted({self.starter, self.responder}))


def build_pair_grid(model_ids: List[str], mode: PairingMode = "all") -> List[ModelPair]:
    """
    Enumerate the ordered cells of the grid.

    Args:
        model_ids: model ids, in config order.
        mode: 'all' for the full M x M grid, 'cross' for off-diagonal cells only,
            'same' for the diagonal only (the EMNLP design).

    Returns:
        Cells in a stable order: for 'all', row-major over model_ids.
    """
    if mode not in ("all", "cross", "same"):
        raise ValueError(f"Unknown pairing mode: {mode!r}")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError(f"Duplicate model ids: {model_ids}")

    pairs = []
    for starter in model_ids:
        for responder in model_ids:
            if mode == "cross" and starter == responder:
                continue
            if mode == "same" and starter != responder:
                continue
            pairs.append(ModelPair(starter=starter, responder=responder))
    return pairs


def allocate_stories(
    pairs: List[ModelPair],
    stories_per_pair: int = 7,
    n_stories: int = None,
) -> Dict[ModelPair, int]:
    """
    Decide how many stories each cell gets.

    By default every cell gets `stories_per_pair`, which keeps the design
    balanced. Passing `n_stories` instead targets a total: cells get
    n_stories // len(pairs) each and the remainder is spread over the first
    `n_stories % len(pairs)` cells, so the allocation is deterministic but no
    longer balanced.

    Returns:
        Cell -> story count, in the order `pairs` was given.
    """
    if not pairs:
        raise ValueError("No model pairs to allocate stories to")

    if n_stories is None:
        if stories_per_pair < 1:
            raise ValueError("stories_per_pair must be >= 1")
        return {pair: stories_per_pair for pair in pairs}

    if n_stories < len(pairs):
        raise ValueError(
            f"n_stories={n_stories} is fewer than the {len(pairs)} cells in the grid"
        )
    base, remainder = divmod(n_stories, len(pairs))
    return {
        pair: base + (1 if i < remainder else 0)
        for i, pair in enumerate(pairs)
    }


def describe_grid(allocation: Dict[ModelPair, int]) -> str:
    """One-line-per-cell summary for the script's stdout."""
    lines = []
    for pair, n in allocation.items():
        kind = "self " if pair.is_self_pair else "cross"
        lines.append(f"  [{kind}] {pair.starter:>28s} -> {pair.responder:<28s} {n:3d} stories")
    total = sum(allocation.values())
    n_self = sum(n for p, n in allocation.items() if p.is_self_pair)
    lines.append(f"  {len(allocation)} cells, {total} stories ({n_self} self-pair, {total - n_self} cross-model)")
    return "\n".join(lines)
