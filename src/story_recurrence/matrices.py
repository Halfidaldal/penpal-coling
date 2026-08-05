"""
Similarity, distance and recurrence matrices for a single story.

Given L2-normalised sentence embeddings E (n x d) for one story:

    S = E @ E.T          cosine similarity, in [-1, 1]
    D = 1 - S            cosine distance,   in [0, 2]
    R = (D <= epsilon)   binary recurrence matrix

Thresholding is the one methodological choice that drives every RQA scalar, so
both standard strategies are supported:

  fixed_rr   (default, recommended for cross-condition comparison)
      epsilon is set per story as the q-th percentile of that story's valid
      distances, so the recurrence rate is constant by construction. Structural
      measures (DET, LAM, L, ENTR) then describe how recurrences are *organised*
      rather than how many there are. This matters because conditions differ in
      baseline self-similarity: with a fixed epsilon, a globally more repetitive
      condition scores higher on every RQA measure at once, which is the same
      confound that made topic-novelty uninterpretable.

  fixed_eps
      one epsilon for the whole corpus. Here RR itself becomes an informative
      variable ("how self-similar is this story at all"), but the structure
      measures are no longer independent of it.

Both are computed by the pipeline and exported side by side.

The Theiler window excludes the neighbourhood of the main diagonal. Points with
|i - j| <= theiler are never counted as recurrences and never counted in the
denominator. theiler=0 removes only the trivial self-match (i == j); theiler=1
also removes adjacent sentences, whose similarity is largely a local-cohesion
effect already captured by other measures.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def cosine_matrices(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity and cosine distance matrices for one story.

    Args:
        embeddings: (n, d) array of L2-normalised sentence embeddings.

    Returns:
        (similarity, distance), each (n, n) float32.
    """
    E = np.asarray(embeddings, dtype=np.float32)
    S = E @ E.T
    np.clip(S, -1.0, 1.0, out=S)          # guard against fp drift
    D = (1.0 - S).astype(np.float32)
    return S, D


def valid_mask(n: int, theiler: int = 0) -> np.ndarray:
    """
    Boolean mask of matrix cells eligible to be recurrences.

    Excludes the Theiler band |i - j| <= theiler.
    """
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return np.abs(i - j) > theiler


def recurrence_matrix(
    distance: np.ndarray,
    theiler: int = 0,
    mode: str = "fixed_rr",
    target_rr: float = 0.05,
    epsilon: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Binarise a distance matrix into a recurrence matrix.

    Args:
        distance: (n, n) cosine distance matrix.
        theiler: Theiler window; cells with |i-j| <= theiler are excluded.
        mode: "fixed_rr" (per-story epsilon giving target_rr) or "fixed_eps".
        target_rr: desired recurrence rate when mode == "fixed_rr", e.g. 0.05.
        epsilon: distance threshold when mode == "fixed_eps".

    Returns:
        (recurrence_matrix as bool array, epsilon actually used)
        Cells outside the Theiler band are always False.
    """
    D = np.asarray(distance, dtype=np.float32)
    n = D.shape[0]
    mask = valid_mask(n, theiler)

    if not mask.any():
        return np.zeros_like(D, dtype=bool), float("nan")

    if mode == "fixed_rr":
        # epsilon = the target_rr quantile of this story's valid distances.
        eps = float(np.quantile(D[mask], target_rr))
    elif mode == "fixed_eps":
        if epsilon is None:
            raise ValueError("mode='fixed_eps' requires an epsilon value.")
        eps = float(epsilon)
    else:
        raise ValueError(f"Unknown thresholding mode: {mode}")

    R = (D <= eps) & mask
    return R, eps


def corpus_epsilon(
    distances: Dict[str, np.ndarray],
    theiler: int = 0,
    target_rr: float = 0.05,
) -> float:
    """
    One corpus-wide epsilon, for use with mode="fixed_eps".

    Pools the valid distances of every story and takes the target_rr quantile,
    so the *corpus* average recurrence rate lands near target_rr while
    individual stories are free to vary. This is what makes RR informative.
    """
    pooled = []
    for D in distances.values():
        n = D.shape[0]
        if n < 2:
            continue
        pooled.append(D[valid_mask(n, theiler)])
    if not pooled:
        return float("nan")
    return float(np.quantile(np.concatenate(pooled), target_rr))
