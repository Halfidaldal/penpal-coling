"""
Recurrence Quantification Analysis (RQA) scalars for a story's recurrence matrix.

Standard RQA measures, with the narrative reading of each:

  RR    recurrence rate        how often the story revisits earlier semantic states
  DET   determinism            share of recurrences lying on diagonal lines, i.e.
                               the story re-runs *sequences* of material in order
                               (an echoed passage), not just isolated matches
  L     mean diagonal length   typical length of an echoed run, in sentences
  L_MAX longest diagonal       the single longest echoed run
  DIV   divergence (1 / L_MAX) inverse of the above
  ENTR  diagonal-length entropy variety of echo lengths; low = all echoes alike
  RATIO DET / RR               structure per unit of recurrence
  LAM   laminarity             share of recurrences on vertical lines, i.e. the
                               story *dwells* — one sentence matches a stretch
                               of consecutive others
  TT    trapping time          mean vertical line length; how long it dwells
  V_MAX longest vertical line

Diagonal lines capture ordered repetition (A-B-C returns as A-B-C). Vertical
lines capture stalling (one state persists while the story moves). For prose,
DET is the closest thing to "this story pays off earlier material", and LAM/TT
index thematic stasis.

Because the recurrence matrix is symmetric, every diagonal line off the main
diagonal appears twice; this is standard in RQA and affects all stories equally.
"""

from typing import Dict

import numpy as np

from .matrices import valid_mask


def _line_lengths(binary_1d: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs in a 1-D boolean array."""
    if binary_1d.size == 0 or not binary_1d.any():
        return np.empty(0, dtype=int)
    # Pad with False so runs at the edges are closed off.
    padded = np.concatenate(([False], binary_1d.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return ends - starts


def _diagonal_lengths(R: np.ndarray, theiler: int) -> np.ndarray:
    """All diagonal line lengths, skipping the excluded Theiler band."""
    n = R.shape[0]
    out = []
    for k in range(-(n - 1), n):
        if abs(k) <= theiler:      # main diagonal / Theiler band: not a line
            continue
        d = np.diagonal(R, offset=k)
        lengths = _line_lengths(d)
        if lengths.size:
            out.append(lengths)
    return np.concatenate(out) if out else np.empty(0, dtype=int)


def _vertical_lengths(R: np.ndarray) -> np.ndarray:
    """All vertical line lengths (columns of the recurrence matrix)."""
    out = []
    for j in range(R.shape[1]):
        lengths = _line_lengths(R[:, j])
        if lengths.size:
            out.append(lengths)
    return np.concatenate(out) if out else np.empty(0, dtype=int)


def _entropy(lengths: np.ndarray, l_min: int) -> float:
    """Shannon entropy (bits) of the distribution of line lengths >= l_min."""
    sel = lengths[lengths >= l_min]
    if sel.size == 0:
        return float("nan")
    counts = np.bincount(sel)[l_min:]
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def rqa_metrics(
    R: np.ndarray,
    theiler: int = 0,
    l_min: int = 2,
    v_min: int = 2,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute RQA scalars from a binary recurrence matrix.

    Args:
        R: (n, n) boolean recurrence matrix (Theiler band already False).
        theiler: the Theiler window used when building R; needed so the
            denominator and the diagonal scan exclude the same cells.
        l_min: minimum diagonal line length counted as a line.
        v_min: minimum vertical line length counted as a line.
        prefix: string prepended to every key, e.g. "rr05_".

    Returns:
        Dict of metric name -> value. NaN where undefined (too few sentences,
        or no recurrences at all).
    """
    R = np.asarray(R, dtype=bool)
    n = R.shape[0]
    keys = ["RR", "DET", "L", "L_MAX", "DIV", "ENTR", "RATIO", "LAM", "TT", "V_MAX"]

    if n < 3:
        return {f"{prefix}{k}": float("nan") for k in keys}

    mask = valid_mask(n, theiler)
    n_valid = int(mask.sum())
    n_rec = int(R.sum())

    if n_valid == 0:
        return {f"{prefix}{k}": float("nan") for k in keys}

    rr = n_rec / n_valid

    if n_rec == 0:
        out = {f"{prefix}{k}": float("nan") for k in keys}
        out[f"{prefix}RR"] = 0.0
        return out

    diag = _diagonal_lengths(R, theiler)
    vert = _vertical_lengths(R)

    diag_sel = diag[diag >= l_min]
    vert_sel = vert[vert >= v_min]

    det = float(diag_sel.sum() / n_rec) if diag_sel.size else 0.0
    lam = float(vert_sel.sum() / n_rec) if vert_sel.size else 0.0

    l_mean = float(diag_sel.mean()) if diag_sel.size else float("nan")
    l_max = float(diag.max()) if diag.size else float("nan")
    tt = float(vert_sel.mean()) if vert_sel.size else float("nan")
    v_max = float(vert.max()) if vert.size else float("nan")

    return {
        f"{prefix}RR": rr,
        f"{prefix}DET": det,
        f"{prefix}L": l_mean,
        f"{prefix}L_MAX": l_max,
        f"{prefix}DIV": (1.0 / l_max) if (l_max and l_max == l_max and l_max > 0) else float("nan"),
        f"{prefix}ENTR": _entropy(diag, l_min),
        f"{prefix}RATIO": (det / rr) if rr > 0 else float("nan"),
        f"{prefix}LAM": lam,
        f"{prefix}TT": tt,
        f"{prefix}V_MAX": v_max,
    }
