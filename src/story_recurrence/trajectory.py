"""
Embedding drift and semantic trajectory geometry.

A story is treated as an ordered walk through embedding space: sentence i is a
point e_i on the unit sphere, and the story is the path e_1 -> e_2 -> ... -> e_n.
These measures describe the *shape* of that path, which is a global property no
local window can see.

Measures
--------
path_length        sum of consecutive step sizes. Total semantic ground covered.
mean_step / sd_step / max_step
                   step-size distribution. Large mean_step = jumpy prose;
                   sd_step = uneven, i.e. a few big leaps among small moves.
net_displacement   straight-line distance from first to last sentence.
straightness       net_displacement / path_length, in [0, 1]. 1 = the story
                   moves consistently in one direction; near 0 = it wanders or
                   returns to where it began.
radius_gyration    RMS distance of sentences from the story centroid. How much
                   of the space the story occupies, independent of order.
centroid_norm      ||mean(e_i)||, in [0, 1]. High = every sentence points the
                   same way (topically concentrated); low = dispersed.
msd_alpha          diffusion exponent from log MSD(lag) ~ alpha * log(lag).
                   alpha ~ 1 random-walk-like; alpha > 1 directed/superdiffusive
                   (the story keeps going somewhere); alpha < 1 confined (it
                   circles a theme). Fitted only when enough lags are available.
msd_alpha_r2       fit quality for the above.
circularity        cosine between the mean of the first and last `edge_frac` of
                   sentences. High = the ending returns to the opening.
first_last_cos     cosine between the single first and last sentence.
mean_pairwise_sim / sd_pairwise_sim
                   order-free summary of overall self-similarity.

All inputs are assumed L2-normalised, so Euclidean distance and cosine distance
are monotonically related: ||a - b||^2 = 2 * (1 - cos(a, b)).
"""

from typing import Dict

import numpy as np

from .matrices import valid_mask


def _safe(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def trajectory_metrics(
    embeddings: np.ndarray,
    similarity: np.ndarray,
    edge_frac: float = 0.2,
    min_lag_points: int = 4,
) -> Dict[str, float]:
    """
    Geometry of one story's sentence trajectory.

    Args:
        embeddings: (n, d) L2-normalised sentence embeddings, in story order.
        similarity: (n, n) cosine similarity matrix for the same story.
        edge_frac: fraction of sentences treated as the opening / closing
            segment for the circularity measure (0.2 = first and last 20%).
        min_lag_points: minimum number of distinct lags required to fit the
            MSD exponent.

    Returns:
        Dict of metric name -> value, NaN where undefined.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    n = E.shape[0]
    keys = [
        "path_length", "mean_step", "sd_step", "max_step", "net_displacement",
        "straightness", "radius_gyration", "centroid_norm", "msd_alpha",
        "msd_alpha_r2", "circularity", "first_last_cos",
        "mean_pairwise_sim", "sd_pairwise_sim",
    ]
    if n < 3:
        return {k: float("nan") for k in keys}

    # --- step statistics -------------------------------------------------
    steps = np.linalg.norm(np.diff(E, axis=0), axis=1)
    path_length = float(steps.sum())
    net_disp = float(np.linalg.norm(E[-1] - E[0]))

    # --- dispersion around the centroid ----------------------------------
    centroid = E.mean(axis=0)
    radius_gyr = float(np.sqrt(np.mean(np.sum((E - centroid) ** 2, axis=1))))
    centroid_norm = float(np.linalg.norm(centroid))

    # --- mean squared displacement vs lag --------------------------------
    # MSD(k) = mean over i of ||e_{i+k} - e_i||^2. A straight fit in log-log
    # space gives the diffusion exponent.
    max_lag = max(1, n // 2)
    lags, msd = [], []
    for k in range(1, max_lag + 1):
        d = E[k:] - E[:-k]
        if d.shape[0] < 2:
            continue
        lags.append(k)
        msd.append(float(np.mean(np.sum(d ** 2, axis=1))))

    alpha, alpha_r2 = float("nan"), float("nan")
    if len(lags) >= min_lag_points:
        x = np.log(np.asarray(lags, dtype=float))
        y = np.asarray(msd, dtype=float)
        if np.all(y > 0):
            y = np.log(y)
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            alpha = float(slope)
            alpha_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # --- circularity: does the ending return to the opening? -------------
    k_edge = max(1, int(round(edge_frac * n)))
    open_vec = E[:k_edge].mean(axis=0)
    close_vec = E[-k_edge:].mean(axis=0)
    denom = np.linalg.norm(open_vec) * np.linalg.norm(close_vec)
    circularity = float(np.dot(open_vec, close_vec) / denom) if denom > 0 else float("nan")

    # --- order-free self-similarity --------------------------------------
    S = np.asarray(similarity, dtype=np.float64)
    off = S[valid_mask(n, 0)]

    return {
        "path_length": path_length,
        "mean_step": _safe(steps.mean()),
        "sd_step": _safe(steps.std(ddof=1)) if steps.size > 1 else float("nan"),
        "max_step": _safe(steps.max()),
        "net_displacement": net_disp,
        "straightness": _safe(net_disp / path_length) if path_length > 0 else float("nan"),
        "radius_gyration": radius_gyr,
        "centroid_norm": centroid_norm,
        "msd_alpha": alpha,
        "msd_alpha_r2": alpha_r2,
        "circularity": circularity,
        "first_last_cos": _safe(np.dot(E[0], E[-1])),
        "mean_pairwise_sim": _safe(off.mean()),
        "sd_pairwise_sim": _safe(off.std(ddof=1)) if off.size > 1 else float("nan"),
    }
