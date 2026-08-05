"""
Semantic decay rate: how fast does similarity fall off with narrative distance?

For each lag k, average the cosine similarity of every sentence pair k apart:

    sim(k) = mean_i  cos(e_i, e_{i+k})

sim(k) almost always decreases with k — sentences far apart in a story are less
alike. *How fast* it decreases is the quantity of interest:

  - fast decay  -> the story is locally coherent but globally drifting; material
                   introduced early has no relation to the ending.
  - slow decay  -> the story holds a stable set of entities/themes across its
                   whole span; distant sentences still resemble each other.
  - a floor well above zero -> a persistent global "topic" the story never leaves.

Three complementary summaries are fitted to the profile, because no single
functional form is obviously correct for prose:

  exponential-with-offset   sim(k) = C + A * exp(-lambda * k)
        lambda  = decay rate (higher = faster forgetting)
        C       = asymptotic floor (persistent global similarity)
        A       = decaying amplitude (locally-bound similarity)
        half_life = ln(2) / lambda, in sentences

  power law                 log sim(k) = log A - beta * log k
        beta    = scale-free decay exponent; appropriate if the story has no
                  characteristic memory length

  linear                    sim(k) = a + b * k
        b       = assumption-free slope, always defined, useful as a fallback
                  and as a sanity check on the two model fits

Reported alongside are the raw anchors (sim at lag 1, at the maximum fitted lag,
and the empirical half-distance), so the analysis never has to trust a fit.

Only lags supported by at least ``min_pairs`` sentence pairs are used, and the
maximum lag is capped at half the story length, because the tail of the profile
is estimated from very few pairs and is extremely noisy.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def lag_similarity_profile(
    similarity: np.ndarray,
    min_pairs: int = 5,
    max_lag_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean cosine similarity as a function of sentence lag.

    Args:
        similarity: (n, n) cosine similarity matrix for one story.
        min_pairs: drop lags supported by fewer than this many pairs.
        max_lag_frac: cap the largest lag at this fraction of story length.

    Returns:
        (lags, mean_sim, n_pairs) as three aligned 1-D arrays.
    """
    S = np.asarray(similarity, dtype=np.float64)
    n = S.shape[0]
    if n < 3:
        return np.empty(0), np.empty(0), np.empty(0)

    max_lag = max(1, int(np.floor(max_lag_frac * n)))
    lags, sims, counts = [], [], []
    for k in range(1, max_lag + 1):
        diag = np.diagonal(S, offset=k)
        if diag.size < min_pairs:
            continue
        lags.append(k)
        sims.append(float(diag.mean()))
        counts.append(int(diag.size))

    return np.asarray(lags), np.asarray(sims), np.asarray(counts)


def _fit_exponential(lags: np.ndarray, sims: np.ndarray) -> Dict[str, float]:
    """Least-squares fit of sim(k) = C + A * exp(-lambda * k)."""
    out = {"decay_lambda": float("nan"), "decay_A": float("nan"),
           "decay_C": float("nan"), "decay_exp_r2": float("nan"),
           "decay_half_life": float("nan"), "decay_fit_at_bound": float("nan")}
    if lags.size < 4:
        return out
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return out

    def model(k, A, lam, C):
        return C + A * np.exp(-lam * k)

    # Sensible starting point: amplitude = drop across the profile, floor = tail.
    a0 = max(float(sims[0] - sims[-1]), 1e-3)
    c0 = float(sims[-1])
    p0 = [a0, 0.3, c0]
    lo = [0.0, 1e-6, -1.0]
    hi = [2.0, 10.0, 1.0]
    try:
        popt, _ = curve_fit(
            model, lags.astype(float), sims.astype(float), p0=p0, maxfev=20000,
            bounds=(lo, hi),
        )
    except Exception:
        return out

    A, lam, C = (float(v) for v in popt)

    # A parameter sitting on its bound means the optimiser could not find an
    # interior solution: the profile is effectively flat, or it collapses within
    # one lag. Such a fit is degenerate and should be screened out downstream
    # rather than treated as a fast/slow decay estimate.
    at_bound = any(
        abs(v - b) < 1e-6
        for v, b in zip((A, lam, C), lo)
    ) or any(
        abs(v - b) < 1e-6
        for v, b in zip((A, lam, C), hi)
    )
    pred = model(lags, A, lam, C)
    ss_res = float(np.sum((sims - pred) ** 2))
    ss_tot = float(np.sum((sims - sims.mean()) ** 2))

    out.update(
        decay_lambda=lam,
        decay_A=A,
        decay_C=C,
        decay_exp_r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        decay_half_life=(float(np.log(2) / lam) if lam > 0 else float("nan")),
        decay_fit_at_bound=float(bool(at_bound)),
    )
    return out


def _fit_power(lags: np.ndarray, sims: np.ndarray) -> Dict[str, float]:
    """Log-log fit of sim(k) = A * k^(-beta). Requires strictly positive sims."""
    out = {"decay_beta": float("nan"), "decay_pow_r2": float("nan")}
    ok = sims > 0
    if ok.sum() < 4:
        return out
    x = np.log(lags[ok].astype(float))
    y = np.log(sims[ok].astype(float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.update(
        decay_beta=float(-slope),
        decay_pow_r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    )
    return out


def _empirical_half_distance(lags: np.ndarray, sims: np.ndarray) -> float:
    """
    First lag at which similarity falls halfway from sim(lag 1) to the profile
    minimum, by linear interpolation. Fit-free companion to decay_half_life.
    """
    if lags.size < 3:
        return float("nan")
    hi, lo = float(sims[0]), float(sims.min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        return float("nan")
    target = lo + 0.5 * (hi - lo)
    for idx in range(1, sims.size):
        if sims[idx] <= target:
            s0, s1 = sims[idx - 1], sims[idx]
            k0, k1 = float(lags[idx - 1]), float(lags[idx])
            if s0 == s1:
                return k1
            return float(k0 + (s0 - target) / (s0 - s1) * (k1 - k0))
    return float("nan")


def decay_metrics(
    similarity: np.ndarray,
    min_pairs: int = 5,
    max_lag_frac: float = 0.5,
) -> Tuple[Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Semantic decay metrics for one story.

    Returns:
        (metrics dict, (lags, sims, counts) profile or None)
        The profile is returned so the pipeline can export it in long format
        for modelling in R.
    """
    lags, sims, counts = lag_similarity_profile(similarity, min_pairs, max_lag_frac)

    keys = ["decay_lambda", "decay_A", "decay_C", "decay_exp_r2", "decay_half_life",
            "decay_fit_at_bound", "decay_beta", "decay_pow_r2", "decay_lin_slope",
            "decay_lin_r2", "decay_half_distance", "sim_lag1", "sim_maxlag",
            "decay_max_lag", "decay_n_lags"]
    if lags.size < 3:
        return {k: float("nan") for k in keys}, None

    metrics: Dict[str, float] = {}
    metrics.update(_fit_exponential(lags, sims))
    metrics.update(_fit_power(lags, sims))

    # Assumption-free linear slope.
    slope, intercept = np.polyfit(lags.astype(float), sims.astype(float), 1)
    pred = slope * lags + intercept
    ss_res = float(np.sum((sims - pred) ** 2))
    ss_tot = float(np.sum((sims - sims.mean()) ** 2))
    metrics["decay_lin_slope"] = float(slope)
    metrics["decay_lin_r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    metrics["decay_half_distance"] = _empirical_half_distance(lags, sims)
    metrics["sim_lag1"] = float(sims[0])
    metrics["sim_maxlag"] = float(sims[-1])
    metrics["decay_max_lag"] = float(lags[-1])
    metrics["decay_n_lags"] = float(lags.size)

    return metrics, (lags, sims, counts)
