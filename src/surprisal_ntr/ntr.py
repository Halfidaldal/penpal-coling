"""
Window-based Novelty, Transience and Resonance.

Definitions follow the EMNLP analysis, with the *conditional* (marginal)
transience rather than the older isolated form. For window :math:`T_t` with
preceding story context :math:`C_t` and following window :math:`T_{t+1}`:

.. math::

    \\mathrm{Novelty}_t    &= \\bar s(T_t \\mid C_t) - \\bar s(T_t \\mid \\mathrm{BOS}) \\\\
    \\mathrm{Transience}_t &= \\bar s(T_{t+1} \\mid C_t, T_t) - \\bar s(T_{t+1} \\mid C_t) \\\\
    \\mathrm{Resonance}_t  &= \\mathrm{Novelty}_t - \\mathrm{Transience}_t

Both transience terms condition the *future* window on the running context and
differ only in whether the current window is included, so transience is the
marginal contribution of :math:`T_t` given what has already been established.
Novelty and transience therefore share the same running-context construction.

All quantities are in bits and are negative when context *reduces* surprisal, so
a less negative value marks a window the story predicts less well.

Note on interpretation: the window is sub-turn by design, so "the next window"
is usually the same author continuing. Transience here measures local
continuation, not uptake by a partner as in the turn-level EMNLP analysis.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from .lm import LMBundle, mean_surprisal


def make_windows(text: str, window_words: int, min_window_words: int = 3) -> List[str]:
    """Split a story into consecutive fixed-length word windows."""
    words = str(text).split()
    return [
        " ".join(words[i:i + window_words])
        for i in range(0, len(words), window_words)
        if len(words[i:i + window_words]) >= min_window_words
    ]


def compute_story_ntr(
    text: str,
    lm: LMBundle,
    window_words: int,
    min_window_words: int = 3,
    common_index_range: bool = True,
    save_raw_terms: bool = True,
) -> Tuple[Dict[str, float], List[dict]]:
    """
    Novelty / transience / resonance for one story.

    Args:
        text: the full story.
        lm: loaded model bundle.
        window_words: window length in words.
        common_index_range: aggregate all three measures over the same interior
            window range. Novelty is undefined at the first window and transience
            at the last, so averaging each over its own range would make the
            story means mutually inconsistent (resonance would no longer equal
            novelty minus transience).
        save_raw_terms: include the raw surprisal terms in the window records.

    Returns:
        (story summary dict, list of per-window record dicts)
    """
    units = make_windows(text, window_words, min_window_words)
    n = len(units)

    nan_summary = {
        "s_novelty": np.nan, "s_transience": np.nan, "s_resonance": np.nan,
        "n_windows_total": n, "n_windows_used": 0,
    }
    if n < 2:
        return nan_summary, []

    ids = [lm.encode(u) for u in units]

    # s(T_i | BOS): the context-free baseline for each window.
    base = [mean_surprisal(lm, ids[i]) for i in range(n)]

    # s(T_i | C_<i): each window given ALL preceding windows.
    s_ctx = [base[0]] + [
        mean_surprisal(lm, ids[i], torch.cat(ids[:i])) for i in range(1, n)
    ]

    nov = np.full(n, np.nan)
    tra = np.full(n, np.nan)

    for i in range(1, n):
        nov[i] = s_ctx[i] - base[i]

    for i in range(n - 1):
        # s(T_{i+1} | C_<i, T_i) is exactly s_ctx[i+1]; the counterfactual drops T_i.
        ctx_before = torch.cat(ids[:i]) if i > 0 else None
        tra[i] = s_ctx[i + 1] - mean_surprisal(lm, ids[i + 1], ctx_before)

    res = nov - tra

    records = []
    for i in range(n):
        rec = {
            "window_idx": i,
            "n_window_words": len(units[i].split()),
            "s_novelty": float(nov[i]),
            "s_transience": float(tra[i]),
            "s_resonance": float(res[i]),
            "window_text": units[i],
        }
        if save_raw_terms:
            rec["s_base"] = float(base[i])   # s(T_i | BOS)
            rec["s_ctx"] = float(s_ctx[i])   # s(T_i | C_<i)
        records.append(rec)

    if common_index_range:
        lo, hi = 1, n - 1                    # hi exclusive: interior windows only
    else:
        lo, hi = 0, n

    if hi <= lo:
        summary = dict(nan_summary)
    else:
        summary = {
            "s_novelty": float(np.nanmean(nov[lo:hi])),
            "s_transience": float(np.nanmean(tra[lo:hi])),
            "s_resonance": float(np.nanmean(res[lo:hi])),
            "n_windows_total": n,
            "n_windows_used": int(hi - lo),
        }
    return summary, records


def compute_corpus_ntr(
    story,
    lm: LMBundle,
    cfg: dict,
    progress: bool = True,
):
    """
    Run :func:`compute_story_ntr` over every story.

    Args:
        story: story table with a 'text' column (and optionally 'cond').
        lm: loaded model bundle.
        cfg: the ``surprisal`` section of config.yaml.

    Returns:
        (summary DataFrame indexed like `story`, long window-level DataFrame)
    """
    import pandas as pd

    iterator = story["text"].items()
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(story), desc=f"Surprisal ({lm.model_name})")

    summaries, windows = {}, []
    for sid, text in iterator:
        summary, records = compute_story_ntr(
            text=text,
            lm=lm,
            window_words=int(cfg["window_words"]),
            min_window_words=int(cfg.get("min_window_words", 3)),
            common_index_range=bool(cfg.get("common_index_range", True)),
            save_raw_terms=bool(cfg.get("save_raw_terms", True)),
        )
        summaries[sid] = summary
        for rec in records:
            rec["id"] = sid
            if "cond" in story.columns:
                rec["cond"] = story.loc[sid, "cond"]
            windows.append(rec)

    summary_df = pd.DataFrame(summaries).T
    window_df = pd.DataFrame(windows)
    return summary_df, window_df


def identity_gap(summary_df) -> float:
    """
    Largest violation of resonance == novelty - transience across stories.

    Should be ~0 when ``common_index_range`` is enabled; a large value means the
    three means were averaged over different window ranges.
    """
    gap = (
        summary_df["s_resonance"] - (summary_df["s_novelty"] - summary_df["s_transience"])
    ).abs()
    return float(gap.max(skipna=True))
