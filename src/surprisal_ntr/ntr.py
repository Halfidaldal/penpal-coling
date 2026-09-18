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
    units: List[str],
    lm: LMBundle,
    common_index_range: bool = True,
    save_raw_terms: bool = True,
) -> Tuple[Dict[str, float], List[dict]]:
    """
    Novelty / transience / resonance for one story.

    Args:
        units: the story's ordered units. Either fixed-width word windows (see
            :func:`make_windows`) or canonical sentences from the shared
            sentence index — the computation is identical either way, only the
            unit definition changes.
        lm: loaded model bundle.
        common_index_range: aggregate all three measures over the same interior
            range. Novelty is undefined at the first unit and transience at the
            last, so averaging each over its own range would make the story means
            mutually inconsistent (resonance would no longer equal novelty minus
            transience).
        save_raw_terms: include the raw surprisal terms in the unit records.

    Returns:
        (story summary dict, list of per-unit record dicts)
    """
    n = len(units)

    nan_summary = {
        "s_novelty": np.nan, "s_transience": np.nan, "s_resonance": np.nan,
        "n_units_total": n, "n_units_used": 0,
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
            "unit_idx": i,
            "n_unit_words": len(units[i].split()),
            "s_novelty": float(nov[i]),
            "s_transience": float(tra[i]),
            "s_resonance": float(res[i]),
            "unit_text": units[i],
        }
        if save_raw_terms:
            rec["s_base"] = float(base[i])   # s(T_i | BOS)
            rec["s_ctx"] = float(s_ctx[i])   # s(T_i | C_<i)
        records.append(rec)

    if common_index_range:
        lo, hi = 1, n - 1                    # hi exclusive: interior units only
    else:
        lo, hi = 0, n

    if hi <= lo:
        summary = dict(nan_summary)
    else:
        summary = {
            "s_novelty": float(np.nanmean(nov[lo:hi])),
            "s_transience": float(np.nanmean(tra[lo:hi])),
            "s_resonance": float(np.nanmean(res[lo:hi])),
            "n_units_total": n,
            "n_units_used": int(hi - lo),
        }
    return summary, records


def build_units(
    story,
    cfg: dict,
    sentence_index=None,
) -> Dict[str, List[str]]:
    """
    Build the per-story unit lists according to ``surprisal.unit``.

    Args:
        story: story table with a 'text' column, indexed by story id.
        cfg: the ``surprisal`` section of config.yaml.
        sentence_index: canonical sentence index (required when unit == "sentence").

    Returns:
        {story_id: [unit_text, ...]} in story order.
    """
    unit = cfg.get("unit", "window")

    if unit == "sentence":
        if sentence_index is None:
            raise ValueError(
                "unit: sentence requires the canonical sentence index. Run "
                "scripts/00b_build_sentence_index.py first, or set "
                "surprisal.unit: window in config.yaml."
            )
        from penpal_segmentation import sentences_by_story
        by_story = sentences_by_story(sentence_index)
        missing = [sid for sid in story.index if sid not in by_story]
        if missing:
            print(f"[WARNING] {len(missing)} story/ies absent from the sentence "
                  f"index and will be skipped: {missing[:5]}")
        return {sid: by_story.get(sid, []) for sid in story.index}

    if unit == "window":
        w = int(cfg["window_words"])
        mw = int(cfg.get("min_window_words", 3))
        return {sid: make_windows(t, w, mw) for sid, t in story["text"].items()}

    raise ValueError(f"Unknown surprisal.unit: {unit!r} (expected 'sentence' or 'window')")


def compute_corpus_ntr(
    story,
    lm: LMBundle,
    cfg: dict,
    sentence_index=None,
    progress: bool = True,
):
    """
    Run :func:`compute_story_ntr` over every story.

    Args:
        story: story table with a 'text' column (and optionally 'cond').
        lm: loaded model bundle.
        cfg: the ``surprisal`` section of config.yaml.
        sentence_index: canonical sentence index, required when unit == "sentence".

    Returns:
        (summary DataFrame indexed like `story`, long unit-level DataFrame)
        When unit == "sentence" the unit frame carries ``sent_idx`` (== unit_idx),
        which is the join key against the embedding/RQA per-sentence measures.
    """
    import pandas as pd

    unit_kind = cfg.get("unit", "window")
    units_by_story = build_units(story, cfg, sentence_index)

    iterator = units_by_story.items()
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(units_by_story),
                        desc=f"Surprisal [{unit_kind}] ({lm.model_name})")

    summaries, unit_rows = {}, []
    for sid, units in iterator:
        summary, records = compute_story_ntr(
            units=units,
            lm=lm,
            common_index_range=bool(cfg.get("common_index_range", True)),
            save_raw_terms=bool(cfg.get("save_raw_terms", True)),
        )
        summary["unit"] = unit_kind
        summaries[sid] = summary
        for rec in records:
            rec["id"] = sid
            rec["unit"] = unit_kind
            if unit_kind == "sentence":
                # Join key against the embedding / RQA per-sentence measures.
                rec["sent_idx"] = rec["unit_idx"]
            if "cond" in story.columns:
                rec["cond"] = story.loc[sid, "cond"]
            unit_rows.append(rec)

    summary_df = pd.DataFrame(summaries).T
    unit_df = pd.DataFrame(unit_rows)
    return summary_df, unit_df


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
