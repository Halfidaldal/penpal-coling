"""
Endpoint predictability: how the surprisal of a story's ending falls as the
story accumulates.

The story is split into a *body* (sentences 1..n) and a fixed *target* (the
final sentences). We then measure the mean token surprisal of the target as the
body is revealed one sentence at a time:

    S_0 = s(target | BOS)                 nothing of the story known
    S_i = s(target | sentence_1..i)       first i sentences known
    ...
    S_n = s(target | whole body)          the full story known

Two derived series:

    drop_i       = S_{i-1} - S_i          what sentence i contributed
    cumulative_i = S_0 - S_i              how much has been earned by sentence i

This is a genuinely *global* measure: every point conditions on all preceding
text against one fixed target. Where the window measures ask "is this passage
locally predictable", this asks "does the story build toward its ending, or
arrive at it abruptly?".

Positive drops mean a sentence made the ending more expected. Negative drops
mean it made the ending *less* expected -- a swerve away from where the story
was heading.

Shape statistics distinguish stories that converge steadily (high linearity,
early AUC) from those that only resolve in the last few sentences (high tail
ratio, late AUC).
"""

from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import torch

# Shared with the window-based measures: one model bundle, one definition of
# mean token surprisal, so the two families are directly comparable in bits.
from surprisal_ntr.lm import LMBundle, mean_surprisal  # noqa: F401

SENT_REGEX = re.compile(r"(?<=[.!?])\s+")

_NLP_CACHE: dict = {}


def split_sentences(
    text: str,
    segmenter: str = "regex",
    spacy_model: str = "en_core_web_md",
    min_words: int = 1,
) -> List[str]:
    """
    Split a story into sentences.

    ``min_words`` defaults to 1 (keep everything). Unlike the embedding pipeline,
    where short fragments produce unstable vectors and are filtered out, here
    each sentence is only appended to a growing context — so dropping any of them
    would silently remove text from the story the model conditions on.
    """
    text = "" if text is None else str(text).strip()
    if not text:
        return []

    if segmenter == "spacy":
        try:
            if spacy_model not in _NLP_CACHE:
                import spacy
                nlp = spacy.load(spacy_model, exclude=["ner", "lemmatizer", "textcat"])
                nlp.max_length = 2_000_000
                _NLP_CACHE[spacy_model] = nlp
            sents = [s.text.strip() for s in _NLP_CACHE[spacy_model](text).sents]
        except Exception as exc:
            print(f"[WARNING] spaCy unavailable ({exc}); falling back to regex.")
            sents = SENT_REGEX.split(text)
    else:
        sents = SENT_REGEX.split(text)

    return [s.strip() for s in sents if len(s.split()) >= min_words]


def split_body_target(
    sentences: List[str],
    target_mode: str = "last_sentences",
    n_target_sentences: int = 2,
    target_fraction: float = 0.1,
) -> Tuple[List[str], str]:
    """
    Split sentences into (body, target).

    Args:
        target_mode: "last_sentences" takes a fixed count; "last_fraction" takes
            a proportion of the story (at least one sentence).

    Returns:
        (body sentences, target text). Target is empty if the story is too short.
    """
    n = len(sentences)
    if n < 2:
        return [], ""

    if target_mode == "last_fraction":
        k = max(1, int(round(target_fraction * n)))
    else:
        k = max(1, int(n_target_sentences))

    k = min(k, n - 1)          # always leave at least one body sentence
    return sentences[:n - k], " ".join(sentences[n - k:])


def _shape_stats(
    surprisals: np.ndarray,
    tail_fraction: float,
) -> Dict[str, float]:
    """
    Shape statistics of one endpoint-predictability trajectory.

    Args:
        surprisals: S_0 .. S_n, length n+1.
        tail_fraction: fraction of body sentences treated as the tail.
    """
    s0 = float(surprisals[0])
    s_final = float(surprisals[-1])
    drops = -np.diff(surprisals)                 # drop_i = S_{i-1} - S_i
    n = drops.size
    cumulative = s0 - surprisals[1:]             # cumulative_i, length n
    total_drop = s0 - s_final

    out: Dict[str, float] = {
        "ep_s0": s0,
        "ep_s_final": s_final,
        "ep_total_drop": total_drop,
        "ep_n_body_sentences": float(n),
        "ep_mean_delta": float(np.mean(drops)),
        "ep_var_delta": float(np.var(drops)),
        "ep_max_delta": float(np.max(drops)),
        # Where the single most informative sentence sits, on a 0-1 scale.
        "ep_max_delta_pos": float((np.argmax(drops) + 1) / n),
        "ep_frac_negative_drops": float(np.mean(drops < 0)),
    }

    # Linear fit of cumulative gain against NORMALISED position, so the slope is
    # comparable across stories of different length.
    x_norm = np.arange(1, n + 1, dtype=float) / n
    if n >= 2:
        slope, intercept = np.polyfit(x_norm, cumulative, 1)
        pred = slope * x_norm + intercept
        ss_res = float(np.sum((cumulative - pred) ** 2))
        ss_tot = float(np.sum((cumulative - cumulative.mean()) ** 2))
        out["ep_slope_norm"] = float(slope)
        out["ep_linearity_r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        out["ep_slope_norm"] = float("nan")
        out["ep_linearity_r2"] = float("nan")

    # Earliness of convergence: normalised area under the cumulative curve.
    # 0.5 = gain accrues linearly; > 0.5 = the ending is largely determined early;
    # < 0.5 = the story only resolves late. Undefined when total_drop ~ 0.
    if abs(total_drop) > 1e-9:
        frac = cumulative / total_drop
        out["ep_auc"] = float(np.trapezoid(frac, x_norm)) if hasattr(np, "trapezoid") \
            else float(np.trapz(frac, x_norm))
    else:
        out["ep_auc"] = float("nan")

    # Do the last sentences do disproportionate work?
    tail_size = max(1, int(round(tail_fraction * n)))
    tail = drops[-tail_size:]
    rest = drops[:-tail_size]
    mean_tail = float(np.mean(tail))
    mean_rest = float(np.mean(rest)) if rest.size else float("nan")
    out["ep_mean_delta_tail"] = mean_tail
    out["ep_mean_delta_rest"] = mean_rest

    # Share of the total gain earned in the tail. Preferred over the ratio
    # below: bounded, and defined even when the rest of the story contributes
    # nothing -- which is exactly the "resolves only at the very end" case the
    # ratio blows up on. ~tail_fraction means the tail pulls its weight;
    # much higher means the ending arrives abruptly.
    out["ep_tail_share"] = float(np.sum(tail) / total_drop) if abs(total_drop) > 1e-9 \
        else float("nan")

    # Legacy ratio (mean tail drop / mean earlier drop). NaN when the earlier
    # drops average ~0, so prefer ep_tail_share for modelling.
    out["ep_tail_ratio"] = (mean_tail / mean_rest) if (rest.size and abs(mean_rest) > 1e-9) \
        else float("nan")

    return out


def compute_story_endpoint(
    text: str,
    lm: LMBundle,
    sentences: Optional[List[str]] = None,
    target_mode: str = "last_sentences",
    n_target_sentences: int = 2,
    target_fraction: float = 0.1,
    min_body_sentences: int = 5,
    tail_fraction: float = 0.2,
    segmenter: str = "regex",
    spacy_model: str = "en_core_web_md",
) -> Tuple[Dict[str, float], List[dict]]:
    """
    Endpoint predictability for one story.

    Returns:
        (metrics dict, trajectory records)
        The trajectory has one record per step, including step 0 (BOS baseline),
        so it can be modelled directly rather than only through the summary.
    """
    nan_keys = [
        "ep_s0", "ep_s_final", "ep_total_drop", "ep_n_body_sentences",
        "ep_mean_delta", "ep_var_delta", "ep_max_delta", "ep_max_delta_pos",
        "ep_frac_negative_drops", "ep_slope_norm", "ep_linearity_r2", "ep_auc",
        "ep_mean_delta_tail", "ep_mean_delta_rest", "ep_tail_share", "ep_tail_ratio",
    ]

    # Prefer canonical sentences when supplied; only re-segment as a fallback.
    if sentences is None:
        sentences = split_sentences(text, segmenter=segmenter, spacy_model=spacy_model)
    body, target = split_body_target(
        sentences, target_mode, n_target_sentences, target_fraction
    )

    if not target or len(body) < min_body_sentences:
        summary = {k: float("nan") for k in nan_keys}
        summary["ep_n_sentences_total"] = float(len(sentences))
        summary["ep_n_target_sentences"] = float(len(sentences) - len(body)) if sentences else 0.0
        return summary, []

    target_ids = lm.encode(target)
    body_ids = [lm.encode(s) for s in body]

    # S_0: the target with no story context at all.
    surprisals = [mean_surprisal(lm, target_ids, None)]

    # S_i: reveal one more body sentence each step.
    context = torch.empty(0, dtype=torch.long)
    for ids in body_ids:
        context = torch.cat([context, ids])
        surprisals.append(mean_surprisal(lm, target_ids, context))

    surprisals = np.asarray(surprisals, dtype=float)
    summary = _shape_stats(surprisals, tail_fraction)
    summary["ep_n_sentences_total"] = float(len(sentences))
    summary["ep_n_target_sentences"] = float(len(sentences) - len(body))

    records = []
    for i, s in enumerate(surprisals):
        records.append({
            "step": i,                                   # 0 = BOS baseline
            "n_body_revealed": i,
            "pos_norm": (i / len(body)) if len(body) else float("nan"),
            "s_target": float(s),
            "drop": float(surprisals[i - 1] - s) if i > 0 else float("nan"),
            "cumulative_drop": float(surprisals[0] - s),
            "sentence": body[i - 1] if i > 0 else "",
        })

    return summary, records


def compute_corpus_endpoint(story, lm: LMBundle, cfg: dict,
                            sentence_index=None, progress: bool = True):
    """
    Run :func:`compute_story_endpoint` over every story.

    Args:
        story: table with a 'text' column (and optionally 'cond'), indexed by id.
        lm: loaded model bundle.
        cfg: the ``endpoint`` section of config.yaml.

    Returns:
        (summary DataFrame indexed like `story`, long trajectory DataFrame)
    """
    import pandas as pd

    # Prefer the canonical sentence index so this pipeline segments identically
    # to the embedding and surprisal pipelines.
    sentences_by_story = None
    if cfg.get("use_sentence_index", True) and sentence_index is not None:
        from penpal_segmentation import sentences_by_story as _sbs
        sentences_by_story = _sbs(sentence_index)

    iterator = story["text"].items()
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(story), desc=f"Endpoint ({lm.model_name})")

    summaries, traj = {}, []
    for sid, text in iterator:
        summary, records = compute_story_endpoint(
            text=text,
            sentences=sentences_by_story.get(sid) if sentences_by_story else None,
            lm=lm,
            target_mode=cfg.get("target_mode", "last_sentences"),
            n_target_sentences=int(cfg.get("n_target_sentences", 2)),
            target_fraction=float(cfg.get("target_fraction", 0.1)),
            min_body_sentences=int(cfg.get("min_body_sentences", 5)),
            tail_fraction=float(cfg.get("tail_fraction", 0.2)),
            segmenter=cfg.get("segmenter", "regex"),
            spacy_model=cfg.get("spacy_model", "en_core_web_md"),
        )
        summaries[sid] = summary
        for rec in records:
            rec["id"] = sid
            if "cond" in story.columns:
                rec["cond"] = story.loc[sid, "cond"]
            traj.append(rec)

    return pd.DataFrame(summaries).T, pd.DataFrame(traj)
