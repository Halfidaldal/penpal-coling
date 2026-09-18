"""
Canonical sentence segmentation — the single source of truth for sentence units.

Every pipeline that indexes sentences (embeddings/RQA, surprisal NTR, endpoint
predictability) must agree on what sentence *i* is, otherwise a join on
(story_id, sent_idx) silently pairs unrelated text.

That was a real failure mode here: the embedding pipeline used spaCy with a
3-word minimum while the surprisal pipeline used a regex with no minimum. Only
27% of stories segmented identically, and in 72% of stories index *i* referred
to different text in the two pipelines — regex does not split after a closing
quotation mark, which is pervasive in dialogue-heavy fiction:

    spaCy:  '"It was just gone when I woke up."'
    regex:  '"It was just gone when I woke up." Her roommate backed against ...'

The fix is to segment ONCE, write ``sentence_index.csv``, and have every
downstream step consume that file rather than re-segmenting. Remote jobs receive
the same file, so they cannot drift either.

Short sentences are KEPT (``min_words: 1``) so no story text is lost from the
contexts the language model conditions on. Analyses that need to exclude noisy
short units should filter on the exported ``n_words`` column instead.
"""

from pathlib import Path
from typing import Dict, List, Optional
import re

import pandas as pd

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_NLP_CACHE: dict = {}


def _get_spacy(model: str):
    """Load and cache a spaCy pipeline with only sentence splitting enabled."""
    if model not in _NLP_CACHE:
        import spacy
        nlp = spacy.load(model, exclude=["ner", "lemmatizer", "textcat"])
        nlp.max_length = 2_000_000
        _NLP_CACHE[model] = nlp
    return _NLP_CACHE[model]


def split_sentences(
    text: str,
    segmenter: str = "spacy",
    spacy_model: str = "en_core_web_md",
    min_words: int = 1,
    strict: bool = True,
) -> List[str]:
    """
    Split one story into sentences.

    Args:
        segmenter: "spacy" (recommended: handles dialogue and abbreviations) or
            "regex" (dependency-free, splits on [.!?] + whitespace).
        min_words: drop sentences shorter than this. Default 1 keeps everything.
        strict: when True a missing spaCy model raises instead of silently
            falling back to regex — a silent fallback would reintroduce exactly
            the misalignment this module exists to prevent.
    """
    text = "" if text is None else str(text).strip()
    if not text:
        return []

    if segmenter == "spacy":
        try:
            sents = [s.text.strip() for s in _get_spacy(spacy_model)(text).sents]
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"spaCy segmentation failed ({exc}). Install the model with "
                    f"`python -m spacy download {spacy_model}`, or set "
                    f"segmentation.segmenter: regex in config.yaml. Refusing to "
                    f"fall back silently: mixed segmenters misalign sent_idx."
                ) from exc
            print(f"[WARNING] spaCy unavailable ({exc}); using regex.")
            sents = _SENT_RE.split(text)
    elif segmenter == "regex":
        sents = _SENT_RE.split(text)
    else:
        raise ValueError(f"Unknown segmenter: {segmenter!r}")

    return [s.strip() for s in sents if len(s.split()) >= min_words]


def build_sentence_index(texts: Dict[str, str], cfg: dict) -> pd.DataFrame:
    """
    Segment a corpus into the canonical long-format sentence index.

    Args:
        texts: {story_id: story_text}.
        cfg: the ``segmentation`` section of config.yaml.

    Returns:
        DataFrame with columns [story_id, sent_idx, n_words, sentence],
        ordered by story then position. ``sent_idx`` is 0-based and contiguous
        within each story: it is the join key for every downstream measure.
    """
    segmenter = cfg.get("segmenter", "spacy")
    spacy_model = cfg.get("spacy_model", "en_core_web_md")
    min_words = int(cfg.get("min_words", 1))

    rows = []
    for sid, text in texts.items():
        for i, sent in enumerate(
            split_sentences(text, segmenter, spacy_model, min_words)
        ):
            rows.append({
                "story_id": sid,
                "sent_idx": i,
                "n_words": len(sent.split()),
                "sentence": sent,
            })
    return pd.DataFrame(rows)


def load_sentence_index(path) -> pd.DataFrame:
    """Load a canonical sentence index and validate its structure."""
    df = pd.read_csv(path)
    required = {"story_id", "sent_idx", "sentence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    df["sentence"] = df["sentence"].astype(str)
    if "n_words" not in df.columns:
        df["n_words"] = df["sentence"].str.split().str.len()
    return df.sort_values(["story_id", "sent_idx"]).reset_index(drop=True)


def sentences_by_story(index: pd.DataFrame) -> Dict[str, List[str]]:
    """Convert a sentence index into {story_id: [sentence, ...]} in order."""
    out: Dict[str, List[str]] = {}
    for sid, grp in index.sort_values("sent_idx").groupby("story_id", sort=False):
        out[sid] = grp["sentence"].tolist()
    return out


def verify_alignment(index: pd.DataFrame, other: pd.DataFrame, name_a="A", name_b="B") -> bool:
    """
    Check that two sentence indexes describe identical units.

    Use before joining artifacts produced by different pipelines or at different
    times. Returns True when aligned; prints the first divergence otherwise.
    """
    # Compare by (story_id, sent_idx) content, not by row order: a saved index is
    # sorted on load while a freshly built one preserves story order, and that
    # difference must not read as a misalignment.
    a = index.set_index(["story_id", "sent_idx"])["sentence"].sort_index()
    b = other.set_index(["story_id", "sent_idx"])["sentence"].sort_index()

    if len(a) != len(b) or not a.index.equals(b.index):
        only_a = a.index.difference(b.index)
        only_b = b.index.difference(a.index)
        print(f"[MISALIGNED] {name_a} has {len(a)} units, {name_b} has {len(b)}.")
        if len(only_a):
            print(f"  only in {name_a}: {list(only_a[:5])}")
        if len(only_b):
            print(f"  only in {name_b}: {list(only_b[:5])}")
        return False

    diff = a[a.values != b.values]
    if len(diff):
        (sid, idx) = diff.index[0]
        print(f"[MISALIGNED] first divergence at ({sid}, sent_idx={idx}):")
        print(f"  {name_a}: {a.loc[(sid, idx)][:90]!r}")
        print(f"  {name_b}: {b.loc[(sid, idx)][:90]!r}")
        return False
    return True


def summarise(index: pd.DataFrame, conditions: Optional[pd.Series] = None) -> None:
    """Print a short summary of a sentence index."""
    per_story = index.groupby("story_id").size()
    print(f"Sentences: {len(index):,} across {per_story.size} stories")
    print(f"  per story: mean {per_story.mean():.1f}, min {per_story.min()}, "
          f"max {per_story.max()}")
    print(f"  words per sentence: mean {index['n_words'].mean():.1f}, "
          f"median {index['n_words'].median():.0f}, max {index['n_words'].max()}")
    short = (index["n_words"] <= 3).mean()
    print(f"  sentences <= 3 words: {100 * short:.1f}%")
    if conditions is not None:
        merged = per_story.rename("n_sent").to_frame().join(conditions.rename("cond"))
        print("  per story by condition:")
        print(merged.groupby("cond")["n_sent"].mean().round(1).to_string())
