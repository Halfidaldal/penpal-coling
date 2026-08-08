"""
Transparent surface features computed directly from each completed story.

Nothing here depends on a language model, so this module runs on CPU in
seconds and can be re-run independently of the surprisal step.
"""

import re
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WORD_REGEX = re.compile(r"[A-Za-z']+")
SENT_REGEX = re.compile(r"(?<=[.!?])\s+")


def count_syllables(word: str) -> int:
    """Approximate syllable count used by the Flesch readability formula."""
    w = word.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    return max(1, n - (1 if w.endswith("e") else 0))


def story_features(text: str) -> Dict[str, float]:
    """
    Surface features for one story.

    Returns length, sentence-length statistics, lexical variety (TTR, hapax),
    comma density, an approximate Flesch score (higher = easier), and mean
    TF-IDF cosine similarity between adjacent sentences (local continuity).
    """
    words = WORD_REGEX.findall(str(text))
    sentences = [s for s in SENT_REGEX.split(str(text).strip()) if s.split()]

    n_words = len(words)
    n_sents = max(1, len(sentences))
    words_low = [w.lower() for w in words]
    vocab = set(words_low)
    # Counter is O(n) overall rather than O(n^2) from repeated list.count()
    from collections import Counter
    freq = Counter(words_low)
    hapax_count = sum(1 for w, c in freq.items() if c == 1)

    out = {
        "n_words": float(n_words),
        "n_sentences": float(len(sentences)),
        "mean_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mean_sentence_length": n_words / n_sents,
        "ttr": len(vocab) / n_words if n_words else 0.0,
        "hapax_ratio": hapax_count / n_words if n_words else 0.0,
        "comma_density": (str(text).count(",") / n_words) * 1000.0 if n_words else 0.0,
        "flesch_approx": (
            206.835
            - 1.015 * (n_words / n_sents)
            - 84.6 * (sum(count_syllables(w) for w in words) / n_words if n_words else 0.0)
        ),
    }

    try:
        if len(sentences) >= 2:
            sim = cosine_similarity(TfidfVectorizer().fit_transform(sentences))
            out["adjacent_sentence_similarity"] = float(
                np.mean([sim[i, i + 1] for i in range(len(sentences) - 1)])
            )
        else:
            out["adjacent_sentence_similarity"] = np.nan
    except Exception:
        out["adjacent_sentence_similarity"] = np.nan

    return out


def semantic_distinctiveness(texts, ngram_range=(1, 2), min_df: int = 2) -> np.ndarray:
    """
    Corpus-relative distinctiveness: 1 - cosine similarity to the nearest OTHER
    story in TF-IDF space. High values mean the story stands apart from the rest
    of the corpus.
    """
    matrix = TfidfVectorizer(ngram_range=tuple(ngram_range), min_df=min_df).fit_transform(list(texts))
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)
    return 1.0 - sim.max(axis=1)


def compute(story: pd.DataFrame, cfg: dict, progress: bool = True) -> pd.DataFrame:
    """
    Stylometric features for every story in the table.

    Args:
        story: story table with a 'text' column.
        cfg: the ``stylometry`` section of config.yaml.

    Returns:
        DataFrame indexed like `story` with one column per feature.
    """
    iterator = story["text"].items()
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(story), desc="Stylometry")

    feats = pd.DataFrame({i: story_features(t) for i, t in iterator}).T
    feats.index.name = story.index.name

    feats["semantic_distinctiveness"] = semantic_distinctiveness(
        story["text"].tolist(),
        ngram_range=cfg.get("tfidf_ngram_range", (1, 2)),
        min_df=int(cfg.get("tfidf_min_df", 2)),
    )
    return feats
