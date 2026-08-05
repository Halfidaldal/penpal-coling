"""
Sentence segmentation for completed stories.

The unit of analysis for every downstream measure is the sentence, so this
module is the single place where "what counts as a sentence" is decided.

Two segmenters are available:
  - "spacy"  : uses the model named in config (default en_core_web_md). Better on
               dialogue, abbreviations and ellipses, which are common in fiction.
  - "regex"  : dependency-free fallback, splits on [.!?] followed by whitespace.

Sentences shorter than ``min_words`` are dropped *before* embedding. Very short
fragments ("Yes.", "He ran.") produce unstable embeddings and would inflate the
apparent similarity structure, which is exactly the failure mode we saw with
sentence-level LDA.
"""

from typing import List, Optional
import re

_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_NLP_CACHE = {}


def _get_spacy(model: str = "en_core_web_md"):
    """Load and cache a spaCy pipeline with only the sentence splitter enabled."""
    if model in _NLP_CACHE:
        return _NLP_CACHE[model]
    import spacy
    # Only the parser (or senter) is needed; disabling the rest keeps this fast.
    nlp = spacy.load(model, exclude=["ner", "lemmatizer", "textcat"])
    nlp.max_length = 2_000_000
    _NLP_CACHE[model] = nlp
    return nlp


def split_sentences(
    text: str,
    segmenter: str = "spacy",
    spacy_model: str = "en_core_web_md",
    min_words: int = 3,
) -> List[str]:
    """
    Split one story into a list of sentences.

    Args:
        text: the full story text.
        segmenter: "spacy" or "regex". Falls back to regex if spaCy is missing.
        spacy_model: spaCy model name, used when segmenter == "spacy".
        min_words: drop sentences with fewer than this many whitespace tokens.

    Returns:
        List of sentence strings, in story order.
    """
    text = "" if text is None else str(text).strip()
    if not text:
        return []

    if segmenter == "spacy":
        try:
            nlp = _get_spacy(spacy_model)
            sents = [s.text.strip() for s in nlp(text).sents]
        except Exception as exc:  # missing model / import error
            print(f"[WARNING] spaCy segmentation unavailable ({exc}); using regex fallback.")
            sents = _SENT_RE.split(text)
    else:
        sents = _SENT_RE.split(text)

    return [s.strip() for s in sents if len(s.split()) >= min_words]


def segment_corpus(
    texts: dict,
    segmenter: str = "spacy",
    spacy_model: str = "en_core_web_md",
    min_words: int = 3,
) -> dict:
    """
    Segment a whole corpus.

    Args:
        texts: mapping {story_id: story_text}.

    Returns:
        Mapping {story_id: [sentence, ...]} preserving story order.
    """
    return {
        sid: split_sentences(t, segmenter=segmenter, spacy_model=spacy_model, min_words=min_words)
        for sid, t in texts.items()
    }
