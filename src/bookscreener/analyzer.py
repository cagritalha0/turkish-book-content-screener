"""Book-level analysis: score every sentence, aggregate, explain."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .categories import CATEGORY_LABELS, category_hits
from .features import build_features
from .scorer import SentenceScorer
from .text import atesman_band, atesman_score, split_sentences, turkish_lower

DEFAULT_SENTENCE_THRESHOLD = 0.5
DEFAULT_BOOK_RISK_THRESHOLD = 0.03  # share of risky sentences above which a book is flagged


@dataclass
class BookReport:
    total_sentences: int
    risky_sentences: list[str]
    sentence_probs: np.ndarray
    risk_ratio: float
    atesman: float
    atesman_band: str
    flagged: bool
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return "Needs review" if self.flagged else "Suitable"


class BookAnalyzer:
    """Scores sentences with BERTurk, optionally re-ranks them with a CatBoost meta-model.

    ``meta_model`` is any object with ``predict_proba(X) -> (n, 2)`` trained on
    :data:`bookscreener.features.FEATURE_COLUMNS`. Without it, the BERT
    probability is used directly (this was the stronger model in our evaluation).
    """

    def __init__(
        self,
        scorer: SentenceScorer,
        lexicon: frozenset[str],
        meta_model=None,
        sentence_threshold: float = DEFAULT_SENTENCE_THRESHOLD,
        book_risk_threshold: float = DEFAULT_BOOK_RISK_THRESHOLD,
    ):
        self.scorer = scorer
        self.lexicon = lexicon
        self.meta_model = meta_model
        self.sentence_threshold = sentence_threshold
        self.book_risk_threshold = book_risk_threshold

    def sentence_risk(self, sentences: list[str]) -> np.ndarray:
        model_inputs = [turkish_lower(s) for s in sentences]
        bert_probs = self.scorer.predict_proba(model_inputs)
        if self.meta_model is None:
            return np.asarray(bert_probs, dtype=float)
        features = build_features(model_inputs, bert_probs, self.lexicon)
        return np.asarray(self.meta_model.predict_proba(features))[:, 1]

    def analyze(self, text: str) -> BookReport:
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("No sentences found in the document.")

        probs = self.sentence_risk(sentences)
        risky_idx = np.flatnonzero(probs >= self.sentence_threshold)
        # most confident first, so the UI shows the clearest examples
        risky_idx = risky_idx[np.argsort(-probs[risky_idx])]
        risky = [sentences[i] for i in risky_idx]
        ratio = len(risky) / len(sentences)
        score = atesman_score(text)
        flagged = ratio >= self.book_risk_threshold

        reasons = {}
        if flagged:
            hits = category_hits(" ".join(risky))
            reasons = {CATEGORY_LABELS[c]: n for c, n in hits.most_common()}

        return BookReport(
            total_sentences=len(sentences),
            risky_sentences=risky,
            sentence_probs=probs,
            risk_ratio=ratio,
            atesman=score,
            atesman_band=atesman_band(score),
            flagged=flagged,
            reasons=reasons,
        )
