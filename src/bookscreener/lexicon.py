"""Offensive-word lexicon loading and lexicon-based features."""

from __future__ import annotations

from pathlib import Path

from .text import tokenize_words, turkish_lower

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "lexicon" / "offensive_tr.txt"


def load_lexicon(path: str | Path = DEFAULT_LEXICON_PATH) -> frozenset[str]:
    """Load one term per line; blank lines and ``#`` comments are ignored."""
    terms = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.add(turkish_lower(line))
    return frozenset(terms)


def lexicon_features(text: str, lexicon: frozenset[str]) -> dict[str, float]:
    """Whole-token matches only, so ``am`` never matches inside ``ama`` or ``adam``."""
    tokens = tokenize_words(text)
    count = sum(1 for t in tokens if t in lexicon)
    return {
        "offensive_count": count,
        "has_offensive": int(count > 0),
        "offensive_ratio": count / len(tokens) if tokens else 0.0,
    }
