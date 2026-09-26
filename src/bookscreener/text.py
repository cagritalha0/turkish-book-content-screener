"""Turkish-aware text utilities: casing, cleaning, tokenisation, readability."""

from __future__ import annotations

import re
from functools import lru_cache

_TURKISH_UPPER_MAP = str.maketrans({"I": "ı", "İ": "i"})
_VOWELS = frozenset("aeıioöuü")
_WORD_RE = re.compile(r"[a-zçğıöşüâîû0-9]+")
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?…])\s+")


def turkish_lower(text: str) -> str:
    """Lower-case with Turkish rules.

    ``str.lower()`` maps ``I`` to ``i`` and ``İ`` to ``i`` + combining dot,
    which breaks dictionary look-ups such as ``IRMAK`` -> ``ırmak``.
    """
    return text.translate(_TURKISH_UPPER_MAP).lower()


def clean_social_text(text: str) -> str:
    """Normalise a tweet the same way for training and for inference."""
    text = str(text)
    text = re.sub(r"@[\w_]+", "<user>", text)
    text = re.sub(r"http\S+|www\.\S+", "<url>", text)
    text = text.replace("#", "")
    text = re.sub(r"[^\w\s,.!?<>'\"-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return turkish_lower(text)


def tokenize_words(text: str) -> list[str]:
    """Split into lower-cased Turkish word tokens (punctuation dropped)."""
    return _WORD_RE.findall(turkish_lower(text))


@lru_cache(maxsize=1)
def _punkt_turkish():
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
    except ImportError:
        return None

    def splitter(text: str) -> list[str]:
        return sent_tokenize(text, language="turkish")

    for attempt in range(2):
        try:
            splitter("Deneme. Test.")
            return splitter
        except LookupError:
            if attempt == 0:
                try:
                    nltk.download("punkt_tab", quiet=True)
                except Exception:
                    break
    return None  # Punkt data unavailable (e.g. offline): use the regex fallback


def split_sentences(text: str) -> list[str]:
    """Split a book into sentences, using NLTK Punkt for Turkish when available."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    splitter = _punkt_turkish()
    sentences = splitter(text) if splitter else _SENTENCE_FALLBACK_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def count_syllables(word: str) -> int:
    """In Turkish every syllable has exactly one vowel."""
    return sum(1 for ch in turkish_lower(word) if ch in _VOWELS)


def atesman_score(text: str) -> float:
    """Ateşman (1997) readability score for Turkish text.

    ``198.825 - 40.175 * (syllables / words) - 2.610 * (words / sentences)``.
    Higher is easier; roughly 0-100.
    """
    sentences = [s for s in re.split(r"[.!?…]+", text) if s.strip()]
    words = tokenize_words(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    score = 198.825 - 40.175 * (syllables / len(words)) - 2.610 * (len(words) / len(sentences))
    return round(score, 2)


def atesman_band(score: float) -> str:
    """Ateşman's own difficulty bands."""
    if score >= 90:
        return "very easy"
    if score >= 70:
        return "easy"
    if score >= 50:
        return "medium"
    if score >= 30:
        return "difficult"
    return "very difficult"
