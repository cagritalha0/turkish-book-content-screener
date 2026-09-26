"""Human-readable reasons for a flagged book.

Categories describe *content themes* (violence, drugs, sexual content, profanity,
crime). Identity terms (ethnicity, religion, nationality, orientation, disability)
are deliberately excluded: mentioning a group is not inappropriate content.
"""

from __future__ import annotations

from collections import Counter

from .text import tokenize_words

CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "violence_and_war": frozenset(
        "savaş katliam soykırım terör terörist bomba bombalamak silah tabanca tüfek "
        "mermi kurşun intihar işkence cinayet öldürmek vurmak dövmek".split()
    ),
    "drugs_and_addiction": frozenset(
        "sigara esrar uyuşturucu kokain eroin bonzai amfetamin metamfetamin marihuana torbacı".split()
    ),
    "sexual_content": frozenset("seks cinsel cinsellik çıplak penis vajina tecavüz taciz sübyancı pedofili".split()),
    "profanity_and_insults": frozenset(
        "lan ulan şerefsiz salak aptal budala gerizekalı yavşak puşt pezevenk orospu "
        "piç kaltak bok göt am amına sik sikmek yarrak taşak".split()
    ),
    "crime": frozenset("hırsız gasp şantaj santaj dolandırıcı kaçakçı katil kundaklama kapkaç rüşvet".split()),
}

CATEGORY_LABELS = {
    "violence_and_war": "Violence / war",
    "drugs_and_addiction": "Drugs / addiction",
    "sexual_content": "Sexual content",
    "profanity_and_insults": "Profanity / insults",
    "crime": "Crime",
}


def category_hits(text: str) -> Counter:
    """Count keyword hits per category using whole-token matching."""
    hits: Counter = Counter()
    for token in tokenize_words(text):
        for category, words in CATEGORY_KEYWORDS.items():
            if token in words:
                hits[category] += 1
    return hits
