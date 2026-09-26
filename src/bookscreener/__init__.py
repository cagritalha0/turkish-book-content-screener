"""Screen Turkish books for content that may be unsuitable for children."""

from .analyzer import BookAnalyzer, BookReport
from .lexicon import load_lexicon

__all__ = ["BookAnalyzer", "BookReport", "load_lexicon"]
__version__ = "1.0.0"
