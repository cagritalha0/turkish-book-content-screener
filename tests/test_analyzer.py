import numpy as np
import pytest

from bookscreener.analyzer import BookAnalyzer

LEX = frozenset({"salak"})


class KeywordScorer:
    """Stand-in for BERT: a sentence is risky if it contains 'salak' or 'silah'."""

    def predict_proba(self, texts):
        return np.array([0.95 if ("salak" in t or "silah" in t) else 0.05 for t in texts])


class FlipMeta:
    """Stand-in meta-model that inverts BERT, to prove the meta path is used."""

    def predict_proba(self, X):
        p = 1 - X["bert_offensive_prob"].to_numpy()
        return np.column_stack([1 - p, p])


BOOK = "Kedi bahçede oynadı. Güneş parlıyordu. Adam silah çekti. Kuşlar uçtu."


def test_clean_book_is_suitable():
    report = BookAnalyzer(KeywordScorer(), LEX).analyze("Kedi oynadı. Güneş doğdu.")
    assert not report.flagged
    assert report.verdict == "Suitable"
    assert report.risky_sentences == []


def test_risky_book_is_flagged_with_reasons():
    report = BookAnalyzer(KeywordScorer(), LEX).analyze(BOOK)
    assert report.total_sentences == 4
    assert report.risky_sentences == ["Adam silah çekti."]
    assert report.risk_ratio == pytest.approx(0.25)
    assert report.flagged
    assert "Violence / war" in report.reasons


def test_scorer_receives_turkish_lowercased_text():
    seen = []

    class Spy(KeywordScorer):
        def predict_proba(self, texts):
            seen.extend(texts)
            return super().predict_proba(texts)

    BookAnalyzer(Spy(), LEX).analyze("SALAK IRMAK.")
    assert seen == ["salak ırmak."]


def test_meta_model_is_applied():
    report = BookAnalyzer(KeywordScorer(), LEX, meta_model=FlipMeta()).analyze(BOOK)
    assert "Adam silah çekti." not in report.risky_sentences
    assert len(report.risky_sentences) == 3


def test_empty_document_raises():
    with pytest.raises(ValueError):
        BookAnalyzer(KeywordScorer(), LEX).analyze("   ")
