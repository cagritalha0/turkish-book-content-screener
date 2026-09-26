import numpy as np
import pytest

from bookscreener.categories import category_hits
from bookscreener.features import FEATURE_COLUMNS, build_features
from bookscreener.lexicon import lexicon_features, load_lexicon

LEX = frozenset({"salak", "am", "savaş"})


def test_whole_token_matching_avoids_substring_false_positives():
    # the original notebook used `word in text`, so "am" matched "ama" / "adam"
    assert lexicon_features("Ama adam çok iyiydi.", LEX)["offensive_count"] == 0
    assert category_hits("Ama adam çok iyiydi.") == {}


def test_lexicon_features():
    feats = lexicon_features("SALAK misin sen", LEX)
    assert feats == {"offensive_count": 1, "has_offensive": 1, "offensive_ratio": pytest.approx(1 / 3)}


def test_category_hits_counts_themes():
    hits = category_hits("Savaş başladı, silah sesleri geldi. Salak dedi.")
    assert hits["violence_and_war"] == 2
    assert hits["profanity_and_insults"] == 1


def test_shipped_lexicon_has_no_identity_terms():
    lexicon = load_lexicon()
    identity_terms = {"yahudi", "ermeni", "kürt", "arap", "çingene", "ateist", "gay", "engelli", "alman"}
    assert lexicon, "lexicon should not be empty"
    assert not identity_terms & lexicon


def test_build_features_column_order_and_length_check():
    frame = build_features(["salak", "iyi"], np.array([0.9, 0.1]), LEX)
    assert list(frame.columns) == FEATURE_COLUMNS
    with pytest.raises(ValueError):
        build_features(["a"], np.array([0.1, 0.2]), LEX)
