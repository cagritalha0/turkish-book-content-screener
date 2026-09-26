"""Feature matrix shared by training and inference.

Keeping one function for both sides is what prevents the train/serve skew the
original notebook had (6 features at training time, 5 in the app).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .lexicon import lexicon_features

FEATURE_COLUMNS = [
    "offensive_count",
    "has_offensive",
    "offensive_ratio",
    "bert_offensive_prob",
]


def build_features(texts: Sequence[str], bert_probs: np.ndarray, lexicon: frozenset[str]) -> pd.DataFrame:
    if len(texts) != len(bert_probs):
        raise ValueError(f"{len(texts)} texts but {len(bert_probs)} BERT probabilities")
    rows = [lexicon_features(t, lexicon) for t in texts]
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS[:3])
    frame["bert_offensive_prob"] = np.asarray(bert_probs, dtype=float)
    return frame[FEATURE_COLUMNS]
