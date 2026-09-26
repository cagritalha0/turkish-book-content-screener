"""Loading the Turkish offensive-language dataset (Toygar, Kaggle / Hugging Face)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .text import clean_social_text

SPLITS = ("train", "valid", "test")


def load_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    """Read ``<data_dir>/<split>.csv`` with columns ``id, text, label`` and add ``text_clean``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = Path(data_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see data/README.md for how to download the dataset.")
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["text_clean"] = df["text"].map(clean_social_text)
    return df
