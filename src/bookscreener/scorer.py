"""Sentence-level offensive-content scorer backed by a fine-tuned BERTurk model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

BASE_MODEL = "dbmdz/bert-base-turkish-128k-uncased"
MAX_LENGTH = 128


class SentenceScorer(Protocol):
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return P(offensive) for each text, shape ``(len(texts),)``."""


class BertScorer:
    """Batched inference wrapper around a fine-tuned ``BertForSequenceClassification``."""

    def __init__(self, model_dir: str, device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        probs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(
                self.device
            )
            with torch.inference_mode():
                logits = self.model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
        return np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
