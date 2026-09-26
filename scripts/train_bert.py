"""Fine-tune BERTurk for binary offensive-language classification.

A stratified slice of the training set is held out and never shown to BERT, so the
CatBoost meta-model (scripts/train_meta.py) can be trained on honest,
out-of-sample BERT scores.

    python scripts/train_bert.py --data-dir data/raw --output-dir models/berturk-offensive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from bookscreener.data import load_split
from bookscreener.scorer import BASE_MODEL, MAX_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="models/berturk-offensive")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--meta-holdout", type=float, default=0.1, help="Share of train kept for the meta-model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.data_dir, "train")
    valid_df = load_split(args.data_dir, "valid")
    bert_idx, meta_idx = train_test_split(
        np.arange(len(train_df)), test_size=args.meta_holdout, stratify=train_df["label"], random_state=args.seed
    )
    (out / "meta_holdout_indices.json").write_text(json.dumps(sorted(meta_idx.tolist())))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    def to_dataset(frame):
        ds = Dataset.from_pandas(
            frame[["text_clean", "label"]].rename(columns={"label": "labels"}), preserve_index=False
        )
        return ds.map(
            lambda b: tokenizer(b["text_clean"], truncation=True, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=["text_clean"],
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=to_dataset(train_df.iloc[bert_idx]),
        eval_dataset=to_dataset(valid_df),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "valid_metrics.json").write_text(json.dumps(trainer.evaluate(), indent=2))
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
