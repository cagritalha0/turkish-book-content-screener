"""Evaluate BERT alone and BERT + CatBoost on the held-out test split.

    python scripts/evaluate.py --data-dir data/raw --model-dir models/berturk-offensive \
        [--meta-model models/meta_catboost.cbm] --out-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from bookscreener.data import load_split  # noqa: E402
from bookscreener.features import build_features  # noqa: E402
from bookscreener.lexicon import load_lexicon  # noqa: E402
from bookscreener.scorer import BertScorer  # noqa: E402


def summarise(name: str, y: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    print(f"\n=== {name} ===\n" + classification_report(y, pred, digits=4))
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_offensive": f1_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, prob),
    }


def plot(name: str, y: np.ndarray, prob: np.ndarray, out: Path) -> None:
    ConfusionMatrixDisplay.from_predictions(
        y, (prob >= 0.5).astype(int), display_labels=["not offensive", "offensive"], cmap="Blues", colorbar=False
    )
    plt.title(f"{name}: confusion matrix")
    plt.savefig(out / f"{name}_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, prob):.3f}")
    a.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    a.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    a.legend(loc="lower right")
    b.plot(recall, precision)
    b.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    fig.tight_layout()
    fig.savefig(out / f"{name}_curves.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models/berturk-offensive")
    parser.add_argument("--meta-model")
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_df = load_split(args.data_dir, "test")
    texts, y = test_df["text_clean"].tolist(), test_df["label"].to_numpy()

    bert_prob = BertScorer(args.model_dir).predict_proba(texts)
    results = {"bert": summarise("bert", y, bert_prob)}
    plot("bert", y, bert_prob, out)

    if args.meta_model:
        from catboost import CatBoostClassifier

        meta = CatBoostClassifier()
        meta.load_model(args.meta_model)
        stack_prob = meta.predict_proba(build_features(texts, bert_prob, load_lexicon()))[:, 1]
        results["stack"] = summarise("stack", y, stack_prob)
        plot("stack", y, stack_prob, out)

    (out / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
