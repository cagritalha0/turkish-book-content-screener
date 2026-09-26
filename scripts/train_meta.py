"""Train the CatBoost meta-model on BERT scores + lexicon features.

Uses only the meta hold-out rows written by train_bert.py, i.e. text BERT never
trained on. (The original notebook fitted CatBoost on BERT's *training-set*
predictions, which are over-confident, so the stack could not beat BERT alone.)

    python scripts/train_meta.py --data-dir data/raw --model-dir models/berturk-offensive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from bookscreener.data import load_split
from bookscreener.features import FEATURE_COLUMNS, build_features
from bookscreener.lexicon import load_lexicon
from bookscreener.scorer import BertScorer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models/berturk-offensive")
    parser.add_argument("--output", default="models/meta_catboost.cbm")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import optuna
    from catboost import CatBoostClassifier

    train_df = load_split(args.data_dir, "train")
    meta_idx = json.loads((Path(args.model_dir) / "meta_holdout_indices.json").read_text())
    meta_df = train_df.iloc[meta_idx].reset_index(drop=True)

    scorer = BertScorer(args.model_dir)
    lexicon = load_lexicon()
    X = build_features(meta_df["text_clean"].tolist(), scorer.predict_proba(meta_df["text_clean"].tolist()), lexicon)
    y = meta_df["label"].to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    def cv_f1(params: dict) -> float:
        scores = []
        for tr, va in folds.split(X, y):
            model = CatBoostClassifier(**params, loss_function="Logloss", random_seed=args.seed, verbose=False)
            model.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]), early_stopping_rounds=50)
            scores.append(f1_score(y[va], model.predict(X.iloc[va])))
        return float(np.mean(scores))

    def objective(trial):
        return cv_f1(
            {
                "iterations": trial.suggest_int("iterations", 200, 1000),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
            }
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    print("Best CV F1:", round(study.best_value, 4), study.best_params)

    final = CatBoostClassifier(**study.best_params, loss_function="Logloss", random_seed=args.seed, verbose=False)
    final.fit(X, y)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final.save_model(args.output)  # the real, trained model - not a placeholder
    Path(args.output).with_suffix(".json").write_text(
        json.dumps({"features": FEATURE_COLUMNS, "cv_f1": study.best_value, "params": study.best_params}, indent=2)
    )
    print(f"Saved meta-model to {args.output}")


if __name__ == "__main__":
    main()
