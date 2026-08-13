"""
Train Baseline 2 (Logistic Regression) and Baseline 3 (XGBoost, falling
back to HistGradientBoosting if xgboost isn't installed) on the MVP
dataset's train split, using validation for early sanity checks.

Pipeline: JSONL -> Feature Extractor -> DictVectorizer -> Classifier

Run: PYTHONPATH=. python3 scripts/train_baselines.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from argus.features.extractor import extract_batch
from argus.scenarios.schema import Scenario

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_split(name: str) -> list[Scenario]:
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Scenario.model_validate(json.loads(line)) for line in f]


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    validation = load_split("validation")
    print(f"Loaded train={len(train)}, validation={len(validation)}")

    X_train_dicts = extract_batch(train)
    y_train = [s.gold_decision.value for s in train]
    X_val_dicts = extract_batch(validation)
    y_val = [s.gold_decision.value for s in validation]

    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(X_train_dicts)
    X_val = vectorizer.transform(X_val_dicts)

    # -- Baseline 2: Logistic Regression (scaled — our features mix 0/1
    # flags with small integer counts like request_length, which can
    # destabilize an unscaled linear model) ---------------------------
    print("\nTraining Logistic Regression...")
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    logreg.fit(X_train, y_train)
    val_acc = logreg.score(X_val, y_val)
    print(f"  validation accuracy: {val_acc:.3f}")

    with open(MODELS_DIR / "logistic.pkl", "wb") as f:
        pickle.dump({"model": logreg, "vectorizer": vectorizer}, f)
    print(f"  saved -> {MODELS_DIR / 'logistic.pkl'}")

    # -- Baseline 3: XGBoost (or HistGradientBoosting fallback) ----------
    print("\nTraining Baseline 3...")
    try:
        from xgboost import XGBClassifier

        label_to_idx = {"execute": 0, "ask": 1, "block": 2}
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        y_train_idx = [label_to_idx[y] for y in y_train]
        y_val_idx = [label_to_idx[y] for y in y_val]

        xgb = XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            num_class=3, random_state=42,
        )
        xgb.fit(X_train, y_train_idx)
        val_acc_xgb = (xgb.predict(X_val) == y_val_idx).mean()
        print(f"  (xgboost) validation accuracy: {val_acc_xgb:.3f}")

        with open(MODELS_DIR / "xgboost.pkl", "wb") as f:
            pickle.dump({
                "model": xgb, "vectorizer": vectorizer,
                "label_to_idx": label_to_idx, "idx_to_label": idx_to_label,
                "backend": "xgboost",
            }, f)
        print(f"  saved -> {MODELS_DIR / 'xgboost.pkl'}")

    except ImportError:
        print("  xgboost not installed, falling back to HistGradientBoostingClassifier")
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(random_state=42)
        hgb.fit(X_train, y_train)
        val_acc_hgb = hgb.score(X_val, y_val)
        print(f"  (histgradientboosting) validation accuracy: {val_acc_hgb:.3f}")

        with open(MODELS_DIR / "xgboost.pkl", "wb") as f:
            pickle.dump({"model": hgb, "vectorizer": vectorizer, "backend": "histgb"}, f)
        print(f"  saved -> {MODELS_DIR / 'xgboost.pkl'}")

    print("\nDone. Run scripts/evaluate_baselines.py next.")


if __name__ == "__main__":
    main()
