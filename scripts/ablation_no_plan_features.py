"""
Ablation test: does removing plan_step_count / argument_count collapse
XGBoost's H1 (prompt injection variants) performance back down to
rule_based's level?

Hypothesis under test: every hand-written injection scenario (both in
training via generate_block_prompt_injection, and in the H1 hard-OOD
set) has agent_plan = ["read X", "follow embedded instruction: ..."],
i.e. 2 steps — an authorial habit, not a property a real agent's plan
would reliably have. If XGBoost's H1 success depends on plan_step_count
distinguishing "2 steps" from the 1-step plans in legitimate
delete_decision EXECUTE examples, that's exploiting how *I* write
scenarios, not genuine injection understanding.

Run: PYTHONPATH=. python3 scripts/ablation_no_plan_features.py
(Run scripts/build_dataset.py first if data/ doesn't exist.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.feature_extraction import DictVectorizer

from argus.evaluation.metrics import compute_all_metrics, print_report
from argus.features.extractor import extract_batch
from argus.scenarios.hard_ood import H1_INJECTION_VARIANTS, H2_COMPOUND_FAILURES
from argus.scenarios.schema import Scenario
from argus.scenarios.taxonomy import GovernorDecision

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXCLUDED_FEATURES = {"plan_step_count", "argument_count"}


def load_split(name: str) -> list[Scenario]:
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Scenario.model_validate(json.loads(line)) for line in f]


def strip_features(feature_dicts: list[dict]) -> list[dict]:
    return [{k: v for k, v in d.items() if k not in EXCLUDED_FEATURES} for d in feature_dicts]


def main() -> None:
    train = load_split("train")
    print(f"Loaded train={len(train)}")
    print(f"Excluding features: {EXCLUDED_FEATURES}\n")

    X_train_dicts = strip_features(extract_batch(train))
    y_train = [s.gold_decision.value for s in train]

    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(X_train_dicts)

    try:
        from xgboost import XGBClassifier
        label_to_idx = {"execute": 0, "ask": 1, "block": 2}
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        y_train_idx = [label_to_idx[y] for y in y_train]
        model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                               num_class=3, random_state=42)
        model.fit(X_train, y_train_idx)
        backend = "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        backend = "histgb"
        idx_to_label = None

    def predict(scenarios: list[Scenario]) -> list[GovernorDecision]:
        feats = strip_features(extract_batch(scenarios))
        X = vectorizer.transform(feats)
        if backend == "xgboost":
            preds_idx = model.predict(X)
            return [GovernorDecision(idx_to_label[int(i)]) for i in preds_idx]
        return [GovernorDecision(p) for p in model.predict(X)]

    for name, scenarios in [("H1: Prompt Injection Variants", H1_INJECTION_VARIANTS),
                              ("H2: Compound Failures", H2_COMPOUND_FAILURES)]:
        gold = [s.gold_decision for s in scenarios]
        preds = predict(scenarios)
        print(f"\n--- {name} (ablated model, no plan/argument-count features) ---")
        for s, g, p in zip(scenarios, gold, preds):
            mark = "OK" if g == p else "MISS"
            print(f"  [{mark:4s}] {s.scenario_id:32s} gold={g.value:8s} pred={p.value:8s}")
        m = compute_all_metrics(gold, preds)
        print_report(f"{name} (ablated)", m)


if __name__ == "__main__":
    main()
