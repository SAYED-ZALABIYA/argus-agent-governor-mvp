"""
Evaluate all three baselines (rule-based, logistic, xgboost) on both
test_iid and test_ood, using the metrics defined in argus/evaluation/metrics.py.

Run: PYTHONPATH=. python3 scripts/evaluate_baselines.py
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

from argus.evaluation.metrics import compute_all_metrics, print_report
from argus.features.extractor import extract_batch
from argus.governors.rule_based import RuleBasedGovernor
from argus.scenarios.schema import Scenario
from argus.scenarios.taxonomy import GovernorDecision

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_split(name: str) -> list[Scenario]:
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Scenario.model_validate(json.loads(line)) for line in f]


def predict_rule_based(scenarios: list[Scenario]) -> list[GovernorDecision]:
    gov = RuleBasedGovernor()
    feats = extract_batch(scenarios)
    return gov.decide_batch(feats)


def predict_sklearn_style(scenarios: list[Scenario], model_path: Path) -> list[GovernorDecision]:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    feats = extract_batch(scenarios)
    X = bundle["vectorizer"].transform(feats)
    model = bundle["model"]

    if bundle.get("backend") == "xgboost":
        idx_to_label = bundle["idx_to_label"]
        preds_idx = model.predict(X)
        return [GovernorDecision(idx_to_label[int(i)]) for i in preds_idx]
    else:
        preds = model.predict(X)
        return [GovernorDecision(p) for p in preds]


def write_predictions_csv(scenarios: list[Scenario], predictions: dict[str, list[GovernorDecision]],
                           path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    governor_names = list(predictions.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "domain", "user_request", "gold_decision"] + governor_names)
        for i, s in enumerate(scenarios):
            row = [s.scenario_id, s.domain, s.user_request, s.gold_decision.value]
            row += [predictions[name][i].value for name in governor_names]
            writer.writerow(row)


def evaluate_split(split_name: str, scenarios: list[Scenario]) -> dict:
    gold = [s.gold_decision for s in scenarios]

    predictions = {
        "rule_based": predict_rule_based(scenarios),
        "logistic": predict_sklearn_style(scenarios, MODELS_DIR / "logistic.pkl"),
        "xgboost": predict_sklearn_style(scenarios, MODELS_DIR / "xgboost.pkl"),
    }

    all_metrics = {}
    for gov_name, preds in predictions.items():
        metrics = compute_all_metrics(gold, preds)
        print_report(f"{gov_name} — {split_name}", metrics)
        all_metrics[gov_name] = metrics

    write_predictions_csv(scenarios, predictions, RESULTS_DIR / f"{split_name}_predictions.csv")
    return all_metrics


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_iid = load_split("test_iid")
    test_ood = load_split("test_ood")

    print("#" * 70)
    print("# EVALUATION: test_iid")
    print("#" * 70)
    iid_metrics = evaluate_split("iid", test_iid)

    print("\n" + "#" * 70)
    print("# EVALUATION: test_ood")
    print("#" * 70)
    ood_metrics = evaluate_split("ood", test_ood)

    for gov_name in ["rule_based", "logistic", "xgboost"]:
        with open(RESULTS_DIR / f"{gov_name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"test_iid": iid_metrics[gov_name], "test_ood": ood_metrics[gov_name]},
                      f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SUMMARY: IID vs OOD accuracy per governor (does performance collapse OOD?)")
    print("=" * 70)
    for gov_name in ["rule_based", "logistic", "xgboost"]:
        iid_acc = iid_metrics[gov_name]["accuracy"]
        ood_acc = ood_metrics[gov_name]["accuracy"]
        iid_uer = iid_metrics[gov_name]["unsafe_execution_rate"]
        ood_uer = ood_metrics[gov_name]["unsafe_execution_rate"]
        drop = iid_acc - ood_acc
        print(f"  {gov_name:12s}: IID acc={iid_acc:.3f}  OOD acc={ood_acc:.3f}  "
              f"(drop={drop:+.3f})   IID UER={iid_uer:.3f}  OOD UER={ood_uer:.3f}")

    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
