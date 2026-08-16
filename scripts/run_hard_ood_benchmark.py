"""
ARGUS Hard-OOD Robustness Benchmark — H1 (Prompt Injection Variants) and
H2 (Compound Failures).

Evaluates the ALREADY-TRAINED rule_based and xgboost governors (no
retraining — that would defeat the point) against argus/scenarios/hard_ood.py.

For H1 specifically, also prints which underlying feature fired for each
scenario (prompt_injection_detected vs. contains_sensitive_data /
target_is_public), to distinguish "correctly detected as injection" from
"blocked for an unrelated, coincidental reason."

Run: PYTHONPATH=. python3 scripts/run_hard_ood_benchmark.py
(Run scripts/train_baselines.py first so models/xgboost.pkl exists —
the script still runs rule_based-only if it's missing.)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argus.evaluation.metrics import compute_all_metrics, print_report
from argus.features.extractor import extract_features
from argus.governors.rule_based import RuleBasedGovernor
from argus.scenarios.hard_ood import H1_INJECTION_VARIANTS, H2_COMPOUND_FAILURES, H3_COMBINED_ADVERSARIAL
from argus.scenarios.taxonomy import GovernorDecision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_xgboost_predictor():
    path = MODELS_DIR / "xgboost.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    def predict(features: dict) -> GovernorDecision:
        X = bundle["vectorizer"].transform([features])
        model = bundle["model"]
        if bundle.get("backend") == "xgboost":
            idx = int(model.predict(X)[0])
            return GovernorDecision(bundle["idx_to_label"][idx])
        return GovernorDecision(model.predict(X)[0])

    return predict


def run_set(name: str, scenarios: list, rule_gov: RuleBasedGovernor, xgb_predict) -> dict:
    print(f"\n{'=' * 78}\n{name} (n={len(scenarios)})\n{'=' * 78}")

    gold = [s.gold_decision for s in scenarios]
    rb_preds, xgb_preds = [], []

    for s in scenarios:
        features = extract_features(s)
        rb_decision = rule_gov.decide(features)
        rb_preds.append(rb_decision)

        xgb_decision = xgb_predict(features) if xgb_predict else None
        if xgb_decision is not None:
            xgb_preds.append(xgb_decision)

        rb_mark = "OK" if rb_decision == s.gold_decision else "MISS"
        line = (f"  [{rb_mark:4s}] {s.scenario_id:32s} gold={s.gold_decision.value:8s} "
                f"rule_based={rb_decision.value:8s}")
        if xgb_decision is not None:
            xgb_mark = "OK" if xgb_decision == s.gold_decision else "MISS"
            line += f" xgboost={xgb_decision.value:8s} [{xgb_mark}]"
        # H1-specific: show which underlying signal actually fired
        if "h1" in s.scenario_id:
            line += (f"  | injection_detected={features['prompt_injection_detected']} "
                      f"sensitive_data={features['contains_sensitive_data']} "
                      f"target_public={features['target_is_public']}")
        print(line)

    results = {"rule_based": compute_all_metrics(gold, rb_preds)}
    print_report(f"{name} — rule_based", results["rule_based"])
    if xgb_preds:
        results["xgboost"] = compute_all_metrics(gold, xgb_preds)
        print_report(f"{name} — xgboost", results["xgboost"])
    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rule_gov = RuleBasedGovernor()
    xgb_predict = load_xgboost_predictor()
    if xgb_predict is None:
        print("NOTE: models/xgboost.pkl not found — running rule_based only. "
              "Run scripts/train_baselines.py first for the full comparison.\n")

    h1_results = run_set("H1: Prompt Injection Variants", H1_INJECTION_VARIANTS, rule_gov, xgb_predict)
    h2_results = run_set("H2: Compound Failures", H2_COMPOUND_FAILURES, rule_gov, xgb_predict)
    h3_results = run_set("H3: Combined Adversarial", H3_COMBINED_ADVERSARIAL, rule_gov, xgb_predict)

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for set_name, results in [("H1", h1_results), ("H2", h2_results), ("H3", h3_results)]:
        for gov_name, m in results.items():
            print(f"  {set_name} / {gov_name:12s}: accuracy={m['accuracy']:.3f}  "
                  f"unsafe_execution_rate={m['unsafe_execution_rate']:.3f}")

    with open(RESULTS_DIR / "hard_ood_benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"H1": h1_results, "H2": h2_results, "H3": h3_results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {RESULTS_DIR / 'hard_ood_benchmark.json'}")


if __name__ == "__main__":
    main()
