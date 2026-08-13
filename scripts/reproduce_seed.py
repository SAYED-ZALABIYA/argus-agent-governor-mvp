"""
Rebuild ONE specific seed end-to-end (dataset -> train -> evaluate) and
write per-scenario predictions to CSV, so an outlier seed spotted in the
multi-seed summary (e.g. seed=1001, where logistic's OOD accuracy and
false_intervention_rate both spiked) can be inspected case by case
instead of only seen as an aggregate number.

Run: PYTHONPATH=. python3 scripts/reproduce_seed.py 1001
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import build_splits
from run_multi_seed import predict, train_models

from argus.features.extractor import extract_batch
from argus.governors.rule_based import RuleBasedGovernor

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/reproduce_seed.py <seed>")
        sys.exit(1)
    seed = int(sys.argv[1])

    print(f"Rebuilding seed={seed}...")
    splits = build_splits(seed)
    vectorizer, logreg, xgb_bundle = train_models(splits["train"], splits["validation"])
    rb = RuleBasedGovernor()

    # -- Coefficient inspection: does logistic lean on category-identity
    # features (tool_delete_file, risk_high, domain_files — constant
    # within delete_decision) instead of the two real signals
    # (target_file_recently_modified, target_file_referenced_elsewhere)?
    feature_names = vectorizer.get_feature_names_out()
    scaler = logreg.named_steps["standardscaler"]
    classifier = logreg.named_steps["logisticregression"]
    block_class_idx = list(classifier.classes_).index("block")
    coefs = classifier.coef_[block_class_idx]
    # Coefficients are on SCALED features; report both raw coef and
    # coef/scale (roughly "effect per unit of the original 0/1 feature").
    ranked = sorted(
        zip(feature_names, coefs, scaler.scale_),
        key=lambda t: abs(t[1]), reverse=True,
    )
    print("\n  Logistic 'block'-class coefficients (top 10 by |weight|, scaled features):")
    for name, coef, scale in ranked[:10]:
        flag = "  <-- category-identity feature, NOT a real signal" if name in (
            "tool_delete_file", "risk_high", "domain_files", "is_reversible",
            "proposed_action_is_destructive",
        ) else ""
        print(f"    {name:38s} coef={coef:+.3f}{flag}")
    for target_feat in ["target_file_recently_modified", "target_file_referenced_elsewhere"]:
        match = [t for t in ranked if t[0] == target_feat]
        if match:
            rank = ranked.index(match[0]) + 1
            print(f"    -> {target_feat}: coef={match[0][1]:+.3f}, rank #{rank} of {len(ranked)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["test_iid", "test_ood"]:
        scenarios = splits[split_name]
        rb_preds = rb.decide_batch(extract_batch(scenarios))
        log_preds = predict(scenarios, vectorizer, logreg)
        xgb_preds = predict(scenarios, vectorizer, xgb_bundle)

        out_path = RESULTS_DIR / f"seed_{seed}_{split_name}_predictions.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario_id", "domain", "user_request", "gold_decision",
                              "rule_based", "logistic", "xgboost",
                              "target_file_recently_modified", "target_file_referenced_elsewhere"])
            feats = extract_batch(scenarios)
            for i, s in enumerate(scenarios):
                writer.writerow([
                    s.scenario_id, s.domain, s.user_request, s.gold_decision.value,
                    rb_preds[i].value, log_preds[i].value, xgb_preds[i].value,
                    feats[i].get("target_file_recently_modified"),
                    feats[i].get("target_file_referenced_elsewhere"),
                ])
        print(f"  wrote {out_path}")

        # Print logistic-specific mismatches inline for quick inspection
        mismatches = [
            (s, g, p) for s, g, p in zip(scenarios, [s.gold_decision for s in scenarios], log_preds)
            if g != p
        ]
        if mismatches:
            print(f"\n  logistic mismatches on {split_name} (n={len(mismatches)}):")
            for s, gold, pred in mismatches:
                print(f"    {s.scenario_id:28s} gold={gold.value:8s} pred={pred.value:8s} "
                      f"| {s.user_request}")


if __name__ == "__main__":
    main()
