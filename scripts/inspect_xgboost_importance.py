"""
Diagnostic: why did XGBoost catch 5/5 H1 injection variants that don't
match any known injection phrase, while rule_based (which checks the
same prompt_injection_detected feature plus a few others, but NEVER
reads satisfied_precondition_ratio / missing_precondition_count) caught
only 1/5?

Hypothesis: every H1 scenario has missing_preconditions=
["instruction_from_trusted_source"], which no EXECUTE-gold training
example ever has — so satisfied_precondition_ratio is 0.5-0.67 for H1
instead of the 1.0 XGBoost saw for every safe EXECUTE case in training.
If XGBoost ranks that feature highly, it's plausibly using "some
precondition is unmet" as a general caution signal (legitimate!) rather
than actually understanding the injection semantics — which rule_based
can't do at all because it doesn't consult that feature.

Run: PYTHONPATH=. python3 scripts/inspect_xgboost_importance.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

WATCH_FEATURES = [
    "satisfied_precondition_ratio", "missing_precondition_count",
    "prompt_injection_detected", "target_file_recently_modified",
    "target_file_referenced_elsewhere", "has_required_permission",
    "schema_valid", "ambiguous_entity_count",
]


def main() -> None:
    with open(MODELS_DIR / "xgboost.pkl", "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_names = list(bundle["vectorizer"].get_feature_names_out())

    if bundle.get("backend") == "xgboost":
        importances = model.feature_importances_
    else:
        # HistGradientBoostingClassifier doesn't expose feature_importances_
        # the same way; fall back to permutation-free approx via
        # get_booster-style attr if present, else note unavailable.
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            print("This backend doesn't expose feature_importances_ directly. "
                  "Re-run with the real xgboost package installed for this "
                  "diagnostic (pip install xgboost).")
            return

    ranked = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)

    print("Top 15 features by global importance:")
    for i, (name, imp) in enumerate(ranked[:15], 1):
        flag = "  <-- WATCHING" if name in WATCH_FEATURES else ""
        print(f"  {i:2d}. {name:38s} importance={imp:.4f}{flag}")

    print("\nWatched features (precondition-completeness + injection signals) and their rank:")
    for name in WATCH_FEATURES:
        match = [t for t in ranked if t[0] == name]
        if match:
            rank = ranked.index(match[0]) + 1
            print(f"  {name:38s} importance={match[0][1]:.4f}  rank #{rank} of {len(ranked)}")
        else:
            print(f"  {name:38s} NOT FOUND in vectorizer output")


if __name__ == "__main__":
    main()