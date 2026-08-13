"""
Repeat the full pipeline (generate dataset -> train -> evaluate) across
multiple random seeds, in-process (no file I/O per seed, no subprocesses)
so it's fast enough to run 10+ repetitions.

For each seed, a fresh dataset/train/validation/test split is generated,
fresh models are trained, and metrics are computed on test_iid and
test_ood. We then report mean +/- 95% CI per governor per split per
metric, so a single run's numbers (like the one that showed rule_based
beating logistic by 2 points) can be judged against the actual
seed-to-seed variance instead of over-interpreted.

Run: PYTHONPATH=. python3 scripts/run_multi_seed.py
(Takes longer than the single-run scripts — expect ~10x the runtime of
train_baselines.py + evaluate_baselines.py combined, for N_SEEDS=10.)
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from argus.evaluation.metrics import compute_all_metrics
from argus.features.extractor import extract_batch
from argus.governors.rule_based import RuleBasedGovernor
from argus.scenarios.taxonomy import GovernorDecision

from build_dataset import build_splits

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_SEEDS = 10
BASE_SEED = 1000  # offset from the single-run SEED=42 so runs don't overlap

METRICS_TO_TRACK = ["accuracy", "macro_f1", "unsafe_execution_rate",
                    "false_intervention_rate", "severity_weighted_error"]


def train_models(train, validation):
    X_train_dicts = extract_batch(train)
    y_train = [s.gold_decision.value for s in train]

    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(X_train_dicts)

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    logreg.fit(X_train, y_train)

    try:
        from xgboost import XGBClassifier
        label_to_idx = {"execute": 0, "ask": 1, "block": 2}
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        y_train_idx = [label_to_idx[y] for y in y_train]
        xgb = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                             num_class=3, random_state=42)
        xgb.fit(X_train, y_train_idx)
        xgb_bundle = {"model": xgb, "backend": "xgboost", "idx_to_label": idx_to_label}
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(random_state=42)
        hgb.fit(X_train, y_train)
        xgb_bundle = {"model": hgb, "backend": "histgb"}

    return vectorizer, logreg, xgb_bundle


def predict(scenarios, vectorizer, model_bundle_or_model) -> list[GovernorDecision]:
    feats = extract_batch(scenarios)
    X = vectorizer.transform(feats)
    if isinstance(model_bundle_or_model, dict):
        model = model_bundle_or_model["model"]
        if model_bundle_or_model.get("backend") == "xgboost":
            idx_to_label = model_bundle_or_model["idx_to_label"]
            preds_idx = model.predict(X)
            return [GovernorDecision(idx_to_label[int(i)]) for i in preds_idx]
        preds = model.predict(X)
        return [GovernorDecision(p) for p in preds]
    preds = model_bundle_or_model.predict(X)
    return [GovernorDecision(p) for p in preds]


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% CI using a normal approximation
    (fine for N_SEEDS >= 10; for smaller N consider a t-distribution)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    half_width = 1.96 * std / math.sqrt(n)
    return mean, half_width


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # results[governor][split][metric] = list of values, one per seed
    results: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for i in range(N_SEEDS):
        seed = BASE_SEED + i
        print(f"\n=== Seed {i + 1}/{N_SEEDS} (seed={seed}) ===")
        splits = build_splits(seed)
        vectorizer, logreg, xgb_bundle = train_models(splits["train"], splits["validation"])

        governors = {
            "rule_based": ("rule", RuleBasedGovernor()),
            "logistic": ("sklearn", logreg),
            "xgboost": ("sklearn", xgb_bundle),
        }

        for split_name in ["test_iid", "test_ood"]:
            scenarios = splits[split_name]
            gold = [s.gold_decision for s in scenarios]

            rb_preds = governors["rule_based"][1].decide_batch(extract_batch(scenarios))
            log_preds = predict(scenarios, vectorizer, logreg)
            xgb_preds = predict(scenarios, vectorizer, xgb_bundle)

            for gov_name, preds in [("rule_based", rb_preds), ("logistic", log_preds),
                                     ("xgboost", xgb_preds)]:
                m = compute_all_metrics(gold, preds)
                for metric_name in METRICS_TO_TRACK:
                    results[gov_name][split_name][metric_name].append(m[metric_name])
                print(f"  {gov_name:12s} {split_name:10s} acc={m['accuracy']:.3f} "
                      f"UER={m['unsafe_execution_rate']:.3f}")

    print("\n" + "=" * 78)
    print(f"SUMMARY across {N_SEEDS} seeds: mean [95% CI]")
    print("=" * 78)

    summary_out = {}
    for gov_name in ["rule_based", "logistic", "xgboost"]:
        summary_out[gov_name] = {}
        for split_name in ["test_iid", "test_ood"]:
            print(f"\n{gov_name} — {split_name}:")
            summary_out[gov_name][split_name] = {}
            for metric_name in METRICS_TO_TRACK:
                values = results[gov_name][split_name][metric_name]
                mean, ci = mean_ci(values)
                print(f"    {metric_name:26s}: {mean:.3f} +/- {ci:.3f}   (values: "
                      f"{[round(v, 3) for v in values]})")
                summary_out[gov_name][split_name][metric_name] = {
                    "mean": mean, "ci95": ci, "values": values,
                }

    with open(RESULTS_DIR / "multi_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved -> {RESULTS_DIR / 'multi_seed_summary.json'}")

    print("\n" + "=" * 78)
    print("Does any governor's edge survive multiple seeds? (checking BOTH splits)")
    print("=" * 78)
    for split_name in ["test_iid", "test_ood"]:
        print(f"\n  -- {split_name} --")
        means_cis = {}
        for gov_name in ["rule_based", "logistic", "xgboost"]:
            acc = results[gov_name][split_name]["accuracy"]
            mean, ci = mean_ci(acc)
            means_cis[gov_name] = (mean, ci)
            print(f"  {gov_name:12s}: {mean:.3f} +/- {ci:.3f}")
        # Flag any governor whose CI doesn't overlap with the best other governor's CI
        best_name = max(means_cis, key=lambda k: means_cis[k][0])
        for gov_name, (mean, ci) in means_cis.items():
            if gov_name == best_name:
                continue
            best_mean, best_ci = means_cis[best_name]
            if (mean + ci) < (best_mean - best_ci):
                print(f"  -> {gov_name}'s gap below {best_name} appears REAL "
                      f"(CIs don't overlap).")
        if all((means_cis[a][0] - means_cis[a][1]) <= (means_cis[b][0] + means_cis[b][1])
               for a in means_cis for b in means_cis):
            print(f"  -> All CIs overlap on {split_name}: no statistically "
                  f"distinguishable winner.")


if __name__ == "__main__":
    main()
