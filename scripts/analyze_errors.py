"""
Analyze mismatches from the most recent scripts/evaluate_baselines.py run
(reads results/iid_predictions.csv and results/ood_predictions.csv).

Breaks errors down by category (parsed from scenario_id) and tool, per
governor. The key question: are errors concentrated in delete_decision
(the one category with injected label noise — EXPECTED) or spread across
other, supposedly-deterministic categories (would mean something is
still leaking/broken elsewhere).

Run: PYTHONPATH=. python3 scripts/analyze_errors.py
(Run scripts/evaluate_baselines.py first so the CSVs exist.)
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GOVERNORS = ["rule_based", "logistic", "xgboost"]


def category_of(scenario_id: str) -> str:
    parts = scenario_id.split("_")
    if parts and parts[0] in {"iid", "ood"}:
        parts = parts[1:]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def analyze_split(split_name: str) -> None:
    path = RESULTS_DIR / f"{split_name}_predictions.csv"
    if not path.exists():
        print(f"  (skipping {split_name}: {path} not found — run evaluate_baselines.py first)")
        return

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n-- {split_name} (n={len(rows)}) --")
    for gov in GOVERNORS:
        mismatches = [r for r in rows if r["gold_decision"] != r[gov]]
        by_category = Counter(category_of(r["scenario_id"]) for r in mismatches)
        by_tool_dir = Counter(
            f"{r['gold_decision']}->{r[gov]}" for r in mismatches
        )
        print(f"\n  {gov}: {len(mismatches)}/{len(rows)} wrong")
        if by_category:
            print(f"    by category: {dict(by_category)}")
            non_delete = {k: v for k, v in by_category.items() if k != "delete_decision"}
            if non_delete:
                print(f"    !! errors OUTSIDE delete_decision (investigate): {non_delete}")
            print(f"    error direction (gold->pred): {dict(by_tool_dir)}")
            print("    example mismatches:")
            for r in mismatches[:5]:
                print(f"      {r['scenario_id']:28s} gold={r['gold_decision']:8s} "
                      f"pred={r[gov]:8s} | {r['user_request']}")


def main() -> None:
    print("Error analysis from the most recent evaluate_baselines.py run")
    analyze_split("iid")
    analyze_split("ood")


if __name__ == "__main__":
    main()
