"""
Build the MVP scenario dataset end to end:

  1. Generate IID scenarios (used for train/validation/test_iid) and OOD
     scenarios (used only for test_ood) from disjoint pools.
  2. Run schema-level + registry-level validation on every scenario.
  3. Split IID scenarios 60/20/10 (train/validation/test_iid) — these
     ratios are of the GRAND TOTAL, with OOD fixed at 10%.
  4. Write JSONL files to data/.
  5. Print a class-balance report (by decision and by domain).

Run: PYTHONPATH=. python3 scripts/build_dataset.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from argus.scenarios.generators import GENERATORS_IID, generate_dataset
from argus.scenarios.schema import Scenario
from argus.scenarios.validators import validate_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_PER_CATEGORY_IID = 50   # default for the 8 already-saturated categories
N_PER_CATEGORY_OOD = 6

# delete_decision is the only category with injected label noise (see
# generators.py) — it needs far more examples than the deterministic
# categories for error analysis and CIs to mean anything on ~6 samples.
CATEGORY_OVERRIDES_IID = {"generate_delete_file_decision": 180}
# Chosen so delete_decision is ~31% of OOD too (180/580 in IID), matching
# proportions instead of accidentally testing a different class mix.
CATEGORY_OVERRIDES_OOD = {"generate_delete_file_decision": 22}
SEED = 42


def split_iid(scenarios: list[Scenario], seed: int) -> dict[str, list[Scenario]]:
    """60/20/10 split of the total 90% IID budget -> within the IID pool
    itself that's 6/2/1 (train:val:test_iid)."""
    rng = random.Random(seed)
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 6 / 9)
    n_val = int(n * 2 / 9)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test_iid = shuffled[n_train + n_val:]
    return {"train": train, "validation": val, "test_iid": test_iid}


def write_jsonl(scenarios: list[Scenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n")


def report_balance(name: str, scenarios: list[Scenario]) -> None:
    decisions = Counter(s.gold_decision.value for s in scenarios)
    domains = Counter(s.domain for s in scenarios)
    print(f"\n-- {name} (n={len(scenarios)}) --")
    print(f"   decisions: {dict(decisions)}")
    print(f"   domains  : {dict(domains)}")


def build_splits(seed: int) -> dict[str, list[Scenario]]:
    """Generate + validate + split one full dataset for a given seed.
    Raises SystemExit if validation fails. No file I/O — used both by
    main() (which writes JSONL) and scripts/run_multi_seed.py (which
    doesn't need files, just the in-memory Scenario objects, run many
    times over different seeds)."""
    iid_scenarios = generate_dataset(N_PER_CATEGORY_IID, seed=seed, ood=False,
                                      overrides=CATEGORY_OVERRIDES_IID)
    ood_scenarios = generate_dataset(N_PER_CATEGORY_OOD, seed=seed + 1, ood=True,
                                      overrides=CATEGORY_OVERRIDES_OOD)

    all_scenarios = iid_scenarios + ood_scenarios
    failures = validate_dataset(all_scenarios)
    if failures:
        raise SystemExit(f"{len(failures)} scenario(s) failed validation for "
                          f"seed={seed}: {list(failures.items())[:5]}")

    splits = split_iid(iid_scenarios, seed=seed)
    splits["test_ood"] = ood_scenarios
    for split_name, split_scenarios in splits.items():
        for s in split_scenarios:
            s.split = split_name
    return splits


def main() -> None:
    print(f"Generating {N_PER_CATEGORY_IID} scenarios/category "
          f"across {len(GENERATORS_IID)} categories for IID pool "
          f"(overrides: {CATEGORY_OVERRIDES_IID})...")
    print(f"Generating {N_PER_CATEGORY_OOD} scenarios/category for OOD pool "
          f"(overrides: {CATEGORY_OVERRIDES_OOD})...")

    splits = build_splits(SEED)
    total = sum(len(v) for v in splits.values())
    print(f"\nTotal generated: {total}")
    print("All scenarios passed validation.")

    for split_name, split_scenarios in splits.items():
        write_jsonl(split_scenarios, DATA_DIR / f"{split_name}.jsonl")
        report_balance(split_name, split_scenarios)

    print(f"\nWrote {total} scenarios to {DATA_DIR}/")
    for split_name, split_scenarios in splits.items():
        pct = 100 * len(split_scenarios) / total
        print(f"  {split_name:12s}: {len(split_scenarios):4d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
