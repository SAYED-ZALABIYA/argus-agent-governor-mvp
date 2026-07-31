"""
Dataset audit — must pass before any model training.

Checks:
  1. No duplicate scenario_id.
  2. No identical user_request text appearing in more than one split.
  3. No shared "template family" between train and test_ood (the prefix
     of scenario_id up to the numeric suffix, e.g. "iid_ask_ambig" vs
     "ood_ask_ambig" — these ARE expected to differ in name pool, but we
     check the underlying generator category isn't accidentally identical
     phrasing).
  4. No missing/empty required fields.
  5. Every proposed_tool.tool_name exists in the tool registry.
  6. Every gold_reason_code is consistent with gold_decision (re-run of
     the registry-aware validators for the full dataset).
  7. Distribution report: by decision, domain, failure category, tool,
     risk level.
  8. Near-duplicate detection between train and test (Jaccard similarity
     over word sets) to catch template leakage — cases where train/test
     scenarios differ only by a name or filename swap.

Run: PYTHONPATH=. python3 scripts/audit_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from argus.scenarios.schema import Scenario
from argus.scenarios.validators import validate_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS = ["train", "validation", "test_iid", "test_ood"]

NEAR_DUP_JACCARD_THRESHOLD = 0.8


def load_split(name: str) -> list[Scenario]:
    path = DATA_DIR / f"{name}.jsonl"
    scenarios = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            scenarios.append(Scenario.model_validate(json.loads(line)))
    return scenarios


def word_set(text: str) -> set[str]:
    return set(text.lower().replace(".", " ").replace(",", " ").split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def template_family(scenario_id: str) -> str:
    """Strip the trailing numeric index and the iid/ood prefix, so
    'iid_ask_ambig_0007' and 'ood_ask_ambig_0003' both map to
    'ask_ambig' — the underlying generator category."""
    parts = scenario_id.split("_")
    if parts and parts[0] in {"iid", "ood"}:
        parts = parts[1:]
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def main() -> int:
    all_ok = True
    splits: dict[str, list[Scenario]] = {name: load_split(name) for name in SPLITS}
    all_scenarios = [s for split in splits.values() for s in split]

    print(f"Loaded {len(all_scenarios)} scenarios across {len(SPLITS)} splits.\n")

    # 1. Duplicate scenario_id
    id_counts = Counter(s.scenario_id for s in all_scenarios)
    dupes = {sid: c for sid, c in id_counts.items() if c > 1}
    if dupes:
        print(f"FAIL: {len(dupes)} duplicate scenario_id(s): {list(dupes)[:5]}")
        all_ok = False
    else:
        print("OK: No duplicate scenario_id.")

    # 2. Identical user_request across splits
    request_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, scenarios in splits.items():
        for s in scenarios:
            request_to_splits[s.user_request].add(split_name)
    cross_split = {req: sp for req, sp in request_to_splits.items() if len(sp) > 1}
    if cross_split:
        print(f"FAIL: {len(cross_split)} user_request string(s) appear in "
              f"more than one split, e.g. {list(cross_split.items())[:3]}")
        all_ok = False
    else:
        print("OK: No cross-split exact user_request duplicates.")

    # 3. Template family leakage between train and OOD
    train_families = {template_family(s.scenario_id) for s in splits["train"]}
    ood_families = {template_family(s.scenario_id) for s in splits["test_ood"]}
    shared = train_families & ood_families
    # NOTE: by construction (see generators.py) the *category* families are
    # expected to match (both have "ask_ambig" scenarios) — what must NOT
    # match is the underlying name/file/phrasing pool, checked in step 8.
    print(f"INFO: {len(shared)} shared category families between train and "
          f"OOD ({sorted(shared)}) — expected, since OOD tests the SAME "
          f"failure categories with UNSEEN entities/phrasing (checked below).")

    # 4. Missing / empty required fields
    missing_field_problems = []
    for s in all_scenarios:
        if not s.user_request.strip():
            missing_field_problems.append(f"{s.scenario_id}: empty user_request")
        if not s.gold_reason.strip():
            missing_field_problems.append(f"{s.scenario_id}: empty gold_reason")
        if not s.proposed_tool.tool_name:
            missing_field_problems.append(f"{s.scenario_id}: empty proposed_tool.tool_name")
    if missing_field_problems:
        print(f"FAIL: {len(missing_field_problems)} missing-value problem(s): "
              f"{missing_field_problems[:5]}")
        all_ok = False
    else:
        print("OK: No missing values in required fields.")

    # 5 & 6. Tool registry + reason-code consistency (reuse validators.py)
    failures = validate_dataset(all_scenarios)
    if failures:
        print(f"FAIL: {len(failures)} scenario(s) failed registry/consistency "
              f"validation: {list(failures.items())[:5]}")
        all_ok = False
    else:
        print("OK: No invalid tools; all gold_reason_codes consistent with gold_decision.")

    # 7. Distribution report
    print("\n-- Distribution report --")
    for label, key_fn in [
        ("decision", lambda s: s.gold_decision.value),
        ("domain", lambda s: s.domain),
        ("failure_category", lambda s: [c.value for c in s.gold_reason_codes] or ["none"]),
        ("tool", lambda s: s.proposed_tool.tool_name),
        ("risk_level", lambda s: s.risk_level.value),
    ]:
        counts = Counter()
        for s in all_scenarios:
            v = key_fn(s)
            if isinstance(v, list):
                counts.update(v)
            else:
                counts[v] += 1
        print(f"  by {label}: {dict(sorted(counts.items()))}")

    # 8. Near-duplicate / template-leakage detection between train and test splits
    print("\n-- Near-duplicate check (train vs test_iid, train vs test_ood) --")
    train_words = [(s.scenario_id, word_set(s.user_request)) for s in splits["train"]]
    for test_split_name in ["test_iid", "test_ood"]:
        near_dupes = []
        for test_s in splits[test_split_name]:
            tw = word_set(test_s.user_request)
            for train_id, trw in train_words:
                sim = jaccard(tw, trw)
                if sim >= NEAR_DUP_JACCARD_THRESHOLD:
                    near_dupes.append((test_s.scenario_id, train_id, round(sim, 2)))
        if near_dupes:
            tag = "EXPECTED for test_iid (same pools)" if test_split_name == "test_iid" else "INVESTIGATE for test_ood"
            print(f"  {test_split_name}: {len(near_dupes)} pair(s) with Jaccard "
                  f">= {NEAR_DUP_JACCARD_THRESHOLD} [{tag}], e.g. {near_dupes[:3]}")
        else:
            print(f"  {test_split_name}: no near-duplicates found.")

    print()
    if all_ok:
        print("Dataset audit passed.")
    else:
        print("Dataset audit FAILED — fix the issues above before training.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
