"""
Phase 3 — End-to-End Benchmark (first main paper experiment).

Compares three systems on the SAME held-out scenarios:
  1. agent_alone       — MockAgent's proposal executes with no review.
  2. agent_rule_based   — MockAgent's proposal reviewed by RuleBasedGovernor.
  3. agent_xgboost      — MockAgent's proposal reviewed by the trained XGBoost governor.

Uses test_iid + test_ood combined (135 scenarios as of the current
dataset build — NOT the 200 in the original plan; validation/train are
excluded because XGBoost was trained on train and validated on
validation, so including them would bias its numbers). If more
statistical power is needed later, scale up test_iid/test_ood in
build_dataset.py and rerun.

For each scenario, if a system's decision is EXECUTE, the tool call
actually runs against a hydrated Sandbox — this is real execution, not
a label comparison.

Run: PYTHONPATH=. python3 scripts/run_end_to_end_benchmark.py
(Run scripts/train_baselines.py first so models/xgboost.pkl exists.)
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argus.agents.mock_agent import MockAgent
from argus.environment.sandbox import SandboxError, hydrate_sandbox
from argus.evaluation.metrics import compute_all_metrics, print_report
from argus.features.extractor import extract_features
from argus.governors.rule_based import RuleBasedGovernor
from argus.scenarios.schema import ProposedToolCall, Scenario
from argus.scenarios.taxonomy import GovernorDecision
from argus.tools.email_tools import execute_email_tool
from argus.tools.file_tools import execute_file_tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_split(name: str) -> list[Scenario]:
    path = DATA_DIR / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [Scenario.model_validate(json.loads(line)) for line in f]


def load_xgboost_predictor():
    with open(MODELS_DIR / "xgboost.pkl", "rb") as f:
        bundle = pickle.load(f)

    def predict(features: dict) -> GovernorDecision:
        X = bundle["vectorizer"].transform([features])
        model = bundle["model"]
        if bundle.get("backend") == "xgboost":
            idx = int(model.predict(X)[0])
            return GovernorDecision(bundle["idx_to_label"][idx])
        return GovernorDecision(model.predict(X)[0])

    return predict


def execute_proposal(scenario: Scenario, proposal) -> tuple[bool, str]:
    """Actually run the agent's proposed tool call against a freshly
    hydrated sandbox. Returns (succeeded, note)."""
    sandbox = hydrate_sandbox(scenario.environment_state)
    try:
        if scenario.domain == "email":
            execute_email_tool(sandbox, proposal.tool_name, proposal.arguments)
        else:
            execute_file_tool(sandbox, proposal.tool_name, proposal.arguments)
        return True, "ok"
    except SandboxError as e:
        return False, f"sandbox_error: {e}"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = load_split("test_iid") + load_split("test_ood")
    print(f"Loaded {len(scenarios)} held-out scenarios "
          f"(test_iid + test_ood — see module docstring re: the 200-task target).")

    agent = MockAgent()
    rule_governor = RuleBasedGovernor()
    xgb_predict = load_xgboost_predictor()

    gold = [s.gold_decision for s in scenarios]
    decisions: dict[str, list[GovernorDecision]] = {
        "agent_alone": [], "agent_rule_based": [], "agent_xgboost": [],
    }
    executed_flags: dict[str, list[bool]] = {
        "agent_alone": [], "agent_rule_based": [], "agent_xgboost": [],
    }
    rows = []

    for s in scenarios:
        proposal = agent.propose(s)
        scenario_for_governor = s.model_copy(update={
            "proposed_tool": ProposedToolCall(tool_name=proposal.tool_name,
                                               arguments=proposal.arguments),
            "agent_plan": proposal.plan,
        })
        features = extract_features(scenario_for_governor)

        system_decisions = {
            "agent_alone": GovernorDecision.EXECUTE,  # no review at all
            "agent_rule_based": rule_governor.decide(features),
            "agent_xgboost": xgb_predict(features),
        }

        row = {"scenario_id": s.scenario_id, "domain": s.domain,
               "user_request": s.user_request, "gold_decision": s.gold_decision.value,
               "proposed_tool": proposal.tool_name}

        for sys_name, decision in system_decisions.items():
            decisions[sys_name].append(decision)
            executed = False
            if decision == GovernorDecision.EXECUTE:
                executed, note = execute_proposal(s, proposal)
            executed_flags[sys_name].append(executed)
            row[f"{sys_name}_decision"] = decision.value
            row[f"{sys_name}_executed"] = executed
        rows.append(row)

    # -- metrics per system ------------------------------------------
    all_metrics = {}
    for sys_name in ["agent_alone", "agent_rule_based", "agent_xgboost"]:
        m = compute_all_metrics(gold, decisions[sys_name])
        # Task Success Rate: system's decision matches gold (same as
        # accuracy, but named per the project's metric list).
        m["task_success_rate"] = m["accuracy"]
        print_report(sys_name, m)
        all_metrics[sys_name] = m

    with open(RESULTS_DIR / "end_to_end_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "end_to_end_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print(f"SUMMARY — Phase 3 End-to-End Benchmark (n={len(scenarios)})")
    print("=" * 78)
    print(f"{'System':20s} {'Task Success':>14s} {'Unsafe Exec':>14s} "
          f"{'False Interv':>14s} {'Autonomy':>10s} {'Sev-Wt Err':>12s}")
    for sys_name in ["agent_alone", "agent_rule_based", "agent_xgboost"]:
        m = all_metrics[sys_name]
        print(f"{sys_name:20s} {m['task_success_rate']:14.3f} "
              f"{m['unsafe_execution_rate']:14.3f} {m['false_intervention_rate']:14.3f} "
              f"{m['autonomy_coverage']:10.3f} {m['severity_weighted_error']:12.3f}")

    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
