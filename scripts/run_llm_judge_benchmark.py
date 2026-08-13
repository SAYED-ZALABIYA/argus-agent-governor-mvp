"""
Phase 4 — LLM-as-a-Judge comparison (second main paper experiment).

Compares agent_rule_based, agent_xgboost, and agent_llm_judge on the
SAME scenarios, measuring both the safety metrics (as in Phase 3) and
the cost/latency numbers the LLM judge actually incurs.

COST WARNING: every scenario run through the LLM judge is a real,
billed API call. Defaults to a small sample (20) so you can sanity-check
before spending more. Increase --n once you've confirmed the prompt and
parsing work as expected.

Setup:
    pip install anthropic
    Set ANTHROPIC_API_KEY in your environment.
    Edit MODEL_NAME in argus/governors/llm_judge.py to a model you have
    access to.

Run:
    PYTHONPATH=. python3 scripts/run_llm_judge_benchmark.py --n 20
    PYTHONPATH=. python3 scripts/run_llm_judge_benchmark.py --n 135   # full held-out set
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argus.agents.mock_agent import MockAgent
from argus.evaluation.metrics import compute_all_metrics, print_report
from argus.features.extractor import extract_features
from argus.governors.llm_judge import LLMJudgeGovernor
from argus.governors.rule_based import RuleBasedGovernor
from argus.scenarios.schema import ProposedToolCall, Scenario
from argus.scenarios.taxonomy import GovernorDecision

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20,
                         help="Number of scenarios to run through the LLM judge (costs real money).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_scenarios = load_split("test_iid") + load_split("test_ood")
    rng = random.Random(args.seed)
    sample = rng.sample(all_scenarios, min(args.n, len(all_scenarios)))
    print(f"Running {len(sample)} scenarios through rule_based, xgboost, and llm_judge "
          f"(of {len(all_scenarios)} available held-out scenarios).")
    print(f"NOTE: {len(sample)} real API calls will be made now.")

    agent = MockAgent()
    rule_governor = RuleBasedGovernor()
    xgb_predict = load_xgboost_predictor()
    llm_judge = LLMJudgeGovernor()

    gold = [s.gold_decision for s in sample]
    decisions: dict[str, list[GovernorDecision]] = {
        "agent_rule_based": [], "agent_xgboost": [], "agent_llm_judge": [],
    }
    rows = []

    for i, s in enumerate(sample):
        proposal = agent.propose(s)
        scenario_for_governor = s.model_copy(update={
            "proposed_tool": ProposedToolCall(tool_name=proposal.tool_name,
                                               arguments=proposal.arguments),
            "agent_plan": proposal.plan,
        })
        features = extract_features(scenario_for_governor)

        rb_decision = rule_governor.decide(features)
        xgb_decision = xgb_predict(features)
        llm_result = llm_judge.review_scenario(scenario_for_governor)

        decisions["agent_rule_based"].append(rb_decision)
        decisions["agent_xgboost"].append(xgb_decision)
        decisions["agent_llm_judge"].append(llm_result.decision)

        rows.append({
            "scenario_id": s.scenario_id, "gold_decision": s.gold_decision.value,
            "rule_based": rb_decision.value, "xgboost": xgb_decision.value,
            "llm_judge": llm_result.decision.value, "llm_judge_reason": llm_result.reason,
            "llm_latency_s": round(llm_result.latency_seconds, 3),
            "llm_input_tokens": llm_result.input_tokens,
            "llm_output_tokens": llm_result.output_tokens,
        })
        print(f"  [{i + 1}/{len(sample)}] {s.scenario_id}: gold={s.gold_decision.value} "
              f"rb={rb_decision.value} xgb={xgb_decision.value} "
              f"llm={llm_result.decision.value} ({llm_result.latency_seconds:.2f}s)")

    print("\n" + "=" * 78)
    print("SAFETY METRICS")
    print("=" * 78)
    all_metrics = {}
    for sys_name in ["agent_rule_based", "agent_xgboost", "agent_llm_judge"]:
        m = compute_all_metrics(gold, decisions[sys_name])
        print_report(sys_name, m)
        all_metrics[sys_name] = m

    print("\n" + "=" * 78)
    print("COST / LATENCY (LLM judge only — rule_based and xgboost are ~free and instant)")
    print("=" * 78)
    usage = llm_judge.usage_summary()
    for k, v in usage.items():
        print(f"  {k}: {v}")
    print("\n  NOTE: convert avg_input_tokens/avg_output_tokens to a dollar cost "
          "yourself using current published pricing for the model you used — "
          "this script deliberately does not hardcode a $/token figure.")

    with open(RESULTS_DIR / "llm_judge_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": all_metrics, "llm_usage": usage}, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / "llm_judge_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
