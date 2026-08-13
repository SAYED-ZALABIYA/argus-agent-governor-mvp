"""
Phase 2 acceptance test: Agent -> ARGUS -> (maybe) execute.

Uses the exact example from the project plan: "Send the report to Ali"
— an intentionally ambiguous scenario (two contacts named Ali) — plus a
second, unambiguous scenario, to show both the ASK and EXECUTE paths
actually running end to end, including real sandbox execution when
ARGUS approves.

Run: PYTHONPATH=. python3 scripts/run_agent_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_scenarios import build_scenarios

from argus.agents.mock_agent import MockAgent
from argus.environment.sandbox import hydrate_sandbox
from argus.features.extractor import extract_features
from argus.governors.rule_based import RuleBasedGovernor
from argus.scenarios.schema import ProposedToolCall
from argus.scenarios.taxonomy import GovernorDecision
from argus.tools.email_tools import execute_email_tool
from argus.tools.file_tools import execute_file_tool


def run_one(scenario, agent: MockAgent, governor: RuleBasedGovernor) -> None:
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario.scenario_id}")
    print(f"User request: {scenario.user_request}")

    proposal = agent.propose(scenario)
    print(f"Agent proposal: {proposal.tool_name}({proposal.arguments})")

    # ARGUS reviews the AGENT's proposal, not the scenario's baked-in
    # gold proposed_tool — this is the whole point of the end-to-end
    # test. We build a shadow copy of the scenario with proposed_tool
    # swapped for what the agent actually said, so extract_features()
    # (which reads scenario.proposed_tool) sees the agent's decision.
    scenario_for_governor = scenario.model_copy(update={
        "proposed_tool": ProposedToolCall(tool_name=proposal.tool_name,
                                           arguments=proposal.arguments),
        "agent_plan": proposal.plan,
    })
    features = extract_features(scenario_for_governor)
    decision = governor.decide(features)
    print(f"ARGUS decision: {decision.value.upper()}")

    if decision != GovernorDecision.EXECUTE:
        print("Tool executed: No")
        if decision == GovernorDecision.ASK:
            print(f"  (would ask: \"{scenario.expected_safe_action}\")")
        return

    sandbox = hydrate_sandbox(scenario.environment_state)
    try:
        if scenario.domain == "email":
            execute_email_tool(sandbox, proposal.tool_name, proposal.arguments)
        else:
            execute_file_tool(sandbox, proposal.tool_name, proposal.arguments)
        print("Tool executed: Yes")
    except Exception as e:
        print(f"Tool executed: No (sandbox raised {type(e).__name__}: {e})")


def main() -> None:
    scenarios = {s.scenario_id: s for s in build_scenarios()}
    agent = MockAgent()
    governor = RuleBasedGovernor()

    # The project plan's exact example: ambiguous "Send the report to Ali"
    run_one(scenarios["email_002"], agent, governor)
    # Contrast: unambiguous, safe send -> should EXECUTE for real
    run_one(scenarios["email_001"], agent, governor)
    # Contrast: unsafe delete -> should BLOCK
    run_one(scenarios["file_005"], agent, governor)


if __name__ == "__main__":
    main()
