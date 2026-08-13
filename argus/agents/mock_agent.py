"""
MockAgent: a trivial agent used to prove the end-to-end pipeline
(Agent -> ARGUS -> tool execution) works before any real model is
involved.

Phase 2 scope (per project plan): starts by echoing the scenario's own
proposed_tool — which was already hand/programmatically designed to be
sometimes correct and sometimes deliberately flawed (wrong recipient,
wrong file, wrong tool choice, unsafe deletion). This is enough to prove
the pipeline reacts correctly to both good and bad proposals without
yet needing an agent that reasons independently.

A later revision can make MockAgent generate proposals independently of
scenario.proposed_tool (e.g. from environment_state alone), once the
pipeline itself is confirmed working end to end.
"""

from __future__ import annotations

from argus.agents.base_agent import Agent, AgentProposal
from argus.scenarios.schema import Scenario


class MockAgent(Agent):
    name = "mock_agent"

    def propose(self, scenario: Scenario) -> AgentProposal:
        return AgentProposal(
            tool_name=scenario.proposed_tool.tool_name,
            arguments=dict(scenario.proposed_tool.arguments),
            plan=list(scenario.agent_plan) or ["Resolve entities", "Call selected tool"],
            confidence=0.75,
        )
