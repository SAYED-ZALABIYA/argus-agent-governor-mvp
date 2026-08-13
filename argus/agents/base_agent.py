"""
Common interface for anything that proposes a tool call: given a
Scenario (user request + environment state), an Agent proposes what to
do. ARGUS (a Governor) then reviews that proposal — it never sees the
scenario's own gold-label proposed_tool at decision time in the
end-to-end pipeline; it only sees what the Agent actually proposed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from argus.scenarios.schema import Scenario


@dataclass
class AgentProposal:
    tool_name: str
    arguments: dict[str, Any]
    plan: list[str] = field(default_factory=list)
    confidence: float | None = None


class Agent(ABC):
    name: str = "base"

    @abstractmethod
    def propose(self, scenario: Scenario) -> AgentProposal:
        """Given a scenario (user request + environment state), propose
        a tool call. Must NOT read scenario.gold_decision, gold_reason,
        gold_reason_codes, or expected_safe_action — those are the
        answer key, not something a real agent would have access to."""
        raise NotImplementedError
