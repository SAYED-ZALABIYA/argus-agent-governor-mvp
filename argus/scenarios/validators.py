"""
Deterministic, LLM-free scenario validators.

`schema.py`'s pydantic validators only check *internal* consistency
(does gold_decision match gold_reason_codes?). They know nothing about
the actual tools. This module cross-checks each Scenario against the
real ToolSpec registry (email_tools.py / file_tools.py), so a generator
bug like "labeled delete_file as LOW risk" gets caught immediately
instead of silently entering the training set.

Every check here is a plain boolean/comparison — no model calls, no
judgment calls. That's the point: ground truth must be verifiable
without an LLM judge (Stage 2 requirement).
"""

from __future__ import annotations

from argus.scenarios.schema import Scenario, ToolSpec
from argus.scenarios.taxonomy import GovernorDecision, MVP_DECISIONS
from argus.tools.email_tools import EMAIL_TOOL_SPECS
from argus.tools.file_tools import FILE_TOOL_SPECS

TOOL_REGISTRY: dict[str, ToolSpec] = {**EMAIL_TOOL_SPECS, **FILE_TOOL_SPECS}


class ValidationError(Exception):
    """Raised with a list of every problem found, not just the first one."""

    def __init__(self, scenario_id: str, problems: list[str]):
        self.scenario_id = scenario_id
        self.problems = problems
        super().__init__(f"{scenario_id}: {len(problems)} problem(s): {problems}")


def _check_tool_exists(scenario: Scenario, problems: list[str]) -> ToolSpec | None:
    spec = TOOL_REGISTRY.get(scenario.proposed_tool.tool_name)
    if spec is None:
        problems.append(
            f"proposed_tool '{scenario.proposed_tool.tool_name}' is not a "
            f"registered tool (known tools: {sorted(TOOL_REGISTRY)})"
        )
    return spec


def _check_risk_metadata_matches_registry(
    scenario: Scenario, spec: ToolSpec, problems: list[str]
) -> None:
    if scenario.risk_level != spec.risk_level:
        problems.append(
            f"scenario.risk_level={scenario.risk_level.value} but the "
            f"registered tool '{spec.name}' has risk_level={spec.risk_level.value}"
        )
    if scenario.reversible != spec.reversible:
        problems.append(
            f"scenario.reversible={scenario.reversible.value} but the "
            f"registered tool '{spec.name}' has reversible={spec.reversible.value}"
        )
    if set(scenario.required_permissions) != set(spec.required_permissions):
        problems.append(
            f"scenario.required_permissions={scenario.required_permissions} "
            f"but the registered tool '{spec.name}' requires "
            f"{spec.required_permissions}"
        )


def _check_required_arguments_present(
    scenario: Scenario, spec: ToolSpec, problems: list[str]
) -> None:
    """Only enforced when gold_decision == EXECUTE: a governor that
    approves execution must be approving a well-formed call."""
    if scenario.gold_decision != GovernorDecision.EXECUTE:
        return
    for field_name in spec.required_fields():
        value = scenario.proposed_tool.arguments.get(field_name)
        if value is None or value == "":
            problems.append(
                f"gold_decision=EXECUTE but required argument "
                f"'{field_name}' is missing/empty for tool '{spec.name}'"
            )


def _check_mvp_scope(scenario: Scenario, problems: list[str]) -> None:
    if scenario.domain not in {"email", "files"}:
        problems.append(
            f"domain='{scenario.domain}' is outside the MVP scope "
            f"(email, files only)"
        )
    if scenario.gold_decision not in MVP_DECISIONS:
        problems.append(
            f"gold_decision={scenario.gold_decision.value} is outside the "
            f"MVP decision set {sorted(d.value for d in MVP_DECISIONS)}"
        )


def _check_precondition_lists_dont_overlap(scenario: Scenario, problems: list[str]) -> None:
    overlap = set(scenario.satisfied_preconditions) & set(scenario.missing_preconditions)
    if overlap:
        problems.append(
            f"preconditions {overlap} appear in BOTH satisfied and missing lists"
        )


def validate_scenario(scenario: Scenario, raise_on_error: bool = True) -> list[str]:
    """Run every deterministic check. Returns the list of problems found
    (empty list == passes). If raise_on_error, raises ValidationError
    instead of returning a non-empty list."""
    problems: list[str] = []

    _check_mvp_scope(scenario, problems)
    _check_precondition_lists_dont_overlap(scenario, problems)

    spec = _check_tool_exists(scenario, problems)
    if spec is not None:
        _check_risk_metadata_matches_registry(scenario, spec, problems)
        _check_required_arguments_present(scenario, spec, problems)

    if problems and raise_on_error:
        raise ValidationError(scenario.scenario_id, problems)
    return problems


def validate_dataset(scenarios: list[Scenario]) -> dict[str, list[str]]:
    """Validate a whole dataset without stopping at the first failure.
    Returns {scenario_id: [problems]} for every scenario that failed —
    empty dict means the whole dataset is clean."""
    failures: dict[str, list[str]] = {}
    for s in scenarios:
        problems = validate_scenario(s, raise_on_error=False)
        if problems:
            failures[s.scenario_id] = problems
    return failures
