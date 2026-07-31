"""
Structured schemas for ARGUS tools and scenarios.

Using pydantic (not plain dataclasses) on purpose: scenario generation is
programmatic and error-prone, and a bad gold label silently corrupts every
downstream number in the paper. Validators here catch internal
inconsistencies (e.g. "gold_decision=EXECUTE but reason_codes is non-empty
and includes IRREVERSIBLE_HIGH_RISK_ACTION") at generation time, not after
a model has already trained on them.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from argus.scenarios.taxonomy import (
    CANONICAL_ASK_CATEGORIES,
    CANONICAL_BLOCK_CATEGORIES,
    FailureCategory,
    GovernorDecision,
    MVP_DECISIONS,
    Reversibility,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
    """Static metadata describing one callable tool. Every tool in the
    sandbox must be registered as a ToolSpec — this is what the governor's
    "tool-specific features" (Stage 5) read from."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(
        description="JSON-schema-like dict: {field_name: {'type': ..., 'required': bool}}"
    )
    preconditions: list[str] = Field(default_factory=list)
    risk_level: RiskLevel
    reversible: Reversibility
    required_permissions: list[str] = Field(default_factory=list)

    def required_fields(self) -> list[str]:
        return [
            f for f, spec in self.input_schema.items()
            if spec.get("required", False)
        ]


# ---------------------------------------------------------------------------
# Proposed tool call (what the agent wants to do)
# ---------------------------------------------------------------------------

class ProposedToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scenario record
# ---------------------------------------------------------------------------

class Scenario(BaseModel):
    scenario_id: str
    domain: str  # "email" | "files" | "calendar" | "database"

    user_request: str
    environment_state: dict[str, Any] = Field(default_factory=dict)

    # What a (possibly flawed) small agent proposed to do.
    agent_plan: list[str] = Field(default_factory=list)
    proposed_tool: ProposedToolCall

    # Tool-level risk metadata (denormalized copy of the ToolSpec fields
    # relevant to this specific call, so the record is self-contained).
    risk_level: RiskLevel
    reversible: Reversibility
    required_permissions: list[str] = Field(default_factory=list)

    # Which preconditions are satisfied vs missing in this scenario.
    satisfied_preconditions: list[str] = Field(default_factory=list)
    missing_preconditions: list[str] = Field(default_factory=list)

    # Ground truth
    gold_decision: GovernorDecision
    gold_reason_codes: list[FailureCategory] = Field(default_factory=list)
    gold_reason: str
    expected_safe_action: Optional[str] = Field(
        default=None,
        description="Free-text description of what SHOULD happen "
                     "(e.g. the clarifying question to ask, or 'no-op').",
    )

    # OOD split marker, filled in at dataset-build time.
    split: Optional[str] = None  # "train" | "validation" | "test_iid" | "test_ood"

    # ---- Internal consistency validators -----------------------------

    @field_validator("gold_reason_codes")
    @classmethod
    def _dedupe_reason_codes(cls, v: list[FailureCategory]) -> list[FailureCategory]:
        seen = []
        for code in v:
            if code not in seen:
                seen.append(code)
        return seen

    @model_validator(mode="after")
    def _check_decision_reason_consistency(self) -> "Scenario":
        decision = self.gold_decision
        codes = set(self.gold_reason_codes)

        if decision == GovernorDecision.EXECUTE:
            if codes and codes != {FailureCategory.NONE}:
                raise ValueError(
                    f"{self.scenario_id}: gold_decision=EXECUTE but "
                    f"gold_reason_codes is non-trivial ({codes}). "
                    "A safe EXECUTE must carry no failure category."
                )
            if self.missing_preconditions:
                raise ValueError(
                    f"{self.scenario_id}: gold_decision=EXECUTE but "
                    f"missing_preconditions={self.missing_preconditions}. "
                    "EXECUTE requires all preconditions satisfied."
                )
        else:
            if not codes or codes == {FailureCategory.NONE}:
                raise ValueError(
                    f"{self.scenario_id}: gold_decision={decision} requires "
                    "at least one non-NONE failure category in gold_reason_codes."
                )

        if decision == GovernorDecision.ASK:
            if not codes.issubset(CANONICAL_ASK_CATEGORIES | {
                FailureCategory.STALE_OR_IRRELEVANT_MEMORY,
                FailureCategory.INCORRECT_TOOL_SELECTION,
            }):
                unexpected = codes - CANONICAL_ASK_CATEGORIES
                raise ValueError(
                    f"{self.scenario_id}: gold_decision=ASK with reason codes "
                    f"{unexpected} that are not canonically ASK-resolvable "
                    "under the MVP policy. Use BLOCK/VERIFY/ESCALATE instead, "
                    "or update CANONICAL_ASK_CATEGORIES deliberately."
                )

        if decision == GovernorDecision.BLOCK:
            if not (codes & CANONICAL_BLOCK_CATEGORIES) and not self.required_permissions_missing():
                raise ValueError(
                    f"{self.scenario_id}: gold_decision=BLOCK but none of "
                    f"{codes} are canonical block categories. If this is "
                    "intentional, add the category to CANONICAL_BLOCK_CATEGORIES."
                )

        return self

    @model_validator(mode="after")
    def _check_mvp_scope(self) -> "Scenario":
        if self.domain not in {"email", "files"}:
            # Non-MVP domains are allowed to exist in the schema (calendar,
            # database) but generators should not emit them until Stage 4+.
            pass
        return self

    def required_permissions_missing(self) -> bool:
        return bool(self.missing_preconditions) and any(
            "permission" in p.lower() for p in self.missing_preconditions
        )

    def is_mvp_compatible(self) -> bool:
        """True if this scenario only uses MVP-scope decisions/domains."""
        return (
            self.gold_decision in MVP_DECISIONS
            and self.domain in {"email", "files"}
        )
