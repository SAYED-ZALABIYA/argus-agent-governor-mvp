"""
Baseline 1: Rule-Based Governor.

Fixed if/elif policy over the extracted feature dict — no learning, no
parameters. This is not meant to be optimal; it's the reference point
every learned governor (logistic, xgboost) must beat to justify its
added complexity.

Policy order matters: prompt injection and missing permission are
checked first because they are absolute — no combination of other
signals should override them.
"""

from __future__ import annotations

from argus.governors.base import Governor
from argus.scenarios.taxonomy import GovernorDecision


class RuleBasedGovernor(Governor):
    name = "rule_based"

    def decide(self, features: dict[str, int | float]) -> GovernorDecision:
        if features.get("prompt_injection_detected"):
            return GovernorDecision.BLOCK

        if not features.get("has_required_permission", 1):
            return GovernorDecision.BLOCK

        privacy_exposure = (
            features.get("contains_sensitive_data")
            and features.get("target_is_public")
        )
        unsafe_destructive_action = (
            features.get("proposed_action_is_destructive")
            and not features.get("is_reversible")
            and features.get("risk_high")
            and (features.get("target_file_recently_modified")
                 or features.get("target_file_referenced_elsewhere"))
        )
        if privacy_exposure or unsafe_destructive_action:
            return GovernorDecision.BLOCK

        missing_required_information = (
            not features.get("schema_valid", 1)
            or features.get("missing_argument_count", 0) > 0
        )
        if missing_required_information:
            return GovernorDecision.ASK

        if features.get("ambiguous_entity_count", 0) > 0:
            return GovernorDecision.ASK

        if not features.get("tool_matches_user_intent", 1):
            return GovernorDecision.ASK

        return GovernorDecision.EXECUTE
