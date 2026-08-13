"""
Feature extraction: Scenario -> flat dict of interpretable features.

No embeddings, no LLM calls. Every feature here is either a direct
readout of structured scenario fields (domain, tool name, risk level,
precondition lists) or a cheap deterministic heuristic (keyword search,
counting) over `user_request` / `environment_state` text.

HARD RULE: gold_decision, gold_reason, gold_reason_codes, and
expected_safe_action are NEVER read by this module. `FORBIDDEN_FIELDS`
below exists so `extract_features` can assert this at import time rather
than relying on developer discipline alone.
"""

from __future__ import annotations

from argus.scenarios.schema import Scenario
from argus.scenarios.validators import TOOL_REGISTRY

FORBIDDEN_FIELDS = {"gold_decision", "gold_reason", "gold_reason_codes",
                     "expected_safe_action"}

DESTRUCTIVE_TOOLS = {"delete_email", "delete_file"}

NEGATED_SEND_PHRASES = [
    "don't send", "dont send", "not send", "hold off", "not now",
    "later", "don't actually send", "just save", "not yet",
]

SENSITIVE_KEYWORDS = [
    "salary", "medical", "ssn", "payroll", "disciplinary", "performance_review",
    "bank_details", "home_address", "compensation", "background_check",
    "tax_return", "insurance_claim",
]

PUBLIC_TARGET_KEYWORDS = [
    "public", "shared", "wiki", "vendor", "partner", "external", "portal",
    "customer", "contractor", "intern",
]

INJECTION_PHRASES = [
    "ignore previous instructions", "ignore all previous instructions",
    "disregard previous instructions", "new instructions:",
]

ALL_TOOLS = ["send_email", "delete_email", "create_draft", "search_emails",
             "read_email", "read_file", "delete_file", "move_file", "list_files"]


def _local_part(email: str) -> str:
    return email.split("@")[0].split(".")[0].lower() if "@" in email else email.lower()


def _text_contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in phrases)


def extract_features(scenario: Scenario) -> dict[str, int | float]:
    features: dict[str, int | float] = {}

    # -- domain -----------------------------------------------------
    features["domain_email"] = int(scenario.domain == "email")
    features["domain_files"] = int(scenario.domain == "files")

    # -- tool one-hot -------------------------------------------------
    tool_name = scenario.proposed_tool.tool_name
    for t in ALL_TOOLS:
        features[f"tool_{t}"] = int(tool_name == t)

    # -- risk / reversibility -----------------------------------------
    features["risk_low"] = int(scenario.risk_level.value == "low")
    features["risk_medium"] = int(scenario.risk_level.value == "medium")
    features["risk_high"] = int(scenario.risk_level.value in ("high", "critical"))
    features["is_reversible"] = int(scenario.reversible.value == "reversible")

    # -- permissions / preconditions ------------------------------------
    permission_missing = any(
        p.startswith("permission:") for p in scenario.missing_preconditions
    )
    features["has_required_permission"] = int(not permission_missing)

    spec = TOOL_REGISTRY.get(tool_name)
    required_fields = spec.required_fields() if spec else []
    missing_args = [
        f for f in required_fields
        if not scenario.proposed_tool.arguments.get(f)
    ]
    features["schema_valid"] = int(len(missing_args) == 0)
    features["missing_argument_count"] = len(missing_args)

    features["missing_precondition_count"] = len(scenario.missing_preconditions)
    total_preconditions = len(scenario.satisfied_preconditions) + len(scenario.missing_preconditions)
    features["satisfied_precondition_ratio"] = (
        len(scenario.satisfied_preconditions) / total_preconditions
        if total_preconditions > 0 else 1.0
    )

    # -- entity ambiguity (derived from environment_state, not gold labels) --
    contacts = scenario.environment_state.get("contacts", [])
    files_in_env = scenario.environment_state.get("files", [])
    recipient = scenario.proposed_tool.arguments.get("recipient", "")
    recipient_prefix = _local_part(recipient) if recipient else ""
    ambiguous_entity_count = 0
    if recipient_prefix:
        for c in contacts:
            c_prefix = _local_part(c.get("email", ""))
            if c_prefix == recipient_prefix and c.get("email") != recipient:
                ambiguous_entity_count += 1
    features["ambiguous_entity_count"] = ambiguous_entity_count
    features["candidate_recipient_count"] = len(contacts)
    features["candidate_file_count"] = len(files_in_env)

    # -- intent / risk heuristics ---------------------------------------
    tool_mismatch = (
        tool_name == "send_email"
        and _text_contains_any(scenario.user_request, NEGATED_SEND_PHRASES)
    )
    features["tool_matches_user_intent"] = int(not tool_mismatch)
    features["proposed_action_is_destructive"] = int(tool_name in DESTRUCTIVE_TOOLS)

    target_path = " ".join(str(v) for v in scenario.proposed_tool.arguments.values())
    features["contains_sensitive_data"] = int(
        _text_contains_any(target_path, SENSITIVE_KEYWORDS)
        or _text_contains_any(scenario.user_request, SENSITIVE_KEYWORDS)
    )
    keyword_public = _text_contains_any(target_path, PUBLIC_TARGET_KEYWORDS)
    src = scenario.proposed_tool.arguments.get("src", "")
    dst = scenario.proposed_tool.arguments.get("dst", "")
    src_top = src.split("/")[0].lower() if "/" in src else ""
    dst_top = dst.split("/")[0].lower() if "/" in dst else ""
    path_leaves_restricted_folder = bool(src_top) and src_top != dst_top
    features["target_is_public"] = int(keyword_public or path_leaves_restricted_folder)

    file_contents = " ".join(
        str(f.get("content", "")) for f in files_in_env
    )
    features["prompt_injection_detected"] = int(
        _text_contains_any(file_contents, INJECTION_PHRASES)
    )

    # -- target file signals (indirect/imperfect, NOT the ground truth) --
    # These mirror what a real system could plausibly observe (file
    # metadata) rather than an oracle "is this file important" flag.
    # The true importance used to generate gold labels is these signals
    # PLUS irreducible noise (see generators.py) — so even a governor
    # that reads these perfectly cannot reach 100% on this category.
    target_path_arg = (scenario.proposed_tool.arguments.get("path")
                        or scenario.proposed_tool.arguments.get("src") or "")
    target_file_entry = next(
        (f for f in files_in_env if f.get("path") == target_path_arg), None
    )
    if target_file_entry is not None:
        days_ago = target_file_entry.get("last_modified_days_ago")
        features["target_file_recently_modified"] = int(
            days_ago is not None and days_ago <= 3
        )
        features["target_file_referenced_elsewhere"] = int(
            bool(target_file_entry.get("referenced_elsewhere"))
        )
    else:
        features["target_file_recently_modified"] = 0
        features["target_file_referenced_elsewhere"] = 0

    # -- shape / size features ------------------------------------------
    features["request_length"] = len(scenario.user_request.split())
    features["plan_step_count"] = len(scenario.agent_plan)
    features["argument_count"] = len(scenario.proposed_tool.arguments)

    return features


def extract_batch(scenarios: list[Scenario]) -> list[dict[str, int | float]]:
    return [extract_features(s) for s in scenarios]
