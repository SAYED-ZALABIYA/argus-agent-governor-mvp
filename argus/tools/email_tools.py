"""
Email tool domain: search_emails, read_email, create_draft, send_email,
delete_email.

Each tool has two parts:
  1. A ToolSpec (static risk/schema metadata the Governor reads).
  2. A callable that executes against a Sandbox instance.
"""

from __future__ import annotations

from typing import Any

from argus.environment.sandbox import Sandbox
from argus.scenarios.taxonomy import Reversibility, RiskLevel
from argus.scenarios.schema import ToolSpec

EMAIL_TOOL_SPECS: dict[str, ToolSpec] = {
    "search_emails": ToolSpec(
        name="search_emails",
        description="Search emails by keyword in subject or body.",
        input_schema={"query": {"type": "string", "required": True}},
        preconditions=[],
        risk_level=RiskLevel.LOW,
        reversible=Reversibility.REVERSIBLE,
        required_permissions=["email.read"],
    ),
    "read_email": ToolSpec(
        name="read_email",
        description="Read a single email by id.",
        input_schema={"email_id": {"type": "string", "required": True}},
        preconditions=["email_exists"],
        risk_level=RiskLevel.LOW,
        reversible=Reversibility.REVERSIBLE,
        required_permissions=["email.read"],
    ),
    "create_draft": ToolSpec(
        name="create_draft",
        description="Create a draft email without sending it.",
        input_schema={
            "recipient": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
        },
        preconditions=["recipient_is_unambiguous"],
        risk_level=RiskLevel.LOW,
        reversible=Reversibility.REVERSIBLE,
        required_permissions=["email.write"],
    ),
    "send_email": ToolSpec(
        name="send_email",
        description="Send an email immediately.",
        input_schema={
            "recipient": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
            "attachment": {"type": "string", "required": False},
        },
        preconditions=["recipient_is_unambiguous", "message_content_exists"],
        risk_level=RiskLevel.MEDIUM,
        reversible=Reversibility.IRREVERSIBLE,
        required_permissions=["email.send"],
    ),
    "delete_email": ToolSpec(
        name="delete_email",
        description="Permanently delete an email.",
        input_schema={"email_id": {"type": "string", "required": True}},
        preconditions=["email_exists"],
        risk_level=RiskLevel.MEDIUM,
        reversible=Reversibility.IRREVERSIBLE,
        required_permissions=["email.delete"],
    ),
}


def execute_email_tool(sandbox: Sandbox, tool_name: str, arguments: dict[str, Any]):
    """Dispatch a validated tool call to the sandbox. The Governor is
    expected to have already approved this call (decision == EXECUTE)
    before this function is ever invoked."""
    if tool_name == "search_emails":
        return sandbox.search_emails(**arguments)
    if tool_name == "read_email":
        return sandbox.read_email(**arguments)
    if tool_name == "create_draft":
        return sandbox.create_draft(**arguments)
    if tool_name == "send_email":
        return sandbox.send_email(**arguments)
    if tool_name == "delete_email":
        return sandbox.delete_email(**arguments)
    raise ValueError(f"unknown email tool: {tool_name}")
