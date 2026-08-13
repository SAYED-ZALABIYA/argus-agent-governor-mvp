"""
Files tool domain: list_files, read_file, edit_file, move_file, delete_file.
"""

from __future__ import annotations

from typing import Any

from argus.environment.sandbox import Sandbox
from argus.scenarios.taxonomy import Reversibility, RiskLevel
from argus.scenarios.schema import ToolSpec

FILE_TOOL_SPECS: dict[str, ToolSpec] = {
    "list_files": ToolSpec(
        name="list_files",
        description="List files matching a path prefix.",
        input_schema={"path_prefix": {"type": "string", "required": False}},
        preconditions=[],
        risk_level=RiskLevel.LOW,
        reversible=Reversibility.REVERSIBLE,
        required_permissions=["files.read"],
    ),
    "read_file": ToolSpec(
        name="read_file",
        description="Read the contents of a file.",
        input_schema={"path": {"type": "string", "required": True}},
        preconditions=["file_exists"],
        risk_level=RiskLevel.LOW,
        reversible=Reversibility.REVERSIBLE,
        required_permissions=["files.read"],
    ),
    "edit_file": ToolSpec(
        name="edit_file",
        description="Overwrite the contents of an existing file.",
        input_schema={
            "path": {"type": "string", "required": True},
            "new_content": {"type": "string", "required": True},
        },
        preconditions=["file_exists"],
        risk_level=RiskLevel.MEDIUM,
        reversible=Reversibility.PARTIALLY_REVERSIBLE,
        required_permissions=["files.write"],
    ),
    "move_file": ToolSpec(
        name="move_file",
        description="Move/rename a file.",
        input_schema={
            "src": {"type": "string", "required": True},
            "dst": {"type": "string", "required": True},
        },
        preconditions=["file_exists", "destination_available"],
        risk_level=RiskLevel.LOW,
        reversible=Reversibility.REVERSIBLE,
        required_permissions=["files.write"],
    ),
    "delete_file": ToolSpec(
        name="delete_file",
        description="Permanently delete a file.",
        input_schema={"path": {"type": "string", "required": True}},
        preconditions=["file_exists"],
        risk_level=RiskLevel.HIGH,
        reversible=Reversibility.IRREVERSIBLE,
        required_permissions=["files.delete"],
    ),
}


def execute_file_tool(sandbox: Sandbox, tool_name: str, arguments: dict[str, Any]):
    """Dispatch a validated tool call to the sandbox. The Governor is
    expected to have already approved this call before this runs."""
    if tool_name == "list_files":
        return sandbox.list_files(**arguments)
    if tool_name == "read_file":
        return sandbox.read_file(**arguments)
    if tool_name == "edit_file":
        return sandbox.edit_file(**arguments)
    if tool_name == "move_file":
        return sandbox.move_file(**arguments)
    if tool_name == "delete_file":
        return sandbox.delete_file(**arguments)
    raise ValueError(f"unknown file tool: {tool_name}")
