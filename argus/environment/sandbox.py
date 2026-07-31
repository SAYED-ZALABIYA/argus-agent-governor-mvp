"""
Deterministic local sandbox for ARGUS.

This is an in-memory simulation of an email client and a file system.
Nothing here ever touches a real network, real filesystem, or real
database — that is a hard project constraint (see Development
constraints: "never send real emails, delete real files, or modify real
databases").

The sandbox also records an execution history, which downstream feature
extraction (Stage 5, "tool previously failed?") reads from.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional


class SandboxError(Exception):
    """Raised when a tool call violates a precondition inside the sandbox."""


@dataclass
class Contact:
    name: str
    email: str


@dataclass
class EmailRecord:
    id: str
    sender: str
    recipient: str
    subject: str
    body: str
    is_draft: bool = False
    deleted: bool = False


@dataclass
class FileRecord:
    path: str
    content: str
    is_temporary: bool = False
    important: bool = False
    deleted: bool = False


@dataclass
class ExecutionLogEntry:
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    error: Optional[str] = None


@dataclass
class SandboxState:
    contacts: list[Contact] = field(default_factory=list)
    emails: dict[str, EmailRecord] = field(default_factory=dict)
    files: dict[str, FileRecord] = field(default_factory=dict)
    granted_permissions: set[str] = field(default_factory=set)
    execution_log: list[ExecutionLogEntry] = field(default_factory=list)
    _next_email_id: int = 1

    def snapshot(self) -> "SandboxState":
        """Deep copy for reproducible scenario construction / repeated runs."""
        return copy.deepcopy(self)


class Sandbox:
    """Mutable environment the tools operate on. One Sandbox instance =
    one scenario's environment_state, hydrated at scenario load time."""

    def __init__(self, state: Optional[SandboxState] = None):
        self.state = state or SandboxState()

    # -- helpers ----------------------------------------------------

    def has_permission(self, permission: str) -> bool:
        return permission in self.state.granted_permissions

    def resolve_contacts(self, name_fragment: str) -> list[Contact]:
        frag = name_fragment.lower().strip()
        return [c for c in self.state.contacts if frag in c.name.lower()]

    def _log(self, tool_name: str, arguments: dict[str, Any],
              success: bool, error: Optional[str] = None) -> None:
        self.state.execution_log.append(
            ExecutionLogEntry(tool_name, arguments, success, error)
        )

    def previous_failures(self, tool_name: str) -> int:
        return sum(
            1 for e in self.state.execution_log
            if e.tool_name == tool_name and not e.success
        )

    # -- email primitives --------------------------------------------

    def search_emails(self, query: str) -> list[EmailRecord]:
        q = query.lower()
        results = [
            e for e in self.state.emails.values()
            if not e.deleted and (q in e.subject.lower() or q in e.body.lower())
        ]
        self._log("search_emails", {"query": query}, True)
        return results

    def read_email(self, email_id: str) -> EmailRecord:
        email = self.state.emails.get(email_id)
        if email is None or email.deleted:
            self._log("read_email", {"email_id": email_id}, False, "not_found")
            raise SandboxError(f"email {email_id} not found")
        self._log("read_email", {"email_id": email_id}, True)
        return email

    def create_draft(self, recipient: str, subject: str, body: str) -> EmailRecord:
        eid = f"draft_{self.state._next_email_id}"
        self.state._next_email_id += 1
        record = EmailRecord(
            id=eid, sender="me", recipient=recipient,
            subject=subject, body=body, is_draft=True,
        )
        self.state.emails[eid] = record
        self._log("create_draft", {"recipient": recipient, "subject": subject}, True)
        return record

    def send_email(self, recipient: str, subject: str, body: str,
                    attachment: Optional[str] = None) -> EmailRecord:
        if not recipient or "@" not in recipient:
            self._log("send_email", {"recipient": recipient}, False, "invalid_recipient")
            raise SandboxError("invalid or missing recipient")
        if attachment and attachment not in self.state.files:
            self._log("send_email", {"attachment": attachment}, False, "attachment_not_found")
            raise SandboxError(f"attachment {attachment} not found")
        eid = f"sent_{self.state._next_email_id}"
        self.state._next_email_id += 1
        record = EmailRecord(id=eid, sender="me", recipient=recipient,
                              subject=subject, body=body, is_draft=False)
        self.state.emails[eid] = record
        self._log("send_email", {"recipient": recipient, "subject": subject}, True)
        return record

    def delete_email(self, email_id: str) -> None:
        email = self.state.emails.get(email_id)
        if email is None or email.deleted:
            self._log("delete_email", {"email_id": email_id}, False, "not_found")
            raise SandboxError(f"email {email_id} not found")
        email.deleted = True
        self._log("delete_email", {"email_id": email_id}, True)

    # -- file primitives ----------------------------------------------

    def list_files(self, path_prefix: str = "") -> list[FileRecord]:
        results = [
            f for f in self.state.files.values()
            if not f.deleted and f.path.startswith(path_prefix)
        ]
        self._log("list_files", {"path_prefix": path_prefix}, True)
        return results

    def read_file(self, path: str) -> FileRecord:
        f = self.state.files.get(path)
        if f is None or f.deleted:
            self._log("read_file", {"path": path}, False, "not_found")
            raise SandboxError(f"file {path} not found")
        self._log("read_file", {"path": path}, True)
        return f

    def edit_file(self, path: str, new_content: str) -> FileRecord:
        f = self.state.files.get(path)
        if f is None or f.deleted:
            self._log("edit_file", {"path": path}, False, "not_found")
            raise SandboxError(f"file {path} not found")
        f.content = new_content
        self._log("edit_file", {"path": path}, True)
        return f

    def move_file(self, src: str, dst: str) -> FileRecord:
        f = self.state.files.get(src)
        if f is None or f.deleted:
            self._log("move_file", {"src": src, "dst": dst}, False, "not_found")
            raise SandboxError(f"file {src} not found")
        if dst in self.state.files:
            self._log("move_file", {"src": src, "dst": dst}, False, "dst_exists")
            raise SandboxError(f"destination {dst} already exists")
        f.path = dst
        del self.state.files[src]
        self.state.files[dst] = f
        self._log("move_file", {"src": src, "dst": dst}, True)
        return f

    def delete_file(self, path: str) -> None:
        f = self.state.files.get(path)
        if f is None or f.deleted:
            self._log("delete_file", {"path": path}, False, "not_found")
            raise SandboxError(f"file {path} not found")
        f.deleted = True
        self._log("delete_file", {"path": path}, True)
