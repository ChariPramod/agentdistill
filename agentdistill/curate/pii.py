"""PII redaction hook.

The rule from the plan: run the redaction hook, and drop traces where redaction changed a *tool argument*. The
model must not learn placeholder tokens as valid arguments -- a student that emits `<EMAIL>` into a real API call
is worse than no student.

Redaction inside assistant prose and tool results is applied in place and kept; only argument changes are fatal.
The default patterns are deliberately conservative. Projects with real PII should pass their own redactor:

    from agentdistill.curate import pii
    pii.set_redactor(my_callable)   # str -> str
"""

from __future__ import annotations

import re
from collections.abc import Callable

Redactor = Callable[[str], str]

DEFAULT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("<EMAIL>", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("<CREDIT_CARD>", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("<SSN>", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("<PHONE>", re.compile(r"\b(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("<IP>", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

_redactor: Redactor | None = None


def default_redact(text: str) -> str:
    for token, pattern in DEFAULT_PATTERNS:
        text = pattern.sub(token, text)
    return text


def set_redactor(fn: Redactor | None) -> None:
    global _redactor
    _redactor = fn


def redact(text: str) -> str:
    return (_redactor or default_redact)(text)


def redact_trace(trace: dict) -> tuple[dict, bool, list[str]]:
    """Return (redacted_trace, argument_changed, notes).

    `argument_changed` True means the caller must drop this trace.
    """
    notes: list[str] = []
    argument_changed = False
    messages = []
    for i, m in enumerate(trace["messages"]):
        msg = dict(m)
        if msg.get("content"):
            new = redact(msg["content"])
            if new != msg["content"]:
                notes.append(f"turn {i}: redacted {msg['role']} content")
                msg["content"] = new
        if msg.get("tool_calls"):
            calls = []
            for c in msg["tool_calls"]:
                args = c["function"]["arguments"]
                new_args = redact(args)
                if new_args != args:
                    argument_changed = True
                    notes.append(f"turn {i}: redaction altered arguments of {c['function']['name']}")
                calls.append({**c, "function": {**c["function"], "arguments": new_args}})
            msg["tool_calls"] = calls
        messages.append(msg)
    out = {**trace, "messages": messages}
    if trace.get("task_input"):
        ti = trace["task_input"]
        if isinstance(ti, dict):
            out["task_input"] = {k: redact(v) if isinstance(v, str) else v for k, v in ti.items()}
        elif isinstance(ti, str):
            out["task_input"] = redact(ti)
    return out, argument_changed, notes
