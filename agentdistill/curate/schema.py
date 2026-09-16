"""Tool-call schema validation.

A trace whose tool calls do not validate teaches the student to emit arguments the serving stack will reject.
This filter is the cheapest quality win in the pipeline.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


@lru_cache(maxsize=512)
def _validator_for(schema_json: str) -> Draft202012Validator | None:
    """Validators are expensive to build and tools repeat across every trace in a project, so cache by schema text.
    A tool whose own schema is invalid cannot judge anything; that is reported separately rather than failing
    every trace that uses it."""
    try:
        schema = json.loads(schema_json)
        Draft202012Validator.check_schema(schema)
    except (json.JSONDecodeError, SchemaError):
        return None
    return Draft202012Validator(schema)


def tool_schemas(trace: dict) -> dict[str, Any]:
    return {
        t["function"]["name"]: t["function"].get("parameters", {"type": "object"})
        for t in trace.get("tools") or []
    }


def tool_calls_valid(trace: dict) -> tuple[bool, list[str]]:
    """Return (ok, problems). Every problem names the turn index and the tool so the curation report is actionable."""
    schemas = tool_schemas(trace)
    problems: list[str] = []
    for i, m in enumerate(trace["messages"]):
        for c in m.get("tool_calls") or []:
            name = c["function"]["name"]
            if name not in schemas:
                problems.append(f"turn {i}: unknown tool {name}")
                continue
            try:
                args = json.loads(c["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                problems.append(f"turn {i}: {name} arguments are not JSON")
                continue
            if not isinstance(args, dict):
                problems.append(f"turn {i}: {name} arguments are not a JSON object")
                continue
            validator = _validator_for(json.dumps(schemas[name], sort_keys=True))
            if validator is None:
                problems.append(f"turn {i}: {name} has an invalid parameter schema; cannot validate")
                continue
            for err in validator.iter_errors(args):
                problems.append(f"turn {i}: {name}: {err.message}")
    return (not problems), problems


def tool_results_paired(trace: dict) -> tuple[bool, list[str]]:
    """Every tool call must have a matching result, and every result must answer a call that was made.

    An unanswered call means the trace was truncated mid-flight; training on it teaches the student to stop
    after calling a tool.
    """
    problems: list[str] = []
    pending: dict[str, int] = {}
    for i, m in enumerate(trace["messages"]):
        if m["role"] == "tool":
            tcid = m.get("tool_call_id")
            if tcid not in pending:
                problems.append(f"turn {i}: tool result {tcid} answers no call")
            else:
                pending.pop(tcid)
            continue
        for c in m.get("tool_calls") or []:
            pending[c["id"]] = i
    for tcid, turn in sorted(pending.items(), key=lambda kv: kv[1]):
        problems.append(f"turn {turn}: tool call {tcid} has no result")
    return (not problems), problems
