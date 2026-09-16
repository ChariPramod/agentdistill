"""Anthropic <-> OpenAI message conversion, canonicalization, and content hashing.

Every source is converted to one shape so curation, tokenization, and eval never branch on provider. The converter
is bijective enough that the gateway can answer in either dialect: `anthropic -> openai -> anthropic` preserves tool
names, arguments, and tool-result pairing. It does not preserve every block-level detail (a tool_result carrying a
list of content blocks comes back as flattened text), which is why `roundtrip_equivalent` compares the parts that
matter rather than raw equality.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "trace.schema.json"

_ROLES_WITH_TEXT = {"system", "user"}


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    with SCHEMA_PATH.open() as fh:
        return Draft202012Validator(json.load(fh))


# --------------------------------------------------------------------------------------------------------------
# Anthropic -> OpenAI
# --------------------------------------------------------------------------------------------------------------


def anthropic_to_openai(system: str | list | None, messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Convert an Anthropic Messages-API conversation to normalized OpenAI-style messages."""
    out: list[dict] = []
    if system:
        text = system if isinstance(system, str) else "".join(b.get("text", "") for b in system)
        if text:
            out.append({"role": "system", "content": text})
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)},
                }
                for b in content
                if b["type"] == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn: may hold tool_result blocks and/or text
            for b in content:
                if b["type"] == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "content": _flatten_tool_result(b.get("content")),
                        }
                    )
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    return {"messages": out, "tools": anthropic_tools_to_openai(tools or [])}


def _flatten_tool_result(content: Any) -> str:
    """Anthropic tool_result content is a string or a list of blocks; the normalized form is always a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict) and x.get("type") == "text":
                parts.append(x.get("text", ""))
            else:
                parts.append(json.dumps(x, ensure_ascii=False, sort_keys=True))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object"}),
            },
        }
        for t in tools
    ]


# --------------------------------------------------------------------------------------------------------------
# OpenAI -> Anthropic
# --------------------------------------------------------------------------------------------------------------


def openai_to_anthropic(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Inverse of `anthropic_to_openai`. Returns {system, messages, tools} in Anthropic shape.

    Consecutive `role=tool` messages coalesce into one user turn of tool_result blocks, which is how Anthropic
    requires parallel tool results to be returned.
    """
    system_parts: list[str] = []
    out: list[dict] = []
    pending_results: list[dict] = []

    def flush_results() -> None:
        nonlocal pending_results
        if pending_results:
            out.append({"role": "user", "content": pending_results})
            pending_results = []

    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m.get("content") or "")
            continue
        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m.get("content") or "",
                }
            )
            continue
        flush_results()
        if role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m.get("content") or ""}]})
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": c["id"],
                        "name": c["function"]["name"],
                        "input": _loads_or_raw(c["function"]["arguments"]),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            raise ValueError(f"unknown role {role!r}")
    flush_results()

    return {
        "system": "\n".join(p for p in system_parts if p) or None,
        "messages": out,
        "tools": openai_tools_to_anthropic(tools or []),
    }


def _loads_or_raw(arguments: str) -> Any:
    """Tool arguments that are not valid JSON are kept verbatim; the schema_valid filter is what rejects them,
    not the converter. Silently dropping a malformed call would hide exactly the traces curation must see."""
    try:
        return json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments


def openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = t["function"]
        entry: dict[str, Any] = {"name": fn["name"], "input_schema": fn.get("parameters", {"type": "object"})}
        if fn.get("description"):
            entry["description"] = fn["description"]
        out.append(entry)
    return out


# --------------------------------------------------------------------------------------------------------------
# Canonicalization and hashing
# --------------------------------------------------------------------------------------------------------------


def canonical_arguments(arguments: str) -> str:
    """Canonical form of a tool call's arguments: key order and whitespace normalized so that two calls that mean
    the same thing hash the same. Unparseable arguments hash as their raw text."""
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return f"<raw>{arguments}"
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_tool_call(call: dict) -> list[Any]:
    """Call ids are generated per request and carry no meaning, so they are excluded from the canonical form."""
    return [call["function"]["name"], canonical_arguments(call["function"]["arguments"])]


def canonical_message(m: dict) -> list[Any]:
    role = m["role"]
    content = m.get("content") or ""
    calls = [canonical_tool_call(c) for c in (m.get("tool_calls") or [])]
    # tool_call_id is positional information already implied by message order, and it is request-scoped noise.
    return [role, content, calls]


def canonical_messages(messages: list[dict]) -> str:
    return json.dumps(
        [canonical_message(m) for m in messages], sort_keys=False, separators=(",", ":"), ensure_ascii=False
    )


def canonical_tools(tools: list[dict]) -> str:
    """Tool order varies between requests without changing meaning, so the canonical form sorts by name."""
    entries = sorted(
        (
            [
                t["function"]["name"],
                t["function"].get("description", ""),
                json.dumps(t["function"].get("parameters", {}), sort_keys=True, separators=(",", ":")),
            ]
            for t in tools
        ),
        key=lambda e: e[0],
    )
    return json.dumps(entries, separators=(",", ":"), ensure_ascii=False)


def content_hash(trace: dict) -> str:
    """sha256 over the canonicalized messages and tools. Two traces with the same hash are the same trajectory."""
    payload = canonical_messages(trace["messages"]) + "\x00" + canonical_tools(trace.get("tools") or [])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------------------------
# Trace assembly and validation
# --------------------------------------------------------------------------------------------------------------


def count_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


def count_tool_calls(messages: list[dict]) -> int:
    return sum(len(m.get("tool_calls") or []) for m in messages)


def extract_task_input(messages: list[dict]) -> dict | None:
    """The task as posed, before any assistant turn: the system prompt plus the leading user turns.

    This is what gets embedded for clustering and n-gram matched for decontamination, so it must not include
    anything the assistant produced.
    """
    system = next((m.get("content") or "" for m in messages if m["role"] == "system"), "")
    user_parts: list[str] = []
    for m in messages:
        if m["role"] == "assistant":
            break
        if m["role"] == "user":
            user_parts.append(m.get("content") or "")
    if not system and not user_parts:
        return None
    return {"system": system, "user": "\n".join(user_parts)}


def normalize_trace(raw: dict, source: str, source_ref: str | None = None) -> dict:
    """Fill in derived fields and return a trace that satisfies `schemas/trace.schema.json`.

    Derived fields are always recomputed, never trusted from the input: a stale `n_tool_calls` in a JSONL export
    would otherwise silently corrupt the length filter.
    """
    messages = _clean_messages(raw["messages"])
    tools = raw.get("tools") or []
    trace: dict[str, Any] = {
        "source": source,
        "source_ref": source_ref if source_ref is not None else raw.get("source_ref"),
        "task_id": raw.get("task_id"),
        "task_input": raw.get("task_input") or extract_task_input(messages),
        "messages": messages,
        "tools": tools,
        "teacher_model": raw.get("teacher_model"),
        "success": raw.get("success"),
        "grader": raw.get("grader"),
        "score": raw.get("score"),
        "prompt_tokens": raw.get("prompt_tokens"),
        "completion_tokens": raw.get("completion_tokens"),
        "cost_usd": raw.get("cost_usd"),
        "created_at": raw.get("created_at"),
        "metadata": raw.get("metadata"),
    }
    trace["n_turns"] = count_turns(messages)
    trace["n_tool_calls"] = count_tool_calls(messages)
    trace["content_hash"] = content_hash(trace)
    trace["id"] = raw.get("id") or f"tr_{trace['content_hash'][:16]}"
    trace["cluster"] = raw.get("cluster")
    return trace


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Drop keys the schema does not allow and normalize absent-vs-null so hashing is stable across sources."""
    out = []
    for m in messages:
        msg: dict[str, Any] = {"role": m["role"]}
        content = m.get("content")
        if content is not None or m["role"] in _ROLES_WITH_TEXT:
            msg["content"] = content if content is not None else ""
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]},
                }
                for c in m["tool_calls"]
            ]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            msg["name"] = m["name"]
        if m["role"] == "assistant" and "content" not in msg:
            msg["content"] = None
        out.append(msg)
    return out


def validate_trace(trace: dict) -> list[str]:
    """Return a list of schema problems; empty means valid."""
    return [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in _validator().iter_errors(trace)]


def roundtrip_equivalent(a: dict, b: dict) -> bool:
    """True when two normalized traces agree on everything that training depends on."""
    return canonical_messages(a["messages"]) == canonical_messages(b["messages"]) and canonical_tools(
        a.get("tools") or []
    ) == canonical_tools(b.get("tools") or [])
