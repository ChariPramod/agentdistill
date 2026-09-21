"""TEST ORACLE for the lockstep and sequential runners.

Deliberately simple, one task at a time, and independent of agentdistill.eval's shared stepper.
Do not refactor this to share code with the production runners: its only value is that it does not.
It reproduces the harness's documented conventions (the two error payloads below), which are the
contract, not an implementation detail.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

BAD_JSON = json.dumps({"error": "arguments were not valid JSON"})
DIVERGED = json.dumps({"error": "replay divergence: this call was not recorded"})


@dataclass
class OracleOutcome:
    task_id: str
    messages: list[dict]
    n_turns: int
    n_tool_calls: int
    diverged: bool
    token_estimate: int


def run_reference(trace: dict, next_turn: Callable[[list[dict], list[dict]], dict],
                  lookup: Callable[[str, dict], tuple[bool, str]], max_turns: int = 12,
                  token_estimate: Callable[[str], int] = lambda s: len(s) // 4) -> OracleOutcome:
    """next_turn(messages, tools) -> assistant message. lookup(tool, args) -> (found, content)."""
    msgs = [m for m in trace["messages"][:2] if m["role"] in ("system", "user")]
    n_calls, est, diverged = 0, 0, False
    for _ in range(max_turns):
        reply = next_turn([dict(m) for m in msgs], trace["tools"])
        a = {k: v for k, v in reply.items() if not k.startswith("_")}
        msgs.append(a)
        est += token_estimate((a.get("content") or "") + json.dumps(a.get("tool_calls") or []))
        calls = a.get("tool_calls") or []
        if not calls:
            break
        for c in calls:
            n_calls += 1
            raw = c["function"]["arguments"]
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": BAD_JSON})
                continue
            found, content = lookup(c["function"]["name"], args)
            if not found:
                diverged = True
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": DIVERGED})
                break
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": content})
        if diverged:
            break
    return OracleOutcome(trace["task_id"], msgs, sum(1 for m in msgs if m["role"] == "assistant"), n_calls, diverged, est)


def recorded_lookup(trace: dict) -> Callable[[str, dict], tuple[bool, str]]:
    """Exact-match replay keyed on (tool, sorted-key JSON args). Intentionally cruder than canonical hashing,
    so use it only on fixtures whose arguments are already canonical."""
    by_id = {c["id"]: c for m in trace["messages"] for c in (m.get("tool_calls") or [])}
    table: dict[tuple[str, str], str] = {}
    for m in trace["messages"]:
        if m["role"] == "tool":
            c = by_id[m["tool_call_id"]]
            a = c["function"]["arguments"]
            key = (c["function"]["name"], json.dumps(json.loads(a) if isinstance(a, str) else a, sort_keys=True))
            table.setdefault(key, m["content"])

    def lookup(tool: str, args: dict) -> tuple[bool, str]:
        hit = table.get((tool, json.dumps(args, sort_keys=True)))
        return (hit is not None, hit or "")
    return lookup
