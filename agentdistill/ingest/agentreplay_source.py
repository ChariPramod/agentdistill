"""Ingest from an agentreplay store (SQLite or Postgres).

agentreplay records a run as a sequence of `llm_calls` and `tool_calls` rather than as one trajectory. The
trajectory is reconstructed from the **last** llm_call's request messages plus its response: by the final call the
request already contains every prior turn and every tool result, so this reconstruction is exact and does not
depend on our guessing how the agent assembled its context.

Success comes from `eval_results` when the run was graded, otherwise from `labels`. Runs with neither are ingested
with `success = NULL`; the `outcome` filter drops them from SFT, but they remain available for DPO negatives and
for clustering statistics.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text

from agentdistill.config import parse_duration
from agentdistill.ingest.normalize import anthropic_to_openai, normalize_trace, validate_trace


class AgentReplayError(Exception):
    pass


def _url_for(db: str | Path) -> str:
    s = str(db)
    if "://" in s:
        return s
    p = Path(s)
    if not p.exists():
        raise AgentReplayError(f"no agentreplay store at {p}")
    return f"sqlite:///{p.resolve()}"


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _since_iso(since: str) -> str:
    return (datetime.now(UTC) - timedelta(seconds=parse_duration(since))).isoformat()


def load_traces(db: str | Path, since: str = "30d", limit: int | None = None) -> tuple[list[dict], list[str]]:
    """Return (traces, problems). Problems name the run id so a user can inspect it in agentreplay."""
    engine = create_engine(_url_for(db), future=True)
    tables = set(inspect(engine).get_table_names())
    missing = {"runs", "llm_calls"} - tables
    if missing:
        raise AgentReplayError(
            f"{db} does not look like an agentreplay store: missing table(s) {sorted(missing)}. "
            f"Found: {sorted(tables)}"
        )

    traces: list[dict] = []
    problems: list[str] = []
    with engine.connect() as conn:
        q = "SELECT * FROM runs WHERE started_at >= :since ORDER BY started_at"
        if limit:
            q += " LIMIT :limit"
        params: dict[str, Any] = {"since": _since_iso(since)}
        if limit:
            params["limit"] = limit
        runs = conn.execute(text(q), params).mappings().fetchall()

        outcomes = _outcomes(conn, tables)

        for run in runs:
            run_id = run["id"]
            calls = (
                conn.execute(
                    text("SELECT * FROM llm_calls WHERE run_id = :r ORDER BY started_at, id"), {"r": run_id}
                )
                .mappings()
                .fetchall()
            )
            if not calls:
                problems.append(f"run {run_id}: no llm_calls")
                continue
            try:
                trace = _trace_from_run(run, calls, outcomes.get(run_id, {}))
            except (KeyError, TypeError, ValueError) as e:
                problems.append(f"run {run_id}: {e}")
                continue
            errs = validate_trace(trace)
            if errs:
                problems.append(f"run {run_id}: {'; '.join(errs[:3])}")
                continue
            traces.append(trace)
    engine.dispose()
    return traces, problems


def _outcomes(conn: Any, tables: set[str]) -> dict[str, dict]:
    """Prefer a graded eval_result; fall back to a human label. Both are keyed by run id."""
    out: dict[str, dict] = {}
    if "labels" in tables:
        for r in conn.execute(text("SELECT * FROM labels")).mappings():
            rid = r.get("run_id")
            if rid is None:
                continue
            out[rid] = {"success": _as_bool(r.get("value") if "value" in r else r.get("success")), "grader": "label"}
    if "eval_results" in tables:
        for r in conn.execute(text("SELECT * FROM eval_results")).mappings():
            rid = r.get("run_id")
            if rid is None:
                continue
            out[rid] = {
                "success": _as_bool(r.get("passed") if "passed" in r else r.get("success")),
                "grader": r.get("grader") or "eval",
                "score": r.get("score"),
            }
    return out


def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "pass", "passed", "yes", "success"}
    return None


def _trace_from_run(run: Any, calls: Sequence[Any], outcome: dict) -> dict:
    """Rebuild the trajectory from the last llm_call's request plus its response."""
    last = calls[-1]
    request = _json(last.get("request")) or {}
    response = _json(last.get("response")) or {}

    messages = request.get("messages")
    if not messages:
        raise ValueError("last llm_call has no request messages; the store was recorded without content")
    tools = request.get("tools") or []

    # Anthropic-dialect stores keep the system prompt outside `messages` and use content blocks.
    if request.get("system") is not None or _has_blocks(messages):
        converted = anthropic_to_openai(request.get("system"), messages, tools)
        messages, tools = converted["messages"], converted["tools"]

    final = _final_assistant_message(response)
    if final is not None:
        messages = [*messages, final]

    prompt_tokens = sum(int(c.get("prompt_tokens") or 0) for c in calls) or None
    completion_tokens = sum(int(c.get("completion_tokens") or 0) for c in calls) or None
    cost = sum(float(c.get("cost_usd") or 0.0) for c in calls) or None

    raw = {
        "task_id": run.get("task_id") or run.get("id"),
        "messages": messages,
        "tools": tools,
        "teacher_model": last.get("model") or run.get("model"),
        "success": outcome.get("success"),
        "grader": outcome.get("grader"),
        "score": outcome.get("score"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": cost,
        "created_at": _iso(run.get("started_at")),
        "metadata": {"agentreplay_run_id": run["id"], "n_llm_calls": len(calls)},
    }
    return normalize_trace(raw, source="agentreplay", source_ref=str(run["id"]))


def _has_blocks(messages: list[dict]) -> bool:
    return any(isinstance(m.get("content"), list) for m in messages)


def _final_assistant_message(response: dict) -> dict | None:
    """The last response holds the turn that is not yet echoed in the request messages."""
    if not response:
        return None
    # OpenAI shape
    choices = response.get("choices")
    if choices:
        msg = choices[0].get("message") or {}
        if msg.get("content") or msg.get("tool_calls"):
            return {
                "role": "assistant",
                "content": msg.get("content"),
                **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {}),
            }
        return None
    # Anthropic shape
    if response.get("content") is not None and response.get("role") in (None, "assistant"):
        converted = anthropic_to_openai(None, [{"role": "assistant", "content": response["content"]}], [])
        msgs = converted["messages"]
        return msgs[0] if msgs else None
    return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)
