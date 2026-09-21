"""Ingest served requests back into the trace store.

This closes the loop: what the gateway served becomes the next round's training data. Only requests with a
recorded outcome are taken, because a request nobody graded is not evidence of anything -- which is exactly why
the feedback endpoint exists.

Two filters that matter, and both are about not training on the wrong thing:

- **Turns written by the teacher are the teacher's output, not the student's traffic.** They are excluded by
  default (`ingest.exclude_teacher_turns`), because training on them is distillation from a provider's model
  through a side door: a decision to do that belongs in the open, after reading the provider's terms, not in a
  default. The rule is deliberately blunt -- a request is dropped whole if *any* of its assistant turns came from
  the teacher arm -- because a conversation with one escalated turn is not separable into a student half and a
  teacher half by anything the log records. The count is logged and written onto the traces the ingest does keep.
- **Only successful outcomes reach an SFT set**, via the usual `outcome` filter downstream. Failures are kept
  because DPO needs them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from agentdistill.config import parse_duration
from agentdistill.ingest.normalize import normalize_trace, validate_trace


@dataclass
class GatewayIngest:
    """What one read of the request log produced, including what it deliberately left out."""

    traces: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: Requests read from the log, before any filtering.
    n_requests: int = 0
    #: Requests dropped whole because an assistant turn came from the teacher arm.
    excluded_teacher_requests: int = 0
    exclude_teacher_turns: bool = True

    @property
    def note(self) -> str:
        """One line for the CLI, whichever way the flag is set. Never silent: a zero says zero."""
        if not self.exclude_teacher_turns:
            return (
                f"ingest.exclude_teacher_turns is false: {self.excluded_teacher_requests} teacher-written "
                f"request(s) were ingested as training data"
            )
        return (
            f"excluded {self.excluded_teacher_requests} teacher-written request(s) of {self.n_requests} "
            f"(ingest.exclude_teacher_turns)"
        )


def teacher_written(row: dict, messages: list[dict]) -> str | None:
    """Why this request counts as teacher-written, or None.

    Three signals, in the order the log makes them available: the arm that answered, the escalation flag (a
    cascade request whose gate sent a turn to the teacher, the fallback path included), and any per-message arm
    marker a payload happens to carry. Normalization strips unknown message keys, so the marker is read here,
    from the raw payload, or not at all.
    """
    if str(row.get("arm") or "").lower() == "teacher":
        return "served by the teacher arm"
    if row.get("escalated"):
        return "the gate escalated at least one turn to the teacher"
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        arm = m.get("arm") or (m.get("metadata") or {}).get("arm")
        if str(arm or "").lower() == "teacher":
            return "an assistant turn in the conversation is marked as the teacher's"
    return None


def load_traces(
    registry: Any,
    since: str = "7d",
    require_outcome: bool = True,
    limit: int | None = None,
    exclude_teacher_turns: bool = True,
) -> tuple[list[dict], list[str]]:
    """Return (traces, problems) from the gateway's request log.

    Kept at two values for the callers that only want those; `load_gateway` returns the counts as well.
    """
    result = load_gateway(registry, since=since, require_outcome=require_outcome, limit=limit,
                          exclude_teacher_turns=exclude_teacher_turns)
    return result.traces, result.problems


def load_gateway(
    registry: Any,
    since: str = "7d",
    require_outcome: bool = True,
    limit: int | None = None,
    exclude_teacher_turns: bool = True,
) -> GatewayIngest:
    """Read the request log into traces, and say what was left out and why."""
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(seconds=parse_duration(since))).isoformat()
    query = "SELECT * FROM requests WHERE received_at >= :since"
    if require_outcome:
        query += " AND outcome IS NOT NULL"
    query += " ORDER BY received_at"
    if limit:
        query += " LIMIT :limit"

    params: dict[str, Any] = {"since": cutoff}
    if limit:
        params["limit"] = limit
    with registry.engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(text(query), params).mappings()]

    traces: list[dict] = []
    problems: list[str] = []
    excluded = 0
    for row in rows:
        payload = _payload(row)
        messages = payload.get("messages")
        if not messages:
            # The log stores a summary by default, not the conversation. Without `serve --log-messages` there is
            # nothing to train on, and saying so beats silently ingesting nothing.
            problems.append(f"request {row['id']}: no stored messages")
            continue
        teacher_reason = teacher_written(row, messages)
        if teacher_reason:
            # Counted whichever way the flag is set: the number is the point, and a filter that reports nothing
            # when it is switched off cannot be audited against one that is switched on.
            excluded += 1
            if exclude_teacher_turns:
                continue
        raw = {
            "task_id": payload.get("task_id") or row["id"],
            "messages": messages,
            "tools": payload.get("tools") or [],
            "success": bool(row["outcome"]) if row["outcome"] is not None else None,
            "grader": "gateway_feedback",
            "teacher_model": payload.get("model"),
            "prompt_tokens": row.get("student_tokens"),
            "completion_tokens": row.get("teacher_tokens"),
            "cost_usd": row.get("cost_usd"),
            "created_at": row.get("received_at"),
            "metadata": {
                "request_id": row["id"],
                "arm": row.get("arm"),
                "escalated": bool(row.get("escalated")),
                "adapter_id": row.get("adapter_id"),
                "confidence": row.get("confidence"),
                "cluster_id": row.get("cluster_id"),
                "teacher_written": teacher_reason,
            },
        }
        trace = normalize_trace(raw, source="gateway", source_ref=row["id"])
        trace["cluster"] = row.get("cluster_id")
        errors = validate_trace(trace)
        if errors:
            problems.append(f"request {row['id']}: {'; '.join(errors[:2])}")
            continue
        traces.append(trace)

    # The ingest has no row of its own -- the traces it writes are its record -- so the exclusion count goes on
    # each of them. Metadata is outside the content hash, so this does not change which traces count as
    # duplicates. See docs/progress.md, "The gateway ingest has no registry row of its own".
    for trace in traces:
        meta = dict(trace.get("metadata") or {})
        meta["ingest"] = {
            "exclude_teacher_turns": exclude_teacher_turns,
            "excluded_teacher_requests": excluded,
            "n_requests": len(rows),
        }
        trace["metadata"] = meta

    return GatewayIngest(
        traces=traces,
        problems=problems,
        n_requests=len(rows),
        excluded_teacher_requests=excluded,
        exclude_teacher_turns=exclude_teacher_turns,
    )


def _payload(row: dict) -> dict:
    raw = row.get("payload")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def summarize(registry: Any, since: str = "7d") -> dict:
    """What the log holds, before anything is ingested. Printed so an empty ingest is explicable."""
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(seconds=parse_duration(since))).isoformat()
    with registry.engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            text("SELECT arm, escalated, outcome FROM requests WHERE received_at >= :s"), {"s": cutoff}
        ).mappings()]
    graded = [r for r in rows if r["outcome"] is not None]
    return {
        "requests": len(rows),
        "graded": len(graded),
        "ungraded": len(rows) - len(graded),
        "successful": sum(1 for r in graded if r["outcome"]),
        "escalated": sum(1 for r in rows if r["escalated"]),
        # What `ingest.exclude_teacher_turns` will drop, said before the ingest runs rather than after.
        "teacher_written": sum(1 for r in rows if teacher_written(r, [])),
    }
