"""Ingest served requests back into the trace store.

This closes the loop: what the gateway served becomes the next round's training data. Only requests with a
recorded outcome are taken, because a request nobody graded is not evidence of anything -- which is exactly why
the feedback endpoint exists.

Two filters that matter, and both are about not training on the wrong thing:

- **Escalated turns are the teacher's work, not the student's.** They are still worth having (they are teacher
  demonstrations on prefixes the student built, which is precisely what the student is weak at), but they are
  tagged so a dataset can include or exclude them deliberately.
- **Only successful outcomes reach an SFT set**, via the usual `outcome` filter downstream. Failures are kept
  because DPO needs them.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from agentdistill.config import parse_duration
from agentdistill.ingest.normalize import normalize_trace, validate_trace


def load_traces(
    registry: Any, since: str = "7d", require_outcome: bool = True, limit: int | None = None
) -> tuple[list[dict], list[str]]:
    """Return (traces, problems) from the gateway's request log."""
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
    for row in rows:
        payload = _payload(row)
        messages = payload.get("messages")
        if not messages:
            # The log stores a summary by default, not the conversation. Without `serve --log-messages` there is
            # nothing to train on, and saying so beats silently ingesting nothing.
            problems.append(f"request {row['id']}: no stored messages")
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
            },
        }
        trace = normalize_trace(raw, source="gateway", source_ref=row["id"])
        trace["cluster"] = row.get("cluster_id")
        errors = validate_trace(trace)
        if errors:
            problems.append(f"request {row['id']}: {'; '.join(errors[:2])}")
            continue
        traces.append(trace)
    return traces, problems


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
    }
