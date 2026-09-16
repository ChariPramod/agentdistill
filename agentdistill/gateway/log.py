"""The request log.

Every request the gateway serves is recorded: which arm answered, whether the gate escalated, what it cost, and
later -- once a grader or a human says so -- whether it worked. That last column is what makes the log the
source for the next training round, so a row with no outcome is not a failure, it is simply not yet useful.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from agentdistill.registry.base import dumps, utcnow


class RequestLog:
    """Writes to the `requests` table. Async-safe by serializing writes behind a lock.

    SQLite does not take concurrent writers well, and the gateway is the one component where several requests
    land at once.
    """

    def __init__(self, registry: Any) -> None:
        self.registry = registry
        self._lock = asyncio.Lock()

    async def write(self, meta: dict, request: dict, choice: dict, usage: dict) -> str:
        from sqlalchemy import text

        request_id = meta.get("id") or uuid.uuid4().hex
        params = {
            "id": request_id,
            "received_at": utcnow(),
            "cluster_id": meta.get("cluster_id"),
            "arm": meta.get("arm") or meta.get("route") or "unknown",
            "adapter_id": meta.get("adapter"),
            "confidence": meta.get("confidence"),
            "escalated": bool(meta.get("escalated", False)),
            "student_tokens": meta.get("student_tokens"),
            "teacher_tokens": meta.get("teacher_tokens"),
            "cost_usd": meta.get("cost_usd"),
            "latency_ms": meta.get("latency_ms"),
            "outcome": None,
            "trace_id": None,
            "payload": dumps({
                "model": request.get("model"),
                "n_messages": len(request.get("messages") or []),
                "n_tools": len(request.get("tools") or []),
                "route": meta.get("route"),
                "usage": usage,
                "wasted_student_tokens": meta.get("wasted_student_tokens"),
            }),
        }
        async with self._lock:
            with self.registry.engine.begin() as conn:
                conn.execute(
                    text(
                        """INSERT INTO requests (id, received_at, cluster_id, arm, adapter_id, confidence,
                                                 escalated, student_tokens, teacher_tokens, cost_usd, latency_ms,
                                                 outcome, trace_id, payload)
                           VALUES (:id, :received_at, :cluster_id, :arm, :adapter_id, :confidence, :escalated,
                                   :student_tokens, :teacher_tokens, :cost_usd, :latency_ms, :outcome, :trace_id,
                                   :payload)"""
                    ),
                    params,
                )
        return request_id

    async def set_outcome(self, request_id: str, success: bool) -> bool:
        from sqlalchemy import text

        async with self._lock:
            with self.registry.engine.begin() as conn:
                result = conn.execute(
                    text("UPDATE requests SET outcome = :o WHERE id = :id"),
                    {"o": bool(success), "id": request_id},
                )
        return bool(result.rowcount)

    async def get(self, request_id: str) -> dict | None:
        from sqlalchemy import text

        with self.registry.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM requests WHERE id = :id"), {"id": request_id}).mappings().first()
        return dict(row) if row else None

    def recent(self, limit: int = 100) -> list[dict]:
        from sqlalchemy import text

        with self.registry.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM requests ORDER BY received_at DESC LIMIT :n"), {"n": limit}
            ).mappings().fetchall()
        return [dict(r) for r in rows]
