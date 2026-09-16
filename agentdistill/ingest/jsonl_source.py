"""Ingest normalized traces from JSONL, one trace per line.

Also accepts raw Anthropic-shaped records (`{"system":..., "messages":[...], "tools":[...]}` with content blocks)
and converts them, because that is what a hand-rolled logger usually writes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from agentdistill.ingest.normalize import anthropic_to_openai, normalize_trace, validate_trace


class IngestError(Exception):
    """A record that cannot be read at all. Carries the line number so the user can go fix it."""


def _looks_anthropic(record: dict) -> bool:
    """Anthropic records carry content blocks rather than plain strings, and tools without a `type` wrapper."""
    for m in record.get("messages", []):
        if isinstance(m.get("content"), list):
            blocks = m["content"]
            if blocks and isinstance(blocks[0], dict) and blocks[0].get("type") in {"text", "tool_use", "tool_result"}:
                return True
    tools = record.get("tools") or []
    return bool(tools) and isinstance(tools[0], dict) and "input_schema" in tools[0]


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict]]:
    p = Path(path)
    if not p.exists():
        raise IngestError(f"no such file: {p}")
    with p.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise IngestError(f"{p}:{lineno}: not valid JSON: {e}") from e
            if not isinstance(record, dict):
                raise IngestError(f"{p}:{lineno}: expected a JSON object, got {type(record).__name__}")
            yield lineno, record


def load_traces(path: str | Path, strict: bool = True) -> tuple[list[dict], list[str]]:
    """Return (traces, problems).

    In strict mode a record that fails schema validation raises; otherwise it is skipped and reported. Strict is
    the default because a silently dropped trace is a silently smaller dataset.
    """
    p = Path(path)
    traces: list[dict] = []
    problems: list[str] = []
    for lineno, record in read_jsonl(p):
        if "messages" not in record:
            msg = f"{p}:{lineno}: record has no `messages`"
            if strict:
                raise IngestError(msg)
            problems.append(msg)
            continue
        if _looks_anthropic(record):
            converted = anthropic_to_openai(record.get("system"), record["messages"], record.get("tools") or [])
            record = {**record, **converted}
            record.pop("system", None)
        trace = normalize_trace(record, source="jsonl", source_ref=str(p))
        errs = validate_trace(trace)
        if errs:
            msg = f"{p}:{lineno}: {'; '.join(errs[:3])}"
            if strict:
                raise IngestError(msg)
            problems.append(msg)
            continue
        traces.append(trace)
    return traces, problems
