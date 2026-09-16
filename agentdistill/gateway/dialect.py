"""Dialect conversion at the edge.

Requests arrive as OpenAI or Anthropic; everything inside is OpenAI-shaped; responses go back in the dialect they
arrived in. The agent keeps its SDK and never learns that any of this happened.

The conversions reuse `ingest.normalize`, so the shape the gateway serves is the same shape the training data was
built from. Two formats that drift apart would show up as a student that works in eval and fails in production.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from agentdistill.ingest.normalize import anthropic_to_openai

#: Anthropic tool-use ids have a conventional prefix; some clients validate it.
TOOL_USE_PREFIX = "toolu_"


def from_openai_request(body: dict) -> dict:
    return {
        "model": body["model"],
        "messages": body["messages"],
        "tools": body.get("tools") or [],
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 1024,
        "temperature": body.get("temperature", 0.0),
        "stream": bool(body.get("stream")),
    }


def from_anthropic_request(body: dict) -> dict:
    converted = anthropic_to_openai(body.get("system"), body["messages"], body.get("tools") or [])
    return {
        "model": body["model"],
        "messages": converted["messages"],
        "tools": converted["tools"],
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 0.0),
        "stream": bool(body.get("stream")),
    }


def to_openai_response(choice: dict, model: str, usage: dict, request_id: str | None = None) -> dict:
    out = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": _clean_message(choice["message"]),
            "finish_reason": choice.get("finish_reason") or _finish_reason(choice["message"]),
        }],
        "usage": _usage(usage),
    }
    if request_id:
        out["id"] = out["id"]
    return out


def to_anthropic_response(choice: dict, model: str, usage: dict) -> dict:
    message = choice["message"]
    content: list[dict] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        raw = call["function"]["arguments"]
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            # A malformed tool call still has to be representable; the client sees the raw text rather than a
            # dropped block, because silently losing a call is worse than surfacing a bad one.
            args = {"_raw": raw}
        content.append({
            "type": "tool_use",
            "id": _tool_use_id(call.get("id")),
            "name": call["function"]["name"],
            "input": args,
        })
    return {
        "id": f"msg_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _anthropic_stop_reason(choice, message),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }


def _tool_use_id(call_id: Any) -> str:
    if isinstance(call_id, str) and call_id.startswith(TOOL_USE_PREFIX):
        return call_id
    # OpenAI ids look like `call_abc`; the Anthropic SDK expects its own shape.
    return f"{TOOL_USE_PREFIX}{uuid.uuid4().hex[:16]}"


def _finish_reason(message: dict) -> str:
    return "tool_calls" if message.get("tool_calls") else "stop"


def _anthropic_stop_reason(choice: dict, message: dict) -> str:
    if message.get("tool_calls"):
        return "tool_use"
    if choice.get("finish_reason") == "length":
        return "max_tokens"
    return "end_turn"


def _clean_message(message: dict) -> dict:
    """Strip transport-only keys and make sure every tool call carries `type: function`, which the SDK requires."""
    out = {k: v for k, v in message.items() if not k.startswith("_")}
    calls = out.get("tool_calls")
    if calls:
        out["tool_calls"] = [{**c, "type": c.get("type") or "function"} for c in calls]
    else:
        out.pop("tool_calls", None)
    out.setdefault("role", "assistant")
    out.setdefault("content", None)
    return out


def _usage(usage: dict) -> dict:
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens") or prompt + completion),
    }


# --------------------------------------------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------------------------------------------


def openai_stream_events(choice: dict, model: str, usage: dict) -> list[str]:
    """SSE frames for a completed choice.

    The cascade cannot stream honestly -- the gate needs the whole turn before it can decide whether to keep it --
    so a decided turn is emitted as a short stream. The response carries `x-agentdistill-buffered` so a client can
    tell the difference.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    created = int(time.time())
    message = _clean_message(choice["message"])

    def frame(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    events = [frame({"role": "assistant"})]
    if message.get("content"):
        events.append(frame({"content": message["content"]}))
    if message.get("tool_calls"):
        events.append(frame({"tool_calls": [
            {"index": i, "id": c.get("id"), "type": "function",
             "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
            for i, c in enumerate(message["tool_calls"])
        ]}))
    events.append(frame({}, finish=_finish_reason(message)))
    events.append("data: [DONE]\n\n")
    return events


def anthropic_stream_events(choice: dict, model: str, usage: dict) -> list[str]:
    """SSE frames in the Anthropic event shape."""
    response = to_anthropic_response(choice, model, usage)

    def event(name: str, payload: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    events = [event("message_start", {"type": "message_start", "message": {**response, "content": []}})]
    for index, block in enumerate(response["content"]):
        events.append(event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": block["type"], **({"text": ""} if block["type"] == "text" else
                                                        {"id": block["id"], "name": block["name"], "input": {}})},
        }))
        if block["type"] == "text":
            events.append(event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": block["text"]},
            }))
        else:
            events.append(event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
            }))
        events.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": response["stop_reason"], "stop_sequence": None},
        "usage": {"output_tokens": response["usage"]["output_tokens"]},
    }))
    events.append(event("message_stop", {"type": "message_stop"}))
    return events
