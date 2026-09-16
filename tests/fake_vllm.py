"""A fake vLLM server for tests.

The gateway talks to vLLM over an OpenAI-compatible HTTP API, so the whole gateway can be tested on CPU against
this. Tests push scripted assistant messages onto `QUEUE`; the server renders them the way a served model would
-- as text containing `<tool_call>` blocks -- so the gateway's parsing path is the real one, not a shortcut.
"""

from __future__ import annotations

import json
import uuid
from collections import deque

from fastapi import FastAPI

app = FastAPI()

#: Tests push assistant messages here; each request pops one per requested sample.
QUEUE: deque[dict] = deque()
#: Every request body the gateway sent, for assertions about n, logprobs, and tools.
CALLS: list[dict] = []
#: Per-token logprob the fake reports. Tests lower it to drive the cascade's gate.
LOGPROB = -0.2


def reset(logprob: float = -0.2) -> None:
    QUEUE.clear()
    CALLS.clear()
    global LOGPROB
    LOGPROB = logprob


def push(*messages: dict) -> None:
    QUEUE.extend(messages)


def _tokens(text: str) -> list[dict]:
    """Token records shaped like a served model's, so the feature code sees its real input format."""
    out: list[dict] = []
    for word in text.split(" "):
        token = word + " "
        out.append({
            "token": token,
            "logprob": LOGPROB,
            "top_logprobs": [{"token": token, "logprob": LOGPROB}, {"token": "x", "logprob": -3.0}],
        })
    if out:
        out[-1]["token"] = out[-1]["token"].rstrip(" ")
    return out


def render(message: dict) -> str:
    """What a served model would actually emit: prose plus hermes-style tool-call blocks."""
    text = message.get("content") or ""
    for call in message.get("tool_calls") or []:
        raw = call["function"]["arguments"]
        args = json.loads(raw) if isinstance(raw, str) else raw
        text += f'<tool_call>{json.dumps({"name": call["function"]["name"], "arguments": args})}</tool_call>'
    return text


@app.post("/v1/chat/completions")
async def chat(body: dict):
    CALLS.append(body)
    n = int(body.get("n", 1) or 1)
    choices = []
    for i in range(n):
        message = QUEUE.popleft() if QUEUE else {"role": "assistant", "content": "ok"}
        text = render(message)
        choice: dict = {
            "index": i,
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
        }
        if body.get("logprobs"):
            choice["logprobs"] = {"content": _tokens(text)}
        choices.append(choice)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "model": body.get("model", "student"),
        "choices": choices,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "student", "object": "model"},
                                       {"id": "student:support-v1", "object": "model"}]}


@app.get("/health")
async def health():
    return {"ok": True}
