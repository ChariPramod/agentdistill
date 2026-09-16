"""A plain tool-calling loop over any LiteLLM-supported endpoint.

Nothing clever: this is the agent whose traces get distilled, so it should look like an ordinary production loop.
Tool errors are fed back to the model as tool results rather than raised, because recovering from a refusal is
exactly the behaviour worth learning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT = (HERE / "system_prompt.md").read_text().strip()
#: Bumped whenever the prompt changes, and recorded on every trace: traces recorded under different prompts are
#: not the same distribution and should not be mixed in one dataset.
SYSTEM_PROMPT_VERSION = "v1"

MAX_TURNS = 12


class AgentRun(dict):
    """The result of one episode: messages, tools, usage, latency."""


def run_agent(
    task: Any,
    crm: Any,
    tools: list[dict],
    model: str,
    max_turns: int = MAX_TURNS,
    temperature: float = 0.2,
    completion: Any = None,
) -> AgentRun:
    """Run one episode against a live CRM.

    `completion` defaults to `litellm.completion` and is injectable so the scripted agent and the tests can drive
    the same loop without a network call.
    """
    if completion is None:
        import litellm

        completion = litellm.completion

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.user_message},
    ]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    started = time.time()
    stop_reason = "max_turns"

    for _ in range(max_turns):
        response = completion(model=model, messages=messages, tools=tools, temperature=temperature)
        choice = response.choices[0].message
        if getattr(response, "usage", None):
            usage["prompt_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
            usage["completion_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0

        assistant: dict[str, Any] = {"role": "assistant", "content": choice.content}
        tool_calls = getattr(choice, "tool_calls", None)
        if tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in tool_calls
            ]
        messages.append(assistant)

        if not tool_calls:
            stop_reason = "answered"
            break

        for call in assistant["tool_calls"]:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                content = json.dumps({"error": "arguments were not valid JSON"})
            else:
                try:
                    content = json.dumps(crm.call(name, args))
                except Exception as e:
                    # A tool refusal is part of the trajectory. Recovering from it is the interesting behaviour.
                    content = json.dumps({"error": str(e)})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    return AgentRun(
        messages=messages,
        tools=tools,
        usage=usage,
        latency_ms=int((time.time() - started) * 1000),
        stop_reason=stop_reason,
        system_prompt_version=SYSTEM_PROMPT_VERSION,
    )


def final_assistant_text(messages: list[dict]) -> str:
    return next((m.get("content") or "" for m in reversed(messages) if m["role"] == "assistant"), "")
