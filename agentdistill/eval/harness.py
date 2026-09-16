"""The task runner.

Runs one task: start from the recorded system and user messages, let the client produce turns, serve tool results
from the recording, stop when it answers, diverges, or runs out of turns.

Divergence stops the trajectory. The task is then graded as it stands -- usually a failure, since the work was not
finished -- but divergence is reported as its own metric, because "the student went somewhere the recording
cannot follow" and "the student did the wrong thing" are different problems with different fixes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from agentdistill.curate.schema import tool_calls_valid
from agentdistill.eval.replay import Divergence, ReplayToolProvider
from agentdistill.eval.teacher_forced import TurnClient


#: Roughly four characters per token. Only used for a relative comparison between subjects on the same tasks;
#: the real token counts come from the gateway once it exists.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class TaskOutcome:
    task_id: str
    repeat_idx: int
    messages: list[dict]
    final_text: str
    n_turns: int
    n_tool_calls: int
    schema_valid: bool
    diverged: bool
    divergence: dict | None
    replay_stats: dict
    latency_ms: int
    completion_tokens_est: int
    stop_reason: str
    #: Cascade subjects only: how many turns the gate sent to the teacher, and what the discarded student
    #: generations cost. Zero for a plain student or teacher run.
    escalations: int = 0
    wasted_student_tokens: int = 0
    success: bool | None = None
    grader_detail: str = ""
    grader_out: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "task_id": self.task_id,
            "repeat_idx": self.repeat_idx,
            "final_text": self.final_text,
            "n_turns": self.n_turns,
            "n_tool_calls": self.n_tool_calls,
            "schema_valid": self.schema_valid,
            "diverged": self.diverged,
            "divergence": self.divergence,
            "replay_stats": self.replay_stats,
            "latency_ms": self.latency_ms,
            "completion_tokens_est": self.completion_tokens_est,
            "stop_reason": self.stop_reason,
            "escalations": self.escalations,
            "wasted_student_tokens": self.wasted_student_tokens,
            "success": self.success,
            "grader_detail": self.grader_detail,
        }


def initial_messages(trace: dict) -> list[dict]:
    """The task as posed: the system prompt and the first user turn, and nothing the assistant produced."""
    out: list[dict] = []
    for m in trace["messages"]:
        if m["role"] == "assistant":
            break
        if m["role"] in ("system", "user"):
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def run_task(
    trace: dict,
    client: TurnClient,
    provider: ReplayToolProvider,
    repeat_idx: int = 0,
    max_turns: int = 12,
) -> TaskOutcome:
    """Run one task against the replayed environment."""
    tools = trace.get("tools") or []
    messages = initial_messages(trace)
    started = time.time()
    n_calls = 0
    tokens = 0
    diverged = False
    divergence: dict | None = None
    stop_reason = "max_turns"

    for _ in range(max_turns):
        assistant = {k: v for k, v in client.next_turn(messages, tools).items() if not k.startswith("_")}
        messages.append(assistant)
        tokens += estimate_tokens((assistant.get("content") or "") + json.dumps(assistant.get("tool_calls") or []))

        calls = assistant.get("tool_calls")
        if not calls:
            stop_reason = "answered"
            break

        for call in calls:
            n_calls += 1
            raw = call["function"]["arguments"]
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                # Malformed arguments are the student's mistake, not a divergence: the environment can answer,
                # and a real tool would reject it the same way.
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"error": "arguments were not valid JSON"}),
                })
                continue
            try:
                content = provider.lookup(call["function"]["name"], args)
            except Divergence as d:
                diverged, divergence, stop_reason = True, d.to_dict(), "diverged"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"error": "replay divergence: this call was not recorded"}),
                })
                break
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        if diverged:
            break

    final_text = next((m.get("content") or "" for m in reversed(messages) if m["role"] == "assistant"), "")
    schema_ok, _ = tool_calls_valid({"messages": messages, "tools": tools})

    # A cascade client knows what the gate did; a plain client does not have a summary and reports zeros.
    gate = client.summary() if hasattr(client, "summary") else {}

    return TaskOutcome(
        task_id=trace.get("task_id") or trace["id"],
        repeat_idx=repeat_idx,
        messages=messages,
        final_text=final_text,
        n_turns=sum(1 for m in messages if m["role"] == "assistant"),
        n_tool_calls=n_calls,
        schema_valid=schema_ok,
        diverged=diverged,
        divergence=divergence,
        replay_stats=provider.summary(),
        latency_ms=int((time.time() - started) * 1000),
        completion_tokens_est=tokens,
        stop_reason=stop_reason,
        escalations=int(gate.get("escalations", 0)),
        wasted_student_tokens=int(gate.get("wasted_student_tokens", 0)),
    )


def tool_calls_made(outcome: TaskOutcome) -> list[tuple[str, dict]]:
    """Every (tool, args) the student issued, in order. The replay grader replays these against a fresh state."""
    out: list[tuple[str, dict]] = []
    for m in outcome.messages:
        for c in m.get("tool_calls") or []:
            raw = c["function"]["arguments"]
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            out.append((c["function"]["name"], args))
    return out
