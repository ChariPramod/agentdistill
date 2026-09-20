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


class TaskStepper:
    """One task's trajectory as a state machine: ask it for the next request, feed it the reply.

    The sequential `run_task` and the batched lockstep runner both drive this, so there is exactly one copy of the
    per-turn rules (tool-call replay, malformed arguments, divergence, stop reasons, the turn budget, the token
    estimate). Two copies would drift, and the batched throughput figure is only worth anything if the batched
    runner produces the same trajectories as the sequential one.
    """

    def __init__(self, trace: dict, provider: ReplayToolProvider, repeat_idx: int = 0, max_turns: int = 12) -> None:
        self.trace, self.provider = trace, provider
        self.repeat_idx, self.max_turns = repeat_idx, max_turns
        self.tools = trace.get("tools") or []
        self.messages = initial_messages(trace)
        self.started = time.time()
        self.n_calls = 0
        self.tokens = 0
        self.turns_taken = 0
        self.diverged = False
        self.divergence: dict | None = None
        self.stop_reason = "max_turns"
        self.done = max_turns <= 0

    @property
    def request(self) -> tuple[list[dict], list[dict]]:
        """What the client is asked for next: the trajectory so far and the task's tools."""
        return self.messages, self.tools

    def step(self, reply: dict) -> bool:
        """Apply one assistant turn. Returns True when the trajectory has ended."""
        if self.done:
            raise RuntimeError("step() called on a finished task")
        self.turns_taken += 1
        assistant = {k: v for k, v in reply.items() if not k.startswith("_")}
        messages = self.messages
        messages.append(assistant)
        self.tokens += estimate_tokens(
            (assistant.get("content") or "") + json.dumps(assistant.get("tool_calls") or [])
        )

        calls = assistant.get("tool_calls")
        if not calls:
            self.stop_reason = "answered"
            self.done = True
            return True

        for call in calls:
            self.n_calls += 1
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
                content = self.provider.lookup(call["function"]["name"], args)
            except Divergence as d:
                self.diverged, self.divergence, self.stop_reason = True, d.to_dict(), "diverged"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"error": "replay divergence: this call was not recorded"}),
                })
                self.done = True
                return True
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

        if self.turns_taken >= self.max_turns:
            self.done = True
        return self.done

    def outcome(self, gate: dict | None = None) -> TaskOutcome:
        """The finished trajectory as a TaskOutcome. `gate` is a cascade client's per-task summary, if any."""
        messages, gate = self.messages, gate or {}
        final_text = next((m.get("content") or "" for m in reversed(messages) if m["role"] == "assistant"), "")
        schema_ok, _ = tool_calls_valid({"messages": messages, "tools": self.tools})
        return TaskOutcome(
            task_id=self.trace.get("task_id") or self.trace["id"],
            repeat_idx=self.repeat_idx,
            messages=messages,
            final_text=final_text,
            n_turns=sum(1 for m in messages if m["role"] == "assistant"),
            n_tool_calls=self.n_calls,
            schema_valid=schema_ok,
            diverged=self.diverged,
            divergence=self.divergence,
            replay_stats=self.provider.summary(),
            latency_ms=int((time.time() - self.started) * 1000),
            completion_tokens_est=self.tokens,
            stop_reason=self.stop_reason,
            escalations=int(gate.get("escalations", 0)),
            wasted_student_tokens=int(gate.get("wasted_student_tokens", 0)),
        )


def run_task(
    trace: dict,
    client: TurnClient,
    provider: ReplayToolProvider,
    repeat_idx: int = 0,
    max_turns: int = 12,
) -> TaskOutcome:
    """Run one task against the replayed environment."""
    stepper = TaskStepper(trace, provider, repeat_idx=repeat_idx, max_turns=max_turns)
    while not stepper.done:
        stepper.step(client.next_turn(*stepper.request))
    # A cascade client knows what the gate did; a plain client does not have a summary and reports zeros.
    return stepper.outcome(client.summary() if hasattr(client, "summary") else {})


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
