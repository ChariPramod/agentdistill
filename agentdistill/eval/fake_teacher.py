"""A deterministic stand-in for the teacher, for tiny mode only.

The goal is not a good teacher; it is a teacher-shaped row, so that the report's cost block, the calibration
fit, and the cascade verification execute on a laptop instead of being skipped. Every number it produces is
structural: on the holdout set it is a perfect teacher by construction, because it replays what the recording
did. A run made with it carries `metrics.teacher_backend = "replay"`, and the report refuses to show its cost
without saying so.

Stateless on purpose. It finds the task from the conversation (the same task text curation clusters on) and
replays the recorded assistant turn at the conversation's current turn index. That lets it answer a turn in the
middle of a cascade, where the prefix was built by the student and nobody told it which task it is on.
"""

from __future__ import annotations

import json
from typing import Any

#: Fixed per-turn token counts. Realistic in shape for an agent turn -- the prompt dominates, because the system
#: prompt and tool schemas repeat every turn -- and enough to exercise every cost formula. Not a measurement.
PROMPT_TOKENS = 1200
COMPLETION_TOKENS = 90


class UnknownTask(LookupError):
    """The conversation matches no recorded trace, so there is nothing to replay."""


class ReplayTeacherClient:
    """A TurnClient that replays each recorded trace's own assistant turns."""

    backend_name = "replay"

    def __init__(self, traces: list[dict], prompt_tokens: int = PROMPT_TOKENS,
                 completion_tokens: int = COMPLETION_TOKENS) -> None:
        self.by_text: dict[str, dict] = {}
        for t in traces:
            self.by_text.setdefault(self._key(t["messages"]), t)
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def _key(messages: list[dict]) -> str:
        from agentdistill.curate.decontaminate import task_text
        from agentdistill.ingest.normalize import extract_task_input

        return task_text(extract_task_input(messages))

    @classmethod
    def from_registry(cls, registry: Any) -> ReplayTeacherClient:
        # Teacher-recorded traces only: a rollout is the student's own output and must never be replayed as the
        # teacher's answer.
        return cls([t for t in registry.list_traces() if t.get("source") != "rollout"])

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        trace = self.by_text.get(self._key(messages))
        if trace is None:
            raise UnknownTask("the replay teacher has no recorded trace for this task")
        turns = [m for m in trace["messages"] if m["role"] == "assistant"]
        i = sum(1 for m in messages if m["role"] == "assistant")
        self.usage["prompt_tokens"] += self.prompt_tokens
        self.usage["completion_tokens"] += self.completion_tokens
        if i >= len(turns):
            return {"role": "assistant", "content": "Done.", "tool_calls": None}
        turn = json.loads(json.dumps(turns[i]))
        return {"role": "assistant", "content": turn.get("content"), "tool_calls": turn.get("tool_calls") or None}
