"""The synthetic corpus and the deterministic student both lockstep test files run.

One definition, because the equivalence tests are only worth something if the sequential harness, the batched
lockstep runner and the independent oracle are driven by *the same* student on *the same* tasks. The student is
stateless -- its next turn is a function of the prefix alone -- so interleaving tasks in a batch cannot change
what it says, and any difference the tests find is the runner's, not the client's.
"""

from __future__ import annotations

import json

from agentdistill.eval.lockstep import REPLY_INDEX, LockstepItem
from agentdistill.eval.replay import ReplayToolProvider
from agentdistill.ingest.normalize import normalize_trace
from tests.conftest import make_call

TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}

#: Which synthetic tasks misbehave, and how. Chosen so a 40-task corpus exercises every stop reason.
KINDS = {"diverge": {3, 10, 17, 24, 31, 38}, "malformed": {5, 16, 27}, "loop": {8, 33}}


def make_task(i: int) -> dict:
    """Task i makes 1 + i % 4 sequential lookups and then answers, so turn counts vary across the batch."""
    n_calls = 1 + i % 4
    messages = [{"role": "system", "content": "You look things up."},
                {"role": "user", "content": f"task {i}: find the value"}]
    for j in range(n_calls):
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [make_call(f"c{j}", "lookup", {"key": f"k{i}_{j}"})]})
        messages.append({"role": "tool", "tool_call_id": f"c{j}", "content": json.dumps({"v": f"{i}.{j}"})})
    messages.append({"role": "assistant", "content": f"The answer to task {i}.", "tool_calls": None})
    trace = normalize_trace({"id": f"t{i:02d}", "task_id": f"t{i:02d}", "success": True, "tools": [TOOL],
                             "messages": messages}, source="jsonl")
    trace["id"] = f"t{i:02d}"
    trace["cluster"] = i % 3
    return trace


def task_index(messages: list[dict]) -> int:
    user = next(m["content"] for m in messages if m["role"] == "user")
    return int(user.split()[1].rstrip(":"))


def respond(messages: list[dict], diverge: set[int] = frozenset(), malformed: set[int] = frozenset(),
            loop: set[int] = frozenset()) -> dict:
    """A stateless student: the next turn is a function of the prefix alone, so it gives the same answer under
    any runner. Some tasks diverge, some send malformed arguments, some never stop calling tools."""
    i = task_index(messages)
    turn = sum(1 for m in messages if m["role"] == "assistant")
    n_calls = 1 + i % 4
    if i in loop:
        return {"role": "assistant", "content": None,
                "tool_calls": [make_call(f"c{turn}", "lookup", {"key": f"k{i}_0"})]}
    if turn < n_calls:
        if i in diverge and turn == 1 % n_calls:
            return {"role": "assistant", "content": None,
                    "tool_calls": [make_call(f"c{turn}", "lookup", {"key": "somewhere-unrecorded"})]}
        if i in malformed and turn == 0:
            call = make_call("c0", "lookup", {})
            call["function"]["arguments"] = "{not json"
            return {"role": "assistant", "content": None, "tool_calls": [call]}
        return {"role": "assistant", "content": None,
                "tool_calls": [make_call(f"c{turn}", "lookup", {"key": f"k{i}_{turn}"})],
                # Transport-only keys must be stripped identically on every path.
                "_raw": {"turn": turn}}
    return {"role": "assistant", "content": f"The answer to task {i}.", "tool_calls": None}


def oracle_token_estimate(text: str) -> int:
    """The harness's token rule, restated rather than imported: the oracle must not share code with the thing it
    checks. Kept beside the student so the two copies are easy to diff by eye."""
    return max(1, len(text) // 4)


class Sequential:
    """The same student, with no batched method: what the sequential harness sees."""

    def __init__(self, **kinds) -> None:
        self.kinds = kinds

    def next_turn(self, messages, tools):
        return respond(messages, **self.kinds)


class FakeBatchClient(Sequential):
    """Batches by answering each request in turn, and records the size of every batch.

    Each reply carries the index of the prompt it answers, which is the contract a batched client owes the
    runner. `shuffle` returns correctly-indexed replies in the wrong order, as an engine that lost order would.
    """

    def __init__(self, shuffle: bool = False, clock=None, generate_cost: float = 0.0, **kinds) -> None:
        super().__init__(**kinds)
        self.batch_sizes: list[int] = []
        self.shuffle = shuffle
        self.clock, self.generate_cost = clock, generate_cost

    def next_turns_batch(self, requests):
        self.batch_sizes.append(len(requests))
        if self.clock is not None:
            self.clock.advance(self.generate_cost)
        replies = [respond(m, **self.kinds) | {REPLY_INDEX: i} for i, (m, _) in enumerate(requests)]
        if self.shuffle and len(replies) > 1:
            replies = replies[1:] + replies[:1]
        return replies


def items(traces: list[dict], repeats: int = 1) -> list[LockstepItem]:
    """One item per (task, repeat), each with its own provider, exactly as `run_eval` builds them."""
    return [LockstepItem(t, ReplayToolProvider(t), repeat_idx=k) for t in traces for k in range(repeats)]
