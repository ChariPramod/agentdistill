"""The batched lockstep runner.

The sequential harness sends one request at a time, so the throughput it measures is a floor on what a served
model achieves and any cost per token derived from it is an upper bound, typically several times too high. This
runner keeps up to `batch_size` (task, repeat) items in flight and, at each step, asks the client for the next
turn of every live item in one batched call. Finished items leave the batch and queued items take their place.

Each reply goes through the same `TaskStepper` the sequential `run_task` uses, so the trajectories are the same;
only the scheduling differs. That equivalence is tested, because a batched throughput figure measured on
trajectories that differ from the sequential ones would be a number about a different eval.

Order is load-bearing: replies are zipped against the live items, so a client that returns them shuffled would
hand one task's turn to another. The runner checks the count and shape of every batch; the vLLM client checks the
order itself, since only it can see which prompt each output belongs to.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentdistill.eval.harness import TaskOutcome, TaskStepper
from agentdistill.eval.replay import ReplayToolProvider


@dataclass
class LockstepItem:
    trace: dict
    provider: ReplayToolProvider
    repeat_idx: int = 0


@dataclass
class LockstepStats:
    #: True only when every step went through the client's `next_turns_batch`.
    batched: bool = False
    batch_size: int = 0
    #: Client calls made: one per step when batched, one per live item per step otherwise.
    calls: int = 0
    #: Assistant turns produced across all items.
    item_turns: int = 0
    max_inflight: int = 0
    #: Wall seconds spent inside generation calls, which is the denominator of the batched throughput figure.
    generate_seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)


def supports_batching(client: Any) -> bool:
    return callable(getattr(client, "next_turns_batch", None))


def run_lockstep(
    items: Iterable[LockstepItem],
    client: Any,
    batch_size: int,
    max_turns: int = 12,
    on_done: Callable[[LockstepItem, TaskOutcome], None] | None = None,
) -> tuple[list[TaskOutcome], LockstepStats]:
    """Run every item to completion with at most `batch_size` in flight. Outcomes come back in input order.

    A client without `next_turns_batch` is driven one `next_turn` per live item per step. The trajectories are the
    same, but the stats say `batched=False`, and nothing may call the resulting throughput a batched figure.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    queue = deque(enumerate(items))
    batch_fn = getattr(client, "next_turns_batch", None) if supports_batching(client) else None
    stats = LockstepStats(batched=batch_fn is not None, batch_size=batch_size)
    outcomes: dict[int, TaskOutcome] = {}
    live: list[tuple[int, LockstepItem, TaskStepper]] = []

    def finish(idx: int, item: LockstepItem, stepper: TaskStepper) -> None:
        outcome = stepper.outcome()
        outcomes[idx] = outcome
        if on_done:
            on_done(item, outcome)

    while queue or live:
        while queue and len(live) < batch_size:
            idx, item = queue.popleft()
            stepper = TaskStepper(item.trace, item.provider, repeat_idx=item.repeat_idx, max_turns=max_turns)
            if stepper.done:
                # A zero turn budget: nothing to generate, but the item still gets its outcome.
                finish(idx, item, stepper)
                continue
            live.append((idx, item, stepper))
        if not live:
            continue
        stats.max_inflight = max(stats.max_inflight, len(live))
        requests = [stepper.request for _, _, stepper in live]

        started = time.perf_counter()
        if batch_fn is not None:
            replies = batch_fn(requests)
            stats.calls += 1
        else:
            replies = [client.next_turn(m, t) for m, t in requests]
            stats.calls += len(requests)
        stats.generate_seconds += time.perf_counter() - started
        stats.batch_sizes.append(len(requests))
        check_replies(requests, replies)
        stats.item_turns += len(replies)

        still: list[tuple[int, LockstepItem, TaskStepper]] = []
        for (idx, item, stepper), reply in zip(live, replies, strict=True):
            if stepper.step(reply):
                finish(idx, item, stepper)
            else:
                still.append((idx, item, stepper))
        live = still

    return [outcomes[i] for i in sorted(outcomes)], stats


def check_replies(requests: list, replies: Any) -> None:
    """The count and shape a batched client owes the runner. Order is the client's to prove; see the module
    docstring."""
    if not isinstance(replies, list) or len(replies) != len(requests):
        n = len(replies) if isinstance(replies, list) else type(replies).__name__
        raise AssertionError(f"batched client returned {n} replies for {len(requests)} requests")
    for i, reply in enumerate(replies):
        if not isinstance(reply, dict):
            raise AssertionError(f"batched client reply {i} is {type(reply).__name__}, not an assistant dict")
