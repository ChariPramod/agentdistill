"""The batched lockstep runner.

The sequential harness sends one request at a time, so the throughput it measures is a floor on what a served
model achieves and any cost per token derived from it is an upper bound, typically several times too high. This
runner keeps up to `batch_size` (task, repeat) items in flight and, at each step, asks the client for the next
turn of every live item in one batched call. Finished items leave the batch and queued items take their place.

Each reply goes through the same `TaskStepper` the sequential `run_task` uses, so the trajectories are the same;
only the scheduling differs. Because both paths share that stepper, comparing them to each other proves nothing:
the equivalence tests hold this runner to an independent oracle instead (`tests/oracle/reference_harness.py`),
since a batched throughput figure measured on trajectories that differ from the sequential ones would be a
number about a different eval.

Order is load-bearing: replies are zipped against the live items, so a client that returned them shuffled would
hand one task's turn to another, and every trajectory after that point would be fiction. So a batched client owes
the runner an index on every reply naming the prompt it answers, and the runner checks it. The client is the only
layer that can see which prompt an output came from -- the vLLM client fills the index in from request order,
having first checked each output against its own prompt -- but the check belongs here, because this is where the
zip happens.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentdistill.eval.harness import TaskOutcome, TaskStepper

#: The key a batched reply carries to name the prompt it answers. Underscore-prefixed, so `TaskStepper.step`
#: strips it with the rest of the transport keys and it never reaches a trajectory.
REPLY_INDEX = "_index"


@dataclass
class LockstepItem:
    trace: dict
    #: Anything with `lookup(tool, args) -> str` that raises `Divergence`, and `summary() -> dict`:
    #: `ReplayToolProvider` today, a live-tool provider once one exists.
    provider: Any
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
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[TaskOutcome], LockstepStats]:
    """Run every item to completion with at most `batch_size` in flight. Outcomes come back in input order.

    A client without `next_turns_batch` is driven one `next_turn` per live item per step. The trajectories are the
    same, but the stats say `batched=False`, and nothing may call the resulting throughput a batched figure.

    `clock` is injectable so a test can make replay lookups expensive and prove they stay out of
    `generate_seconds`; nothing in production passes it.
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

        # The window is the generation call and nothing else: replay lookups, grading and bookkeeping happen
        # outside it, so the denominator of the throughput figure is serving time rather than harness time.
        started = clock()
        if batch_fn is not None:
            replies = batch_fn(requests)
            stats.calls += 1
        else:
            replies = [client.next_turn(m, t) for m, t in requests]
            stats.calls += len(requests)
        stats.generate_seconds += clock() - started
        stats.batch_sizes.append(len(requests))
        check_replies(requests, replies, indexed=batch_fn is not None)
        stats.item_turns += len(replies)

        still: list[tuple[int, LockstepItem, TaskStepper]] = []
        for (idx, item, stepper), reply in zip(live, replies, strict=True):
            if stepper.step(reply):
                finish(idx, item, stepper)
            else:
                still.append((idx, item, stepper))
        live = still

    return [outcomes[i] for i in sorted(outcomes)], stats


def check_replies(requests: list, replies: Any, indexed: bool = True) -> None:
    """The count, shape and order a batched client owes the runner.

    Reply `i` must answer prompt `i` and must say so, because the caller zips the two lists together. A client
    that cannot name the prompt it answered cannot be zipped safely, so an unindexed batch is refused rather than
    trusted. `indexed=False` is the per-item fallback path, where the runner made the calls itself in order.
    """
    if not isinstance(replies, list) or len(replies) != len(requests):
        n = len(replies) if isinstance(replies, list) else type(replies).__name__
        raise AssertionError(f"batched client returned {n} replies for {len(requests)} requests")
    for i, reply in enumerate(replies):
        if not isinstance(reply, dict):
            raise AssertionError(f"batched client reply {i} is {type(reply).__name__}, not an assistant dict")
        if not indexed:
            continue
        answers = reply.get(REPLY_INDEX)
        if answers is None:
            raise AssertionError(
                f"batched client reply at position {i} carries no {REPLY_INDEX}: it cannot say which of the "
                f"{len(requests)} prompts it answers, and the runner will not guess"
            )
        if answers != i:
            raise AssertionError(
                f"batched client reply at position {i} answers prompt {answers}, not prompt {i}; "
                f"batch order was not preserved"
            )
