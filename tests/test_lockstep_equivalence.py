"""The batched lockstep runner against an independent oracle.

The equivalence tests that existed before this file compared the lockstep runner with the sequential harness.
Both drive the same `TaskStepper`, so they compared the stepper with itself and passed by construction. These
tests compare it instead with `tests/oracle/reference_harness.py`, which shares no code with `agentdistill` and
was written from the harness's documented conventions. A difference between them is a real difference.

The seven properties checked here are the ones a batched throughput figure rests on: equivalence, reply order,
what the timing denominator covers, refill, per-item providers, and isolation of a diverging task.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.eval.harness import run_task
from agentdistill.eval.lockstep import REPLY_INDEX, LockstepItem, run_lockstep
from agentdistill.eval.replay import ReplayToolProvider
from tests.lockstep_corpus import (
    KINDS,
    FakeBatchClient,
    Sequential,
    items,
    make_task,
    oracle_token_estimate,
    respond,
)
from tests.oracle.reference_harness import OracleOutcome, recorded_lookup, run_reference

MAX_TURNS = 6


def oracle(trace: dict, kinds: dict | None = None, max_turns: int = MAX_TURNS) -> OracleOutcome:
    """Run one task through the independent oracle with the same student and the same recording."""
    kinds = kinds or {}
    return run_reference(
        trace,
        next_turn=lambda messages, tools: respond(messages, **kinds),
        lookup=recorded_lookup(trace),
        max_turns=max_turns,
        token_estimate=oracle_token_estimate,
    )


def assert_matches_oracle(outcome, expected: OracleOutcome) -> None:
    """Per-task equality on everything the oracle knows about: the transcript and the four derived numbers."""
    assert outcome.task_id == expected.task_id
    assert outcome.messages == expected.messages, f"{expected.task_id}: transcripts differ"
    assert outcome.n_turns == expected.n_turns, f"{expected.task_id}: turn count differs"
    assert outcome.n_tool_calls == expected.n_tool_calls, f"{expected.task_id}: tool-call count differs"
    assert outcome.diverged == expected.diverged, f"{expected.task_id}: divergence flag differs"
    assert outcome.completion_tokens_est == expected.token_estimate, f"{expected.task_id}: token estimate differs"


# --------------------------------------------------------------------------------------------------------------
# 1 and 2. equivalence against the oracle
# --------------------------------------------------------------------------------------------------------------


def test_forty_tasks_match_an_independent_oracle_turn_for_turn():
    traces = [make_task(i) for i in range(40)]
    expected = [oracle(t, KINDS) for t in traces]
    client = FakeBatchClient(**KINDS)
    batched, stats = run_lockstep(items(traces), client, batch_size=40, max_turns=MAX_TURNS)

    assert len(batched) == 40
    for outcome, want in zip(batched, expected, strict=True):
        assert_matches_oracle(outcome, want)

    # The corpus exercises every stop reason, so the equality above is not vacuous.
    assert {o.stop_reason for o in batched} == {"answered", "diverged", "max_turns"}
    assert sum(o.diverged for o in batched) == len(KINDS["diverge"])
    assert sum("not valid JSON" in json.dumps(o.messages) for o in batched) == len(KINDS["malformed"])
    assert sum(o.stop_reason == "max_turns" for o in batched) == len(KINDS["loop"])
    assert stats.batched


def test_the_sequential_harness_matches_the_same_oracle():
    """Both production paths are held to the oracle, so an equivalence failure names which one moved."""
    traces = [make_task(i) for i in range(40)]
    for trace in traces:
        outcome = run_task(trace, Sequential(**KINDS), ReplayToolProvider(trace), max_turns=MAX_TURNS)
        assert_matches_oracle(outcome, oracle(trace, KINDS))


def test_batch_size_never_changes_a_transcript():
    traces = [make_task(i) for i in range(40)]
    expected = [oracle(t, KINDS) for t in traces]
    for size in (1, 3, 7, 40):
        batched, _ = run_lockstep(items(traces), FakeBatchClient(**KINDS), batch_size=size, max_turns=MAX_TURNS)
        for outcome, want in zip(batched, expected, strict=True):
            assert_matches_oracle(outcome, want)


# Two transcripts written out by hand from the documented rules, so a change that moves both runners together
# still fails. Task 0 answers after one recorded lookup; task 5 sends malformed arguments, is told so, and
# recovers.
TASK_0_TRANSCRIPT = [
    {"role": "system", "content": "You look things up."},
    {"role": "user", "content": "task 0: find the value"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "k0_0"}'}}]},
    {"role": "tool", "tool_call_id": "c0", "content": '{"v": "0.0"}'},
    {"role": "assistant", "content": "The answer to task 0.", "tool_calls": None},
]

TASK_5_TRANSCRIPT = [
    {"role": "system", "content": "You look things up."},
    {"role": "user", "content": "task 5: find the value"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c0", "type": "function", "function": {"name": "lookup", "arguments": "{not json"}}]},
    {"role": "tool", "tool_call_id": "c0", "content": '{"error": "arguments were not valid JSON"}'},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "k5_1"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": '{"v": "5.1"}'},
    {"role": "assistant", "content": "The answer to task 5.", "tool_calls": None},
]


@pytest.mark.parametrize(("task", "transcript"), [(0, TASK_0_TRANSCRIPT), (5, TASK_5_TRANSCRIPT)])
def test_every_runner_reproduces_the_hand_written_transcript(task, transcript):
    trace = make_task(task)
    batched, _ = run_lockstep(items([trace]), FakeBatchClient(**KINDS), batch_size=1, max_turns=MAX_TURNS)
    sequential = run_task(trace, Sequential(**KINDS), ReplayToolProvider(trace), max_turns=MAX_TURNS)
    assert batched[0].messages == transcript
    assert sequential.messages == transcript
    assert oracle(trace, KINDS).messages == transcript


# --------------------------------------------------------------------------------------------------------------
# 3. order: a reply names the prompt it answers
# --------------------------------------------------------------------------------------------------------------


def test_a_reply_that_answers_another_prompt_is_refused_by_name():
    traces = [make_task(i) for i in range(4)]
    with pytest.raises(AssertionError, match=r"position 0 answers prompt 1"):
        run_lockstep(items(traces), FakeBatchClient(shuffle=True), batch_size=4, max_turns=MAX_TURNS)


def test_a_reply_without_an_index_is_refused():
    class Unindexed(FakeBatchClient):
        def next_turns_batch(self, requests):
            return [{k: v for k, v in r.items() if k != REPLY_INDEX}
                    for r in super().next_turns_batch(requests)]

    with pytest.raises(AssertionError, match=r"carries no " + REPLY_INDEX):
        run_lockstep(items([make_task(i) for i in range(3)]), Unindexed(), batch_size=3, max_turns=MAX_TURNS)


def test_the_reply_index_never_reaches_a_trajectory():
    """`REPLY_INDEX` is transport, like `_raw`: it is stripped before the turn joins the messages."""
    traces = [make_task(i) for i in range(4)]
    batched, _ = run_lockstep(items(traces), FakeBatchClient(), batch_size=4, max_turns=MAX_TURNS)
    for outcome in batched:
        assert REPLY_INDEX not in json.dumps(outcome.messages)
        assert all(REPLY_INDEX not in m for m in outcome.messages)


def test_shuffled_replies_would_have_changed_the_trajectories():
    """What the order check prevents: without it the runner hands one task another task's turn."""
    traces = [make_task(i) for i in range(4)]
    good, _ = run_lockstep(items(traces), FakeBatchClient(), batch_size=4, max_turns=MAX_TURNS)

    class Silent(FakeBatchClient):
        """Shuffles the replies *and* relabels them, so the order check cannot see it."""

        def next_turns_batch(self, requests):
            replies = super().next_turns_batch(requests)
            return [r | {REPLY_INDEX: i} for i, r in enumerate(replies)]

    bad, _ = run_lockstep(items(traces), Silent(shuffle=True), batch_size=4, max_turns=MAX_TURNS)
    assert [o.messages for o in good] != [o.messages for o in bad]


# --------------------------------------------------------------------------------------------------------------
# 4. timing: generate_seconds covers the generate calls and nothing else
# --------------------------------------------------------------------------------------------------------------


class Clock:
    """A clock that only moves when a test moves it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class SlowProvider:
    """A replay provider whose lookups take time. Wrapping rather than subclassing so the delay is unmistakably
    outside the client."""

    def __init__(self, trace: dict, clock: Clock, cost: float) -> None:
        self.inner, self.clock, self.cost = ReplayToolProvider(trace), clock, cost

    def lookup(self, tool: str, args: dict) -> str:
        self.clock.advance(self.cost)
        return self.inner.lookup(tool, args)

    def summary(self) -> dict:
        return self.inner.summary()


def _run_with_clock(traces: list[dict], lookup_cost: float) -> tuple[float, float]:
    clock = Clock()
    client = FakeBatchClient(clock=clock, generate_cost=0.25, **KINDS)
    lockstep_items = [LockstepItem(t, SlowProvider(t, clock, lookup_cost)) for t in traces]
    _, stats = run_lockstep(lockstep_items, client, batch_size=8, max_turns=MAX_TURNS, clock=clock)
    return stats.generate_seconds, clock.t


def test_generate_seconds_ignores_time_spent_in_replay_lookups():
    traces = [make_task(i) for i in range(12)]
    fast_generate, fast_total = _run_with_clock(traces, lookup_cost=0.0)
    slow_generate, slow_total = _run_with_clock(traces, lookup_cost=10.0)

    assert fast_generate == pytest.approx(slow_generate)
    assert slow_total > fast_total, "the injected clock did advance on lookups, so the check above means something"
    # With free lookups the clock only moves inside generate, so the two must agree exactly.
    assert fast_generate == pytest.approx(fast_total)
    assert slow_generate < slow_total, "the lookup time leaked into the generation denominator"


def test_generate_seconds_counts_one_quarter_second_per_step():
    traces = [make_task(i) for i in range(12)]
    clock = Clock()
    client = FakeBatchClient(clock=clock, generate_cost=0.25, **KINDS)
    lockstep_items = [LockstepItem(t, SlowProvider(t, clock, 3.0)) for t in traces]
    _, stats = run_lockstep(lockstep_items, client, batch_size=8, max_turns=MAX_TURNS, clock=clock)
    assert stats.generate_seconds == pytest.approx(0.25 * len(client.batch_sizes))
    assert stats.generate_seconds == pytest.approx(0.25 * stats.calls)


# --------------------------------------------------------------------------------------------------------------
# 5. refill: a finished item's slot is filled in the same step
# --------------------------------------------------------------------------------------------------------------


def test_a_finished_item_is_replaced_in_the_next_step_not_after_the_batch_drains():
    traces = [make_task(i) for i in range(5)]
    client = FakeBatchClient()
    finished_at: list[int] = []
    outcomes, stats = run_lockstep(
        items(traces, repeats=3), client, batch_size=4, max_turns=MAX_TURNS,
        # `batch_sizes` has one entry per step so far, so its length names the step that just ran.
        on_done=lambda item, outcome: finished_at.append(len(client.batch_sizes) - 1),
    )

    assert len(outcomes) == 15
    assert [(o.task_id, o.repeat_idx) for o in outcomes] == [(t["id"], k) for t in traces for k in range(3)]
    assert stats.max_inflight == 4
    assert max(client.batch_sizes) == 4

    first = min(finished_at)
    assert first + 1 < len(client.batch_sizes), "the run ended at the first completion; nothing to refill"
    assert client.batch_sizes[first + 1] == 4, "the freed slot was not filled in the very next step"
    # The batch only shrinks once the queue is empty, so every step but the tail runs full.
    shrunk = [i for i, n in enumerate(client.batch_sizes) if n < 4]
    assert client.batch_sizes[min(shrunk):] == sorted(client.batch_sizes[min(shrunk):], reverse=True)


# --------------------------------------------------------------------------------------------------------------
# 6. repeats: one provider per item
# --------------------------------------------------------------------------------------------------------------


class CountingProvider(ReplayToolProvider):
    def __init__(self, trace: dict) -> None:
        super().__init__(trace)
        self.lookups = 0

    def lookup(self, tool: str, args: dict) -> str:
        self.lookups += 1
        return super().lookup(tool, args)


def test_two_repeats_of_one_trace_do_not_share_a_provider():
    trace = make_task(3)  # three recorded lookups before the answer
    repeats = [LockstepItem(trace, CountingProvider(trace), repeat_idx=k) for k in range(2)]
    outcomes, _ = run_lockstep(repeats, FakeBatchClient(), batch_size=2, max_turns=MAX_TURNS)

    expected = oracle(trace).n_tool_calls
    assert expected > 0
    assert [item.provider.lookups for item in repeats] == [expected, expected], "lookups were counted per item"
    assert repeats[0].provider is not repeats[1].provider
    for outcome in outcomes:
        assert outcome.replay_stats["replayed"] == expected
        assert_matches_oracle(outcome, oracle(trace))
    assert [o.repeat_idx for o in outcomes] == [0, 1]


def test_repeats_interleaved_with_other_tasks_keep_their_own_counts():
    traces = [make_task(i) for i in range(4)]
    built = [(t, CountingProvider(t), k) for t in traces for k in range(2)]
    lockstep_items = [LockstepItem(t, p, repeat_idx=k) for t, p, k in built]
    run_lockstep(lockstep_items, FakeBatchClient(), batch_size=3, max_turns=MAX_TURNS)
    for trace, provider, _ in built:
        assert provider.lookups == oracle(trace).n_tool_calls, trace["id"]


# --------------------------------------------------------------------------------------------------------------
# 7. isolation: one task's divergence stops that task and nothing else
# --------------------------------------------------------------------------------------------------------------


def test_one_task_diverging_mid_batch_leaves_every_sibling_equal_to_the_oracle():
    traces = [make_task(i) for i in range(12)]
    outcomes, _ = run_lockstep(items(traces), FakeBatchClient(diverge={5}), batch_size=12, max_turns=MAX_TURNS)

    diverged = [o for o in outcomes if o.diverged]
    assert [o.task_id for o in diverged] == ["t05"]
    assert diverged[0].stop_reason == "diverged"
    for outcome, trace in zip(outcomes, traces, strict=True):
        # Every other task is compared against an oracle run that knows nothing about the divergence.
        assert_matches_oracle(outcome, oracle(trace, {"diverge": {5}}))
        if outcome.task_id != "t05":
            assert_matches_oracle(outcome, oracle(trace))


def test_a_divergence_does_not_change_a_siblings_replay_stats():
    traces = [make_task(i) for i in range(12)]
    clean, _ = run_lockstep(items(traces), FakeBatchClient(), batch_size=12, max_turns=MAX_TURNS)
    one, _ = run_lockstep(items(traces), FakeBatchClient(diverge={5}), batch_size=12, max_turns=MAX_TURNS)
    for a, b in zip(clean, one, strict=True):
        if a.task_id == "t05":
            assert b.diverged and not a.diverged
        else:
            assert (a.messages, a.stop_reason, a.diverged, a.replay_stats) == \
                   (b.messages, b.stop_reason, b.diverged, b.replay_stats)
