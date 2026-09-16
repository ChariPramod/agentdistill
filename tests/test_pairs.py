"""DPO pair construction.

The invariant: a pair is only built when both sides answer the same question, meaning every message before the
divergent assistant turn is identical.
"""

from __future__ import annotations

import json

from agentdistill.data.pairs import (
    first_divergent_pair,
    pairs_against_teacher,
    pairs_from_traces,
    turn_key,
    write_pairs_jsonl,
)
from tests.conftest import make_call


def _trace(tid: str, *, success: bool, second_tool: str = "refund_order", result: str = "found",
           task_id: str = "task-1") -> dict:
    return {
        "id": tid,
        "task_id": task_id,
        "success": success,
        "tools": [],
        "messages": [
            {"role": "user", "content": "refund my order"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [make_call("c1", "lookup", {"id": "o1"})]},
            {"role": "tool", "tool_call_id": "c1", "content": result},
            {"role": "assistant", "content": "Acting.", "tool_calls": [make_call("c2", second_tool, {"id": "o1"})]},
        ],
    }


def test_pair_is_built_at_the_divergent_turn():
    pair = first_divergent_pair(_trace("g", success=True), _trace("b", success=False, second_tool="cancel_order"))
    assert pair is not None
    assert len(pair["prompt"]) == 3, "the shared prefix stops at the divergent assistant turn"
    assert pair["chosen"][0]["tool_calls"][0]["function"]["name"] == "refund_order"
    assert pair["rejected"][0]["tool_calls"][0]["function"]["name"] == "cancel_order"
    assert pair["chosen_trace_id"] == "g" and pair["rejected_trace_id"] == "b"


def test_no_pair_when_the_environment_diverged_first():
    """Different tool results mean different situations, not different decisions."""
    good = _trace("g", success=True, result="found")
    bad = _trace("b", success=False, second_tool="cancel_order", result="not found")
    assert first_divergent_pair(good, bad) is None


def test_no_pair_when_trajectories_are_identical():
    assert first_divergent_pair(_trace("g", success=True), _trace("b", success=False)) is None


def test_no_pair_when_roles_diverge():
    good = _trace("g", success=True)
    bad = _trace("b", success=False)
    bad["messages"][2] = {"role": "user", "content": "actually never mind"}
    assert first_divergent_pair(good, bad) is None


def test_divergence_on_the_first_turn_gives_an_empty_prompt_prefix():
    good = _trace("g", success=True)
    bad = _trace("b", success=False)
    bad["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"id": "WRONG"})
    pair = first_divergent_pair(good, bad)
    assert pair is not None
    assert len(pair["prompt"]) == 1


def test_turn_key_ignores_call_ids_and_argument_formatting():
    a = {"role": "assistant", "content": "x", "tool_calls": [
        {"id": "aaa", "type": "function", "function": {"name": "f", "arguments": '{"a":1,"b":2}'}}]}
    b = {"role": "assistant", "content": "x", "tool_calls": [
        {"id": "zzz", "type": "function", "function": {"name": "f", "arguments": '{"b": 2, "a": 1}'}}]}
    assert turn_key(a) == turn_key(b)


def test_turn_key_distinguishes_content():
    a = {"role": "assistant", "content": "x"}
    b = {"role": "assistant", "content": "y"}
    assert turn_key(a) != turn_key(b)


def test_pairs_from_traces_requires_both_sides():
    only_success = [_trace("g1", success=True), _trace("g2", success=True)]
    pairs, stats = pairs_from_traces(only_success)
    assert pairs == []
    assert stats["tasks_with_both"] == 0


def test_pairs_from_traces_builds_and_reports():
    traces = [_trace("g", success=True), _trace("b", success=False, second_tool="cancel_order")]
    pairs, stats = pairs_from_traces(traces)
    assert len(pairs) == 1
    assert stats["tasks_with_both"] == 1 and stats["pairs_built"] == 1


def test_pairs_are_capped_per_task():
    traces = [_trace(f"g{i}", success=True) for i in range(5)]
    traces += [_trace(f"b{i}", success=False, second_tool=f"tool_{i}") for i in range(5)]
    pairs, stats = pairs_from_traces(traces, max_pairs_per_task=2)
    assert len(pairs) == 2, "one heavily repeated task must not dominate the pair set"
    assert stats["capped"] >= 1


def test_ungraded_traces_are_ignored():
    traces = [_trace("g", success=True), _trace("u", success=None, second_tool="cancel_order")]  # type: ignore[arg-type]
    pairs, _stats = pairs_from_traces(traces)
    assert pairs == []


def test_pairs_against_teacher_uses_student_failures():
    teacher = [_trace("teacher", success=True)]
    rollouts = [
        _trace("student_bad", success=False, second_tool="cancel_order"),
        _trace("student_good", success=True),
    ]
    pairs = pairs_against_teacher(rollouts, teacher)
    assert len(pairs) == 1, "only failed rollouts become rejected sides"
    assert pairs[0]["rejected_trace_id"] == "student_bad"
    assert pairs[0]["source"] == "on_policy_vs_teacher"


def test_write_pairs_jsonl_roundtrips(tmp_path):
    pairs, _ = pairs_from_traces([_trace("g", success=True), _trace("b", success=False, second_tool="cancel_order")])
    p = write_pairs_jsonl(pairs, tmp_path / "pairs.jsonl")
    lines = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(lines) == 1
    assert {"prompt", "chosen", "rejected"} <= set(lines[0])
