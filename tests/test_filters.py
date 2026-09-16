"""Per-trace filters."""

from __future__ import annotations

import json

import pytest

from agentdistill.curate.filters import is_error_result, length, no_error_loops, outcome, quality_judge, teacher
from tests.conftest import make_call, make_trace


def _tool(content) -> dict:
    return {"role": "tool", "tool_call_id": "x", "content": json.dumps(content) if not isinstance(content, str) else content}


@pytest.mark.parametrize(
    "content",
    ['{"error": "not found"}', "Error: timeout", "ERROR unreachable", '{"status": "failed"}',
     '{"status_code": 500}', "Traceback (most recent call last)"],
)
def test_error_results_detected(content):
    assert is_error_result(_tool(content))


@pytest.mark.parametrize(
    "content",
    ['{"ok": true}', "the customer reported an error in the invoice", '{"error": null}', '{"status": "shipped"}',
     '{"status_code": 200}', ""],
)
def test_non_errors_not_detected(content):
    assert not is_error_result(_tool(content))


def test_consecutive_errors_drop_the_trace():
    t = {"messages": [_tool("Error: a"), _tool("Error: b"), _tool("Error: c")]}
    ok, reason = no_error_loops(t)
    assert not ok
    assert "3 consecutive tool errors" in reason


def test_errors_broken_by_a_success_are_tolerated():
    t = {"messages": [_tool("Error: a"), _tool('{"ok": true}'), _tool("Error: b")]}
    assert no_error_loops(t)[0]


def test_repeated_identical_calls_drop_the_trace():
    t = {"messages": [
        {"role": "assistant", "tool_calls": [make_call("1", "s", {"a": 1})]},
        {"role": "assistant", "tool_calls": [make_call("2", "s", {"a": 1})]},
    ]}
    ok, reason = no_error_loops(t)
    assert not ok
    assert "repeated" in reason


def test_repetition_is_detected_across_argument_formatting():
    """`{"a":1}` and `{"a": 1}` are the same call."""
    t = {"messages": [
        {"role": "assistant", "tool_calls": [{"id": "1", "type": "function",
                                              "function": {"name": "s", "arguments": '{"a":1}'}}]},
        {"role": "assistant", "tool_calls": [{"id": "2", "type": "function",
                                              "function": {"name": "s", "arguments": '{"a": 1}'}}]},
    ]}
    assert not no_error_loops(t)[0]


def test_different_arguments_are_not_repetition():
    t = {"messages": [
        {"role": "assistant", "tool_calls": [make_call("1", "s", {"a": 1})]},
        {"role": "assistant", "tool_calls": [make_call("2", "s", {"a": 2})]},
    ]}
    assert no_error_loops(t)[0]


@pytest.mark.parametrize(("turns", "ok"), [(1, False), (2, True), (40, True), (41, False)])
def test_length_bounds(turns, ok):
    assert length({"n_turns": turns}, min_turns=2, max_turns=40)[0] is ok


def test_length_counts_turns_when_not_precomputed():
    t = make_trace("t")
    t.pop("n_turns")
    assert length(t, min_turns=2, max_turns=40)[0]


@pytest.mark.parametrize(
    ("success", "keep", "reason_fragment"),
    [(True, True, ""), (False, False, "task failed"), (None, False, "no recorded outcome")],
)
def test_outcome_filter(success, keep, reason_fragment):
    ok, reason = outcome({"success": success})
    assert ok is keep
    assert reason_fragment in reason


def test_outcome_can_be_disabled_for_dpo():
    assert outcome({"success": False}, require=False)[0]


def test_teacher_filter():
    assert teacher({"teacher_model": "a"}, models=["a", "b"])[0]
    assert not teacher({"teacher_model": "c"}, models=["a", "b"])[0]
    assert teacher({"teacher_model": "c"}, models=[])[0], "no restriction configured means keep"


def test_quality_judge_keeps_unscored_traces():
    """Enabling the filter without running the judge must not empty the dataset."""
    assert quality_judge({"id": "t"}, min_score=3.0, scores={})[0]
    assert quality_judge({"id": "t"}, min_score=3.0, scores={"other": 1.0})[0]
    assert not quality_judge({"id": "t"}, min_score=3.0, scores={"t": 2.0})[0]
    assert quality_judge({"id": "t"}, min_score=3.0, scores={"t": 4.0})[0]
