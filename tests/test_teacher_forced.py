"""Teacher-forced next-action accuracy."""

from __future__ import annotations

import json
import math

import pytest

from agentdistill.eval.clients import RecordedTurnClient, ScriptedTurnClient, parse_assistant
from agentdistill.eval.teacher_forced import (
    TurnResult,
    compare_turns,
    format_summary,
    summarize,
    teacher_forced,
    teacher_forced_batched,
    turn_signature,
)


def call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def trace(task_id: str = "task1", tool: str = "search_orders", args: dict | None = None) -> dict:
    return {
        "id": task_id,
        "task_id": task_id,
        "tools": [],
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "where is my order"},
            {"role": "assistant", "content": "Looking.", "tool_calls": [call(tool, args or {"customer_id": "c_9"})]},
            {"role": "tool", "tool_call_id": "c1", "content": "[]"},
            {"role": "assistant", "content": "It shipped.", "tool_calls": None},
        ],
    }


# --------------------------------------------------------------------------------------------------------------
# the two anchor cases from the plan
# --------------------------------------------------------------------------------------------------------------


def test_echoing_the_teacher_scores_perfectly():
    results = teacher_forced([trace()], RecordedTurnClient(trace()))
    s = summarize(results)
    assert s["full_match"][0] == 1.0
    assert s["name_match_on_tool_turns"] == 1.0
    assert s["args_match_on_tool_turns"] == 1.0
    assert s["n_turns"] == 2 and s["n_tool_turns"] == 1


def test_swapping_a_tool_name_scores_zero_on_name_match():
    t = trace()
    wrong = ScriptedTurnClient([
        {"role": "assistant", "content": "Looking.", "tool_calls": [call("cancel_order", {"customer_id": "c_9"})]},
        {"role": "assistant", "content": "It shipped.", "tool_calls": None},
    ])
    s = summarize(teacher_forced([t], wrong))
    assert s["name_match_on_tool_turns"] == 0.0
    assert s["full_match"][0] == 0.5, "the text turn still matches"


# --------------------------------------------------------------------------------------------------------------
# what counts as a match
# --------------------------------------------------------------------------------------------------------------


def test_argument_formatting_is_not_a_difference():
    """Canonical hashing means key order and whitespace do not count against the student."""
    teacher = {"role": "assistant", "tool_calls": [call("f", {"a": 1, "b": 2})]}
    student = {
        "role": "assistant",
        "tool_calls": [{"id": "z", "type": "function", "function": {"name": "f", "arguments": '{"b": 2,  "a": 1}'}}],
    }
    r = compare_turns(teacher, student, "t", 0)
    assert r.full_match


def test_different_argument_values_are_a_difference():
    teacher = {"role": "assistant", "tool_calls": [call("f", {"a": 1})]}
    student = {"role": "assistant", "tool_calls": [call("f", {"a": 2})]}
    r = compare_turns(teacher, student, "t", 0)
    assert r.name_match and not r.args_match and not r.full_match


def test_parallel_call_order_is_not_a_difference():
    teacher = {"role": "assistant", "tool_calls": [call("a", {}, "1"), call("b", {}, "2")]}
    student = {"role": "assistant", "tool_calls": [call("b", {}, "9"), call("a", {}, "8")]}
    assert compare_turns(teacher, student, "t", 0).full_match


def test_answering_instead_of_calling_a_tool_is_a_kind_mismatch():
    teacher = {"role": "assistant", "tool_calls": [call("f", {})]}
    student = {"role": "assistant", "content": "I think it shipped.", "tool_calls": None}
    r = compare_turns(teacher, student, "t", 0)
    assert not r.kind_match and not r.full_match
    assert r.teacher_kind == "tool" and r.student_kind == "text"


def test_unparseable_arguments_never_match_a_valid_call():
    teacher = {"role": "assistant", "tool_calls": [call("f", {"a": 1})]}
    student = {
        "role": "assistant",
        "tool_calls": [{"id": "z", "type": "function", "function": {"name": "f", "arguments": "{not json"}}],
    }
    r = compare_turns(teacher, student, "t", 0)
    assert r.name_match and not r.args_match


def test_turn_signature_shape():
    kind, names, hashes = turn_signature({"role": "assistant", "tool_calls": [call("f", {"a": 1})]})
    assert kind == "tool" and names == frozenset({"f"}) and len(hashes) == 1
    kind, names, hashes = turn_signature({"role": "assistant", "content": "hi"})
    assert kind == "text" and not names and not hashes


# --------------------------------------------------------------------------------------------------------------
# conditioning and aggregation
# --------------------------------------------------------------------------------------------------------------


def test_name_match_is_conditional_on_the_teacher_calling_a_tool():
    """Averaging over text turns would inflate 'did it pick the right tool'."""
    results = [
        TurnResult("t", 0, "text", "text", name_match=True, args_match=True, kind_match=True),
        TurnResult("t", 1, "tool", "tool", name_match=False, args_match=False, kind_match=True),
    ]
    s = summarize(results)
    assert s["name_match_on_tool_turns"] == 0.0, "only the tool turn counts"
    assert s["kind_match"][0] == 1.0


def test_args_match_is_conditional_on_the_name_matching():
    """Separates 'knows what to do' from 'knows how to fill it in'."""
    results = [
        TurnResult("t", 0, "tool", "tool", name_match=False, args_match=False, kind_match=True),
        TurnResult("t", 1, "tool", "tool", name_match=True, args_match=True, kind_match=True),
    ]
    s = summarize(results)
    assert s["args_match_on_tool_turns"] == 1.0, "the wrong-tool turn is not an argument-filling failure"


def test_no_tool_turns_reports_nan_not_zero():
    """Zero would read as 'got every tool call wrong' when no tool call was scored."""
    results = [TurnResult("t", 0, "text", "text", True, True, True)]
    s = summarize(results)
    assert math.isnan(s["name_match_on_tool_turns"])
    assert math.isnan(s["args_match_on_tool_turns"])


def test_bootstrap_resamples_tasks_not_turns():
    """Turns within a task are correlated; a turn-level bootstrap would report a far too narrow interval."""
    results = []
    for task in range(20):
        for turn in range(5):
            results.append(TurnResult(f"t{task}", turn, "tool", "tool", True, True, task % 2 == 0))
    s = summarize(results, iters=500, seed=0)
    _, (lo, hi) = s["full_match"]
    assert s["n_tasks"] == 20
    assert hi - lo > 0.15, "interval should be wide: half the tasks fail entirely"


def test_summarize_on_a_single_task_reports_no_interval():
    s = summarize([TurnResult("only", 0, "tool", "tool", True, True, True)])
    _, (lo, hi) = s["full_match"]
    assert math.isnan(lo) and math.isnan(hi), "one task gives no basis for a task-level interval"


def test_summarize_empty():
    s = summarize([])
    assert s["n_turns"] == 0
    assert "no turns scored" in format_summary(s)


def test_format_summary_is_one_line():
    s = summarize(teacher_forced([trace()], RecordedTurnClient(trace())))
    line = format_summary(s)
    assert "\n" not in line and "next_action" in line


# --------------------------------------------------------------------------------------------------------------
# traversal and batching
# --------------------------------------------------------------------------------------------------------------


def test_every_assistant_turn_is_scored():
    results = teacher_forced([trace("a"), trace("b")], RecordedTurnClient(trace()))
    assert len(results) == 4
    assert {r.task_id for r in results} == {"a", "b"}


def test_max_turns_per_trace_caps_work():
    results = teacher_forced([trace()], RecordedTurnClient(trace()), max_turns_per_trace=1)
    assert len(results) == 1


def test_client_receives_only_the_prefix():
    """The student must never see the turn it is being asked to predict."""
    seen = []

    class Spy:
        def next_turn(self, messages, tools):
            seen.append(list(messages))
            return {"role": "assistant", "content": "x"}

    teacher_forced([trace()], Spy())
    for prefix in seen:
        assert prefix[-1]["role"] != "assistant" or len(prefix) < 3
    assert len(seen[0]) == 2, "first prefix is system + user"


def test_batched_path_matches_the_sequential_path():
    class Batched:
        def __init__(self, t):
            self.turns = [m for m in t["messages"] if m["role"] == "assistant"]

        def next_turn(self, messages, tools):
            raise AssertionError("the batched path must be used when available")

        def next_turns_batch(self, prompts):
            return [self.turns[i % len(self.turns)] for i in range(len(prompts))]

    t = trace()
    batched = summarize(teacher_forced_batched([t], Batched(t)))
    sequential = summarize(teacher_forced([t], RecordedTurnClient(t)))
    assert batched["full_match"][0] == sequential["full_match"][0]


def test_batched_falls_back_when_the_client_cannot_batch():
    results = teacher_forced_batched([trace()], RecordedTurnClient(trace()))
    assert len(results) == 2


def test_batched_client_returning_the_wrong_count_is_an_error():
    class Broken:
        def next_turns_batch(self, prompts):
            return [{"role": "assistant", "content": "x"}]

    with pytest.raises(ValueError, match="preserve order"):
        teacher_forced_batched([trace()], Broken())


def test_batched_with_no_turns():
    assert teacher_forced_batched([], object()) == []


# --------------------------------------------------------------------------------------------------------------
# parsing generated text
# --------------------------------------------------------------------------------------------------------------


def test_parse_assistant_recovers_a_tool_call():
    out = parse_assistant('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>', None, None, "hermes")
    assert out["tool_calls"][0]["function"]["name"] == "f"
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert out["content"] is None


def test_unrecognized_output_becomes_a_text_answer():
    """That is what the serving stack would do; scoring it otherwise would flatter the student."""
    out = parse_assistant("Your order shipped yesterday.", None, None, "hermes")
    assert out["tool_calls"] is None
    assert out["content"] == "Your order shipped yesterday."


def test_recorded_client_runs_out_gracefully():
    client = RecordedTurnClient(trace())
    for _ in range(5):
        turn = client.next_turn([], [])
    assert turn == {"role": "assistant", "content": "", "tool_calls": None}
    client.reset()
    assert client.next_turn([], [])["content"] == "Looking."
