"""The replay harness.

The anchor test is `test_harness_reproduces_trace`: replaying a recording through the harness must reproduce it
exactly, with zero divergences. If that fails, every number the harness produces is meaningless.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.canonical import args_hash
from agentdistill.eval.clients import RecordedTurnClient, ScriptedTurnClient
from agentdistill.eval.harness import initial_messages, run_task, tool_calls_made
from agentdistill.eval.replay import Divergence, ReplayToolProvider, similarity
from tests.conftest import make_call


def trace(task_id: str = "t1") -> dict:
    return {
        "id": task_id,
        "task_id": task_id,
        "success": True,
        "tools": [
            {"type": "function", "function": {"name": "get_customer",
                                              "parameters": {"type": "object",
                                                             "properties": {"email": {"type": "string"}},
                                                             "required": ["email"]}}},
            {"type": "function", "function": {"name": "list_orders",
                                              "parameters": {"type": "object",
                                                             "properties": {"customer_id": {"type": "string"},
                                                                            "status": {"type": "string"}},
                                                             "required": ["customer_id"]}}},
        ],
        "messages": [
            {"role": "system", "content": "You are support."},
            {"role": "user", "content": "where is my order, a@b.com"},
            {"role": "assistant", "content": "Looking you up.",
             "tool_calls": [make_call("c1", "get_customer", {"email": "a@b.com"})]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"customer_id": "c_1"}'},
            {"role": "assistant", "content": "Checking orders.",
             "tool_calls": [make_call("c2", "list_orders", {"customer_id": "c_1"})]},
            {"role": "tool", "tool_call_id": "c2", "content": '{"orders": [{"order_id": "o_1"}]}'},
            {"role": "assistant", "content": "Order o_1 is on its way.", "tool_calls": None},
        ],
    }


def always_success(trace_: dict, outcome) -> tuple[bool, dict]:
    return True, {"detail": "stub"}


# --------------------------------------------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------------------------------------------


def test_harness_reproduces_trace():
    t = trace()
    outcome = run_task(t, RecordedTurnClient(t), ReplayToolProvider(t))
    recorded = [m for m in t["messages"] if m["role"] == "assistant"]
    produced = [m for m in outcome.messages if m["role"] == "assistant"]
    assert [m.get("content") for m in produced] == [m.get("content") for m in recorded]
    assert not outcome.diverged
    assert outcome.stop_reason == "answered"
    assert outcome.n_turns == 3
    assert outcome.n_tool_calls == 2
    assert outcome.schema_valid


def test_replayed_tool_results_match_the_recording():
    t = trace()
    outcome = run_task(t, RecordedTurnClient(t), ReplayToolProvider(t))
    recorded = [m["content"] for m in t["messages"] if m["role"] == "tool"]
    produced = [m["content"] for m in outcome.messages if m["role"] == "tool"]
    assert produced == recorded


def test_run_starts_from_the_task_not_the_solution():
    """The student must never see the assistant turns it is being asked to reproduce."""
    t = trace()
    start = initial_messages(t)
    assert [m["role"] for m in start] == ["system", "user"]
    assert all(m["role"] != "assistant" for m in start)


def test_initial_messages_stops_at_the_first_assistant_turn():
    t = trace()
    t["messages"].insert(2, {"role": "user", "content": "extra context before any reply"})
    assert len(initial_messages(t)) == 3


# --------------------------------------------------------------------------------------------------------------
# divergence
# --------------------------------------------------------------------------------------------------------------


def test_unrecorded_call_diverges_in_strict_mode():
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [make_call("x", "get_customer", {"email": "someone-else@z.com"})]},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t, policy="strict"))
    assert outcome.diverged
    assert outcome.stop_reason == "diverged"
    assert outcome.divergence["tool"] == "get_customer"
    assert outcome.divergence["nearest_args"] == {"email": "a@b.com"}


def test_divergence_stops_the_trajectory():
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [make_call("x", "get_customer", {"email": "nope@z.com"})]},
        {"role": "assistant", "content": "this turn should never be reached"},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t))
    assert outcome.n_turns == 1
    assert "never be reached" not in outcome.final_text


def test_argument_formatting_does_not_diverge():
    """Canonical hashing means key order and whitespace are not behavioral differences."""
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [
            {"id": "z", "type": "function",
             "function": {"name": "get_customer", "arguments": '{"email":  " a@b.com "}'}}]},
        {"role": "assistant", "content": "done"},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t))
    assert not outcome.diverged


def test_fuzzy_policy_serves_a_drifted_call():
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [make_call("x", "list_orders",
                                                       {"customer_id": "c_1", "status": "shipped"})]},
        {"role": "assistant", "content": "done"},
    ])
    strict = run_task(t, client, ReplayToolProvider(t, policy="strict"))
    assert strict.diverged

    client2 = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [make_call("x", "list_orders",
                                                       {"customer_id": "c_1", "status": "shipped"})]},
        {"role": "assistant", "content": "done"},
    ])
    fuzzy = run_task(t, client2, ReplayToolProvider(t, policy="fuzzy", fuzzy_threshold=0.4))
    assert not fuzzy.diverged
    assert fuzzy.replay_stats["fuzzy"] == 1
    assert fuzzy.replay_stats["fuzzy_share"] == 1.0


def test_fuzzy_still_refuses_a_genuinely_different_call():
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [make_call("x", "list_orders", {"customer_id": "SOMEONE_ELSE"})]},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t, policy="fuzzy", fuzzy_threshold=0.9))
    assert outcome.diverged, "a different customer is a different call, not phrasing drift"


def test_calling_an_unrecorded_tool_diverges_with_no_nearest():
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [make_call("x", "issue_refund", {"order_id": "o_1"})]},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t))
    assert outcome.diverged
    assert outcome.divergence["nearest_args"] is None
    assert outcome.divergence["nearest_score"] == 0.0


# --------------------------------------------------------------------------------------------------------------
# student mistakes that are not divergences
# --------------------------------------------------------------------------------------------------------------


def test_malformed_arguments_are_a_tool_error_not_a_divergence():
    """A real tool would reject it the same way; the environment can still answer."""
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [
            {"id": "z", "type": "function", "function": {"name": "get_customer", "arguments": "{not json"}}]},
        {"role": "assistant", "content": "sorry"},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t))
    assert not outcome.diverged
    tool_msg = next(m for m in outcome.messages if m["role"] == "tool")
    assert json.loads(tool_msg["content"])["error"] == "arguments were not valid JSON"


def test_answering_immediately_is_allowed():
    t = trace()
    client = ScriptedTurnClient([{"role": "assistant", "content": "I think it shipped.", "tool_calls": None}])
    outcome = run_task(t, client, ReplayToolProvider(t))
    assert outcome.stop_reason == "answered"
    assert outcome.n_tool_calls == 0
    assert not outcome.diverged


def test_max_turns_is_enforced():
    t = trace()
    looping = [{"role": "assistant", "tool_calls": [make_call("c", "get_customer", {"email": "a@b.com"})]}] * 20
    outcome = run_task(t, ScriptedTurnClient(looping), ReplayToolProvider(t), max_turns=4)
    assert outcome.stop_reason == "max_turns"
    assert outcome.n_turns == 4


def test_schema_invalid_call_is_flagged():
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [make_call("x", "get_customer", {})]},  # missing required email
        {"role": "assistant", "content": "done"},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t))
    assert not outcome.schema_valid


# --------------------------------------------------------------------------------------------------------------
# the provider
# --------------------------------------------------------------------------------------------------------------


def test_provider_indexes_every_recorded_call():
    p = ReplayToolProvider(trace())
    assert p.n_recorded == 2
    assert p.lookup("get_customer", {"email": "a@b.com"}) == '{"customer_id": "c_1"}'


def test_first_occurrence_wins_for_a_repeated_call():
    """A tool called twice with the same arguments and different results is non-deterministic; replaying the
    first is the only reproducible choice."""
    t = trace()
    t["messages"] += [
        {"role": "assistant", "tool_calls": [make_call("c3", "get_customer", {"email": "a@b.com"})]},
        {"role": "tool", "tool_call_id": "c3", "content": '{"customer_id": "DIFFERENT"}'},
    ]
    p = ReplayToolProvider(t)
    assert p.lookup("get_customer", {"email": "a@b.com"}) == '{"customer_id": "c_1"}'


def test_orphan_tool_result_is_skipped_not_guessed():
    t = trace()
    t["messages"].append({"role": "tool", "tool_call_id": "no_such_call", "content": "{}"})
    p = ReplayToolProvider(t)
    assert p.n_recorded == 2


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match=r"strict.*fuzzy"):
        ReplayToolProvider(trace(), policy="lenient")


def test_provider_summary_counts():
    t = trace()
    p = ReplayToolProvider(t)
    p.lookup("get_customer", {"email": "a@b.com"})
    with pytest.raises(Divergence):
        p.lookup("get_customer", {"email": "z@z.com"})
    s = p.summary()
    assert s["replayed"] == 1 and s["diverged"] == 1 and s["fuzzy"] == 0


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ({"x": 1}, {"x": 1}, 1.0),
        ({"x": 1}, {"x": 2}, 0.0),
        ({"x": 1}, {"x": 1, "y": 2}, 0.5),
        ({}, {}, 1.0),
    ],
)
def test_similarity(a, b, expected):
    assert similarity("tool", a, b) == pytest.approx(expected)


def test_similarity_walks_nested_structures():
    assert similarity("t", {"a": {"b": 1}}, {"a": {"b": 1}}) == 1.0
    assert similarity("t", {"a": {"b": 1}}, {"a": {"b": 2}}) == 0.0


def test_divergence_hash_matches_canonical():
    t = trace()
    p = ReplayToolProvider(t)
    with pytest.raises(Divergence) as e:
        p.lookup("get_customer", {"email": "z@z.com"})
    assert e.value.hash == args_hash("get_customer", {"email": "z@z.com"})


def test_divergence_does_not_shadow_exception_args():
    """BaseException.args is a tuple; shadowing it breaks exception handling in subtle ways."""
    d = Divergence("t", {"a": 1}, "h" * 64, None, 0.0)
    assert isinstance(d.args, tuple)
    assert d.call_args == {"a": 1}


def test_tool_calls_made_extracts_in_order():
    t = trace()
    outcome = run_task(t, RecordedTurnClient(t), ReplayToolProvider(t))
    assert tool_calls_made(outcome) == [
        ("get_customer", {"email": "a@b.com"}),
        ("list_orders", {"customer_id": "c_1"}),
    ]


def test_tool_calls_made_skips_unparseable():
    t = trace()
    client = ScriptedTurnClient([
        {"role": "assistant", "tool_calls": [
            {"id": "z", "type": "function", "function": {"name": "get_customer", "arguments": "{bad"}}]},
        {"role": "assistant", "content": "done"},
    ])
    outcome = run_task(t, client, ReplayToolProvider(t))
    assert tool_calls_made(outcome) == []
