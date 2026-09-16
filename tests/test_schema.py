"""Tool-call schema validation and call/result pairing."""

from __future__ import annotations

import json

from agentdistill.curate.schema import tool_calls_valid, tool_results_paired, tool_schemas
from tests.conftest import make_call, make_trace


def test_valid_trace_passes(trace):
    ok, problems = tool_calls_valid(trace)
    assert ok and problems == []


def test_unparseable_arguments_are_rejected():
    t = make_trace("t")
    t["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{not json"
    ok, problems = tool_calls_valid(t)
    assert not ok
    assert "not JSON" in problems[0]
    assert "turn 2" in problems[0], "the problem must name the turn"


def test_arguments_violating_the_schema_are_rejected():
    t = make_trace("t", args={"customer_id": "c_9", "limit": "not a number"})
    ok, problems = tool_calls_valid(t)
    assert not ok
    assert "limit" in problems[0] or "not of type" in problems[0]


def test_missing_required_argument_is_rejected():
    t = make_trace("t", args={"limit": 5})
    ok, problems = tool_calls_valid(t)
    assert not ok
    assert "customer_id" in problems[0]


def test_unknown_tool_is_rejected():
    t = make_trace("t")
    t["messages"][2]["tool_calls"][0]["function"]["name"] = "no_such_tool"
    ok, problems = tool_calls_valid(t)
    assert not ok
    assert "unknown tool no_such_tool" in problems[0]


def test_non_object_arguments_are_rejected():
    t = make_trace("t")
    t["messages"][2]["tool_calls"][0]["function"]["arguments"] = json.dumps([1, 2, 3])
    ok, problems = tool_calls_valid(t)
    assert not ok
    assert "not a JSON object" in problems[0]


def test_invalid_tool_schema_is_reported_not_crashed():
    t = make_trace("t")
    t["tools"][0]["function"]["parameters"] = {"type": "not-a-real-type"}
    ok, problems = tool_calls_valid(t)
    assert not ok
    assert "invalid parameter schema" in problems[0]


def test_unanswered_call_is_rejected():
    """A call with no result means the trace was truncated; training on it teaches the model to stop."""
    t = make_trace("t")
    t["messages"] = [m for m in t["messages"] if m["role"] != "tool"]
    ok, problems = tool_results_paired(t)
    assert not ok
    assert "has no result" in problems[0]


def test_orphan_result_is_rejected():
    t = make_trace("t")
    t["messages"].insert(4, {"role": "tool", "tool_call_id": "nonexistent", "content": "{}"})
    ok, problems = tool_results_paired(t)
    assert not ok
    assert "answers no call" in problems[0]


def test_parallel_calls_pair_correctly():
    t = make_trace("t")
    t["messages"][2]["tool_calls"].append(make_call("c2", "search_orders", {"customer_id": "c_10"}))
    t["messages"].insert(4, {"role": "tool", "tool_call_id": "c2", "content": "[]"})
    ok, problems = tool_results_paired(t)
    assert ok, problems


def test_tool_schemas_extracts_by_name(trace):
    schemas = tool_schemas(trace)
    assert "search_orders" in schemas
    assert schemas["search_orders"]["required"] == ["customer_id"]
