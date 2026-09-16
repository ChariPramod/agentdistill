"""Normalization must be lossless for everything training depends on."""

from __future__ import annotations

import json

import pytest

from agentdistill.ingest.normalize import (
    anthropic_to_openai,
    canonical_arguments,
    canonical_messages,
    content_hash,
    extract_task_input,
    normalize_trace,
    openai_to_anthropic,
    roundtrip_equivalent,
    validate_trace,
)


def test_anthropic_to_openai_shape(anthropic_conversation):
    out = anthropic_to_openai(
        anthropic_conversation["system"], anthropic_conversation["messages"], anthropic_conversation["tools"]
    )
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    calls = out["messages"][2]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["search_orders", "search_orders"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"customer_id": "c_9"}
    assert out["tools"][0]["function"]["parameters"]["required"] == ["customer_id"]


def test_parallel_tool_results_become_separate_tool_messages(anthropic_conversation):
    out = anthropic_to_openai(None, anthropic_conversation["messages"], [])
    tool_msgs = [m for m in out["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2"]
    # A tool_result carrying content blocks is flattened to text.
    assert tool_msgs[0]["content"] == "shipped"


def test_roundtrip_preserves_names_arguments_and_pairing(anthropic_conversation):
    once = anthropic_to_openai(
        anthropic_conversation["system"], anthropic_conversation["messages"], anthropic_conversation["tools"]
    )
    back = openai_to_anthropic(once["messages"], once["tools"])
    twice = anthropic_to_openai(back["system"], back["messages"], back["tools"])
    assert roundtrip_equivalent(once, twice)


def test_roundtrip_coalesces_parallel_results_back_into_one_user_turn(anthropic_conversation):
    once = anthropic_to_openai(None, anthropic_conversation["messages"], [])
    back = openai_to_anthropic(once["messages"], [])
    result_turns = [m for m in back["messages"] if m["role"] == "user" and isinstance(m["content"], list)
                    and m["content"][0]["type"] == "tool_result"]
    assert len(result_turns) == 1, "parallel tool results must return as one Anthropic user turn"
    assert len(result_turns[0]["content"]) == 2


def test_malformed_arguments_survive_conversion():
    """The schema_valid filter rejects bad arguments; the converter must not hide them by dropping the call."""
    messages = [{"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c", "type": "function",
                                 "function": {"name": "f", "arguments": "{not json"}}]}]
    back = openai_to_anthropic(messages, [])
    assert back["messages"][0]["content"][0]["input"] == "{not json"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ('{"a": 1, "b": 2}', '{"b":2,"a":1}'),          # key order
        ('{"a": 1}', '{"a":1}'),                          # whitespace
        ('{"a": [1, 2]}', '{"a":[1,2]}'),                 # nested whitespace
    ],
)
def test_canonical_arguments_ignores_formatting(a, b):
    assert canonical_arguments(a) == canonical_arguments(b)


def test_canonical_arguments_keeps_meaningful_difference():
    assert canonical_arguments('{"a": 1}') != canonical_arguments('{"a": 2}')


def test_content_hash_ignores_call_ids(trace):
    other = json.loads(json.dumps(trace))
    for m in other["messages"]:
        for c in m.get("tool_calls") or []:
            c["id"] = "different_id"
        if m.get("tool_call_id"):
            m["tool_call_id"] = "different_id"
    assert content_hash(other) == content_hash(trace), "request-scoped ids must not affect identity"


def test_content_hash_ignores_tool_order(trace):
    from agentdistill.ingest.normalize import canonical_tools

    tools = [
        {"type": "function", "function": {"name": "b", "description": "", "parameters": {}}},
        {"type": "function", "function": {"name": "a", "description": "", "parameters": {}}},
    ]
    assert canonical_tools(tools) == canonical_tools(list(reversed(tools)))


def test_content_hash_changes_with_content(trace):
    other = json.loads(json.dumps(trace))
    other["messages"][-1]["content"] = "something else entirely"
    assert content_hash(other) != content_hash(trace)


def test_message_order_matters():
    a = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    assert canonical_messages(a) != canonical_messages(list(reversed(a)))


def test_normalize_recomputes_derived_fields():
    """A stale count in an export must not be trusted."""
    raw = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": [],
        "n_turns": 99,
        "n_tool_calls": 99,
    }
    t = normalize_trace(raw, source="jsonl")
    assert t["n_turns"] == 2
    assert t["n_tool_calls"] == 1


def test_normalized_trace_validates_against_schema(trace):
    assert validate_trace(trace) == []


def test_task_input_excludes_assistant_output():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the question"},
        {"role": "assistant", "content": "the answer"},
        {"role": "user", "content": "a follow-up after the assistant spoke"},
    ]
    ti = extract_task_input(messages)
    assert ti == {"system": "sys", "user": "the question"}
    assert "answer" not in json.dumps(ti)
    assert "follow-up" not in json.dumps(ti)


def test_id_is_derived_from_content_when_absent():
    t = normalize_trace({"messages": [{"role": "user", "content": "x"}], "tools": []}, source="jsonl")
    assert t["id"].startswith("tr_")
    assert t["content_hash"].startswith(t["id"][3:])
