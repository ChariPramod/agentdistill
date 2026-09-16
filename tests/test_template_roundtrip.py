"""Tool-call round trip: does a parser recover what the template rendered?

The fallback regex path runs everywhere. The vLLM path runs only where vLLM is installed (Linux/CUDA), and when
both run they must agree -- otherwise the fallback is giving false assurance about the parser that will actually
serve the model.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.data.template_check import (
    FALLBACK_PARSERS,
    SAMPLE_TOOLS,
    detect_family,
    example_from_schema,
    parse_fallback,
    parse_with_vllm,
    roundtrip_tool_call,
)

PARSEABLE = ["toolchat.jinja", "hermes.jinja", "llama3.jinja"]


# --------------------------------------------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("template", PARSEABLE)
def test_roundtrip_recovers_name_and_arguments(tokenizer_factory, template):
    result = roundtrip_tool_call(tokenizer_factory(template))
    assert result["ok"], f"{template}: {result['detail']} | rendered={result['model_output']!r}"
    (name, args), = result["parsed"]
    assert name == SAMPLE_TOOLS[0]["function"]["name"]
    assert args == result["expected"][0][1]


@pytest.mark.parametrize(("template", "family"),
                         [("toolchat.jinja", "agentdistill_fixture"), ("hermes.jinja", "hermes"),
                          ("llama3.jinja", "llama3_json")])
def test_family_is_detected_from_what_the_template_renders(tokenizer_factory, template, family):
    assert detect_family(tokenizer_factory(template)) == family


def test_prefix_unstable_template_cannot_be_round_tripped(tokenizer_factory):
    """Without prefix stability the model's own output cannot be separated from the prompt."""
    result = roundtrip_tool_call(tokenizer_factory("unstable.jinja"))
    assert not result["ok"]
    assert "prefix-stable" in result["detail"]


def test_unparseable_template_is_reported_not_silently_passed(tokenizer_factory):
    """A template rendering no recognizable tool call must fail, not return an empty parse that compares equal."""
    result = roundtrip_tool_call(tokenizer_factory("no_tools.jinja"))
    assert not result["ok"]
    assert result["parsed"] in (None, [])


def test_explicit_family_overrides_detection(tokenizer_factory):
    result = roundtrip_tool_call(tokenizer_factory("hermes.jinja"), family="llama3_json")
    assert not result["ok"], "forcing the wrong family must fail rather than fall back to detection"


def test_unknown_family_is_rejected(tokenizer_factory):
    from agentdistill.data.template_check import TemplateError

    with pytest.raises(TemplateError, match="unknown tool-call family"):
        parse_fallback("anything", "not_a_family")


# --------------------------------------------------------------------------------------------------------------
# the fallback parsers themselves
# --------------------------------------------------------------------------------------------------------------


def test_hermes_parser_on_canonical_output():
    out = '<tool_call>\n{"name": "search_orders", "arguments": {"customer_id": "c_9"}}\n</tool_call>'
    assert parse_fallback(out, "hermes") == [("search_orders", {"customer_id": "c_9"})]


def test_llama3_parser_on_canonical_output():
    out = '{"name": "search_orders", "parameters": {"customer_id": "c_9"}}'
    assert parse_fallback(out, "llama3_json") == [("search_orders", {"customer_id": "c_9"})]


def test_hermes_parser_recovers_parallel_calls():
    out = ('<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
           '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>')
    assert parse_fallback(out, "hermes") == [("a", {"x": 1}), ("b", {"y": 2})]


def test_parsers_find_nothing_in_plain_prose():
    for family in FALLBACK_PARSERS:
        assert parse_fallback("Your order shipped yesterday.", family) == []


def test_parse_with_vllm_returns_none_when_unavailable():
    """None means 'could not check', which must not be confused with [] meaning 'no call found'."""
    result = parse_with_vllm('<tool_call>{"name":"a","arguments":{}}</tool_call>', None, "hermes")
    try:
        import vllm  # noqa: F401
    except ImportError:
        assert result is None
    else:  # pragma: no cover - only on a CUDA box
        assert result is not None


def test_parse_with_vllm_returns_none_without_a_parser_name():
    assert parse_with_vllm("anything", None, None) is None


# --------------------------------------------------------------------------------------------------------------
# vLLM agreement, where vLLM exists
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("template", "family", "parser_name"),
                         [("hermes.jinja", "hermes", "hermes"), ("llama3.jinja", "llama3_json", "llama3_json")])
def test_vllm_and_fallback_agree(tokenizer_factory, template, family, parser_name):
    """Where both parsers run, they must agree -- otherwise the fallback is false assurance."""
    pytest.importorskip("vllm", reason="vLLM is Linux/CUDA only")
    tok = tokenizer_factory(template)
    trip = roundtrip_tool_call(tok, family=family)
    via_vllm = roundtrip_tool_call(tok, parser_name=parser_name, family=family)
    assert via_vllm["parser"] == "vllm"
    assert via_vllm["parsed"] == trip["parsed"]


# --------------------------------------------------------------------------------------------------------------
# schema example generation
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": "string"}, "x"),
        ({"type": "integer"}, 7),
        ({"type": "number"}, 7.5),
        ({"type": "boolean"}, True),
        ({"type": "string", "enum": ["a", "b"]}, "a"),
        ({"type": "array", "items": {"type": "integer"}}, [7]),
    ],
)
def test_example_from_schema_scalars(schema, expected):
    assert example_from_schema(schema) == expected


def test_example_from_schema_fills_only_required_properties():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
    }
    assert example_from_schema(schema) == {"a": "x"}


def test_example_from_schema_nests():
    schema = {
        "type": "object",
        "properties": {"addr": {"type": "object", "properties": {"city": {"type": "string"}},
                                "required": ["city"]}},
        "required": ["addr"],
    }
    assert example_from_schema(schema) == {"addr": {"city": "x"}}


def test_example_round_trips_through_json():
    """The example becomes a tool call's `arguments` string, so it must survive serialization."""
    example = example_from_schema(SAMPLE_TOOLS[0]["function"]["parameters"])
    assert json.loads(json.dumps(example)) == example
