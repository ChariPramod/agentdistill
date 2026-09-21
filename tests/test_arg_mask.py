"""Argument token masking.

The gate's most useful features are about the arguments specifically -- a model is often sure which tool to call
and much less sure what to put in it. If the mask is wrong, those features describe the wrong tokens and the gate
learns from noise.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.cascade.arg_mask import arg_char_spans, arg_token_mask, mask_coverage, token_spans


def call(name: str, arguments: str) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def masked(tokens: list[str], mask: list[bool]) -> str:
    return "".join(t for t, m in zip(tokens, mask, strict=True) if m)


def test_token_spans_are_contiguous():
    spans = token_spans(["ab", "cde", "f"])
    assert spans == [(0, 2), (2, 5), (5, 6)]


def test_mask_covers_exactly_the_argument_json():
    tokens = ["<tool_call>", '{"order', '_id": "o', '_1", "amount": ', "20}", "</tool_call>"]
    text = "".join(tokens)
    mask = arg_token_mask(tokens, text, [call("refund_order", '{"order_id": "o_1", "amount": 20}')])
    assert masked(tokens, mask) == '{"order_id": "o_1", "amount": 20}'
    assert not mask[0] and not mask[-1], "template markers are not arguments"


def test_mask_excludes_surrounding_prose():
    tokens = ["I ", "will ", "refund ", "that", ". ", '{"order_id": "o_1"}', " done"]
    mask = arg_token_mask(tokens, "".join(tokens), [call("refund_order", '{"order_id": "o_1"}')])
    assert masked(tokens, mask) == '{"order_id": "o_1"}'


def test_mask_handles_a_reserialized_compact_form():
    """The server may have compacted the JSON on the way out."""
    tokens = ['{"a":1,', '"b":2}']
    mask = arg_token_mask(tokens, "".join(tokens), [call("f", '{"a": 1, "b": 2}')])
    assert all(mask)


def test_mask_falls_back_to_individual_values():
    """A partial mask over the values beats no argument features at all."""
    tokens = ["name", "=", "o_1", " amount", "=", "20"]
    mask = arg_token_mask(tokens, "".join(tokens), [call("f", '{"order_id": "o_1", "amount": 20}')])
    assert mask[2] and mask[5], "the values were located even though the JSON was not"
    assert not mask[0]


def test_parallel_calls_are_both_masked():
    tokens = ['{"a": 1}', " and ", '{"b": 2}']
    mask = arg_token_mask(tokens, "".join(tokens), [call("f", '{"a": 1}'), call("g", '{"b": 2}')])
    assert mask == [True, False, True]


def test_repeated_identical_arguments_advance_the_cursor():
    """Two calls with the same arguments must mask two spans, not the same one twice."""
    tokens = ['{"a": 1}', "|", '{"a": 1}']
    mask = arg_token_mask(tokens, "".join(tokens), [call("f", '{"a": 1}'), call("f", '{"a": 1}')])
    assert mask == [True, False, True]


def test_no_tool_calls_masks_nothing():
    tokens = ["Your ", "order ", "shipped."]
    assert arg_token_mask(tokens, "".join(tokens), []) == [False, False, False]
    assert mask_coverage([False, False, False]) == 0.0


def test_unparseable_arguments_still_match_verbatim():
    tokens = ["{not ", "json"]
    mask = arg_token_mask(tokens, "".join(tokens), [call("f", "{not json")])
    assert all(mask)


def test_arguments_absent_from_the_text_mask_nothing():
    """A server that did not echo the arguments must not produce a bogus mask."""
    tokens = ["Your ", "order ", "shipped."]
    mask = arg_token_mask(tokens, "".join(tokens), [call("f", '{"order_id": "zzz"}')])
    assert not any(mask)


def test_tokens_that_do_not_concatenate_to_the_text_align_on_the_tokens():
    """Byte-fallback tokens: the spans index into the joined string, so that is what must be searched."""
    tokens = ['{"a":', ' "x"}']
    reported_text = 'DIFFERENT {"a": "x"}'
    mask = arg_token_mask(tokens, reported_text, [call("f", '{"a": "x"}')])
    assert all(mask)


def test_mask_length_always_matches_token_count():
    tokens = ["a", "b", "c"]
    for calls in ([], [call("f", "{}")], [call("f", '{"x": 1}')]):
        assert len(arg_token_mask(tokens, "".join(tokens), calls)) == len(tokens)


def test_empty_token_list():
    assert arg_token_mask([], "", [call("f", "{}")]) == []


@pytest.mark.parametrize("arguments", ['{"a": 1}', '{"a": "text with spaces"}', '{"nested": {"b": [1, 2]}}'])
def test_round_trips_for_several_argument_shapes(arguments):
    tokens = ["pre ", arguments, " post"]
    mask = arg_token_mask(tokens, "".join(tokens), [call("f", arguments)])
    assert masked(tokens, mask) == arguments


def test_mask_matches_an_object_the_template_spaced_its_own_way():
    """Since arguments reach a chat template as an object, the text carries the *template's* serialization.

    Nothing here can know what separators it chose, so nothing here may depend on them. This spacing matches
    neither the trace's string nor either `json.dumps` spelling.
    """
    text = '<tool_call>\n{"name": "f", "arguments": {"a" : 1 ,  "b" : [1, 2]}}\n</tool_call>'
    spans = arg_char_spans(text, [call("f", '{"a": 1, "b": [1, 2]}')])
    assert len(spans) == 1
    a, b = spans[0]
    assert json.loads(text[a:b]) == {"a": 1, "b": [1, 2]}
    assert text[a] == "{" and text[b - 1] == "}"


def test_mask_matches_an_object_the_template_did_not_escape():
    """The wire-format string escapes non-ASCII; a template's `tojson` does not. Only a structural match
    survives that, and this is the case the pinned Qwen template actually produces."""
    text = 'call {"city": "München"} end'
    spans = arg_char_spans(text, [call("f", json.dumps({"city": "München"}))])
    assert [text[a:b] for a, b in spans] == ['{"city": "München"}']


def test_mask_matches_an_object_whose_keys_were_reordered():
    """Two serializations of one object are the same arguments. Matching bytes would say otherwise."""
    text = 'x {"b": 2, "a": 1} y'
    assert arg_char_spans(text, [call("f", '{"a": 1, "b": 2}')]) == [(2, 18)]


def test_the_argument_object_is_matched_not_the_call_that_wraps_it():
    """The hermes block is `{"name": ..., "arguments": {...}}`. The outer object is the first one in the text
    and is not the arguments; masking it would mark the tool name as an argument token."""
    text = '{"name": "f", "arguments": {"a" : 1}}'
    a, b = arg_char_spans(text, [call("f", '{"a": 1}')])[0]
    assert text[a:b] == '{"a" : 1}'
    assert '"name"' not in text[a:b]


def test_a_different_object_in_the_text_is_not_matched():
    """A structural search must not match an object that merely looks like one."""
    text = '<tool_call>\n{"name": "f", "arguments": {"order_id": "o_99"}}\n</tool_call>'
    assert arg_char_spans(text, [call("f", '{"order_id": "o_1"}')]) == []


def test_unparseable_arguments_are_never_matched_structurally():
    """A malformed call must not be quietly paired with a well-formed object in the text: the gate would then
    score a turn as if the model had produced arguments it did not.

    Verbatim matching still applies (see `test_unparseable_arguments_still_match_verbatim`) -- it is the
    *structural* step that has nothing to compare and must therefore find nothing.
    """
    text = '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'
    assert arg_char_spans(text, [call("f", '{"a": 1, oops}')]) == []


def test_arg_char_spans_are_ordered_and_within_the_text():
    text = 'x {"a": 1} y {"b": 2}'
    spans = arg_char_spans(text, [call("f", '{"a": 1}'), call("g", '{"b": 2}')])
    assert spans == [(2, 10), (13, 21)]
    for a, b in spans:
        assert json.loads(text[a:b])
