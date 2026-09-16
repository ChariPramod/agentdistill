"""Mask invariants, derived from each template rather than hard-coded.

`test_build_masks.py` asserts the mask on one template with known text. This file asserts *properties* that must
hold for any template: the targets contain exactly one end-of-turn marker per assistant turn, no assistant
header leaks in, the last target ends the turn, and every tool call is fully inside the targets.

Deriving the header and end-of-turn string from the template itself is what makes this portable. Hard-coding
`<|im_end|>` would pass on ChatML and quietly skip the check everywhere else.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.data.build import IGNORE_INDEX, build_trajectory_sample
from agentdistill.data.template_check import SAMPLE_MESSAGES, SAMPLE_TOOLS, render

# Every template that is prefix-stable and renders tools. `unstable.jinja` and `no_tools.jinja` are excluded
# because they are rejected before masking; their rejection is asserted in test_build_masks.py.
MASKABLE = ["toolchat.jinja", "hermes.jinja", "llama3.jinja"]

MARKER = "REPLY_MARKER_9f3"


def _header_and_eot(tok, tools) -> tuple[str, str]:
    """Derive the assistant header and the end-of-turn text from the template.

    The header is what `add_generation_prompt` appends. The end-of-turn is whatever trails the content of a
    rendered assistant message.
    """
    sys_user = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with_gen = render(tok, sys_user, tools, add_generation_prompt=True)
    without = render(tok, sys_user, tools, add_generation_prompt=False)
    header = with_gen[len(without) :]

    full = render(tok, [*sys_user, {"role": "assistant", "content": MARKER}], tools, add_generation_prompt=False)
    tail = full[full.index(MARKER) + len(MARKER) :]
    return header, tail.strip()


def _target_text(tok, sample) -> str:
    return tok.decode([i for i in sample.labels if i != IGNORE_INDEX], skip_special_tokens=False)


@pytest.fixture
def sample_for(tokenizer_factory):
    def make(template: str):
        tok = tokenizer_factory(template)
        sample = build_trajectory_sample(tok, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=8192)
        assert sample is not None, f"{template}: no sample was built"
        return tok, sample

    return make


@pytest.mark.parametrize("template", MASKABLE)
def test_one_end_of_turn_marker_per_assistant_turn(sample_for, template):
    """Too few and the student never learns to stop; too many and a turn boundary landed inside the targets."""
    tok, sample = sample_for(template)
    _, eot = _header_and_eot(tok, SAMPLE_TOOLS)
    text = _target_text(tok, sample)
    n_assistant = sum(1 for m in SAMPLE_MESSAGES if m["role"] == "assistant")
    assert text.count(eot) == n_assistant, f"expected {n_assistant} × {eot!r} in targets, got {text.count(eot)}"


@pytest.mark.parametrize("template", MASKABLE)
def test_assistant_header_does_not_leak_into_targets(sample_for, template):
    """A leaked header trains the student to emit `assistant\\n` at the top of every reply."""
    tok, sample = sample_for(template)
    header, _ = _header_and_eot(tok, SAMPLE_TOOLS)
    text = _target_text(tok, sample)
    stripped = header.strip()
    if not stripped:
        pytest.skip(f"{template} has an empty assistant header; nothing to leak")
    assert stripped not in text, f"header {stripped!r} leaked into the targets"


@pytest.mark.parametrize("template", MASKABLE)
def test_last_target_ends_the_turn(sample_for, template):
    tok, sample = sample_for(template)
    _, eot = _header_and_eot(tok, SAMPLE_TOOLS)
    assert _target_text(tok, sample).rstrip().endswith(eot)


@pytest.mark.parametrize("template", MASKABLE)
def test_every_tool_call_is_fully_inside_the_targets(sample_for, template):
    """A tool call split across the mask boundary trains half a call."""
    tok, sample = sample_for(template)
    text = _target_text(tok, sample)
    for m in SAMPLE_MESSAGES:
        for c in m.get("tool_calls") or []:
            assert c["function"]["name"] in text, f"tool name {c['function']['name']} missing from targets"
            for value in json.loads(c["function"]["arguments"]).values():
                assert str(value) in text, f"argument value {value!r} missing from targets"


@pytest.mark.parametrize("template", MASKABLE)
def test_no_environment_text_is_in_the_targets(sample_for, template):
    """The model must never be trained to predict a tool result or a user turn."""
    tok, sample = sample_for(template)
    text = _target_text(tok, sample)
    assert "You are a support agent" not in text, "system prompt in targets"
    assert "Where is the order" not in text, "user turn in targets"
    assert "shipped" not in text.replace("Order o_1 shipped yesterday.", ""), "tool result in targets"


@pytest.mark.parametrize("template", MASKABLE)
def test_targets_are_a_strict_subset_of_the_sequence(sample_for, template):
    _tok, sample = sample_for(template)
    assert 0 < sample.n_target_tokens < sample.n_tokens
    assert len(sample.input_ids) == len(sample.labels)
    for tid, lab in zip(sample.input_ids, sample.labels, strict=True):
        assert lab in (IGNORE_INDEX, tid)


@pytest.mark.parametrize("template", MASKABLE)
def test_target_token_count_is_close_to_the_rendered_assistant_text(sample_for, template):
    """Sanity bound: targets should be roughly the assistant turns, not the whole conversation.

    Loose on purpose -- the exact count depends on the template's markers -- but tight enough to catch a mask
    that unmasked everything or nearly nothing.
    """
    tok, sample = sample_for(template)
    assistant_chars = sum(
        len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or []))
        for m in SAMPLE_MESSAGES
        if m["role"] == "assistant"
    )
    full_chars = len(render(tok, SAMPLE_MESSAGES, SAMPLE_TOOLS, add_generation_prompt=False))
    share = sample.n_target_tokens / sample.n_tokens
    expected_share = assistant_chars / full_chars
    assert 0.25 * expected_share < share < 3.0 * expected_share, (
        f"{share:.2%} of tokens are targets; the assistant turns are {expected_share:.2%} of the text"
    )


@pytest.mark.parametrize("template", MASKABLE)
def test_last_turn_mode_keeps_only_the_final_turn(sample_for, tokenizer_factory, template):
    tok = tokenizer_factory(template)
    sample = build_trajectory_sample(
        tok, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=8192, target="last_turn"
    )
    _, eot = _header_and_eot(tok, SAMPLE_TOOLS)
    text = _target_text(tok, sample)
    assert text.count(eot) == 1
    assert "Order o_1 shipped yesterday." in text
    assert "I will look that up." not in text
