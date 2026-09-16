"""Token-level assertions on loss masks, across three chat templates.

This is the test the whole training stage rests on. If the mask is wrong, nothing downstream reveals it: the loss
curve looks fine while the model learns to predict tool results and user turns.
"""

from __future__ import annotations

import pytest

from agentdistill.data.build import (
    IGNORE_INDEX,
    MaskingError,
    assistant_spans,
    build_samples_for_trace,
    build_trajectory_sample,
    build_turn_windows,
    decode_targets,
)
from agentdistill.data.template_check import SAMPLE_MESSAGES, SAMPLE_TOOLS, check_template, render

# --------------------------------------------------------------------------------------------------------------
# template gating
# --------------------------------------------------------------------------------------------------------------


def test_toolchat_template_passes_every_check(tokenizer_factory):
    report = check_template(tokenizer_factory("toolchat.jinja"), "toolchat")
    assert report.ok, [(c.name, c.detail) for c in report.failures]


def test_template_ignoring_tools_is_rejected(tokenizer_factory):
    report = check_template(tokenizer_factory("no_tools.jinja"), "no_tools")
    assert not report.ok
    assert [c.name for c in report.failures] == ["accepts_tools"]
    with pytest.raises(Exception, match="accepts_tools"):
        report.raise_if_failed()


def test_prefix_unstable_template_is_rejected(tokenizer_factory):
    report = check_template(tokenizer_factory("unstable.jinja"), "unstable")
    assert not report.ok
    assert "prefix_stable" in [c.name for c in report.failures]


def test_prefix_unstable_template_raises_the_named_error(tokenizer_factory):
    """The failure must be a MaskingError naming prefix stability, not a silent wrong mask."""
    tok = tokenizer_factory("unstable.jinja")
    with pytest.raises(MaskingError, match="not prefix-stable"):
        build_trajectory_sample(tok, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096)


# --------------------------------------------------------------------------------------------------------------
# mask correctness
# --------------------------------------------------------------------------------------------------------------


def test_spans_cover_exactly_the_assistant_turns(tokenizer):
    full, spans = assistant_spans(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    assert len(spans) == 2, "two assistant turns in the sample conversation"
    for a, b in spans:
        assert 0 <= a < b <= len(full)
    # Spans must not overlap and must be ordered.
    assert spans[0][1] <= spans[1][0]


def test_trained_text_is_exactly_the_assistant_turns(tokenizer):
    sample = build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096)
    segments = decode_targets(tokenizer, sample)
    assert len(segments) == 2
    joined = " ".join(segments)

    # Everything the assistant said and did is present.
    assert "I will look that up." in joined
    assert "search_orders" in joined
    assert "c_9" in joined
    assert "Order o_1 shipped yesterday." in joined

    # Nothing the assistant did not produce is present.
    assert "You are a support agent" not in joined, "system prompt must be masked"
    assert "Where is the order" not in joined, "user turn must be masked"
    assert "status" not in joined, "tool result must be masked"


def test_masked_token_count_matches_unmasked_positions(tokenizer):
    sample = build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096)
    assert sample.n_target_tokens == sum(1 for x in sample.labels if x != IGNORE_INDEX)
    assert 0 < sample.n_target_tokens < sample.n_tokens
    assert len(sample.input_ids) == len(sample.labels)


def test_labels_equal_input_ids_where_unmasked(tokenizer):
    sample = build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096)
    for tid, lab in zip(sample.input_ids, sample.labels, strict=True):
        assert lab in (IGNORE_INDEX, tid)


def test_end_of_turn_marker_is_trained(tokenizer):
    """A student that never learns its end-of-turn token never stops generating."""
    sample = build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096)
    segments = decode_targets(tokenizer, sample)
    assert all("<|end|>" in seg for seg in segments), segments


def test_last_turn_target_mode_trains_only_the_final_turn(tokenizer):
    sample = build_trajectory_sample(
        tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096, target="last_turn"
    )
    segments = decode_targets(tokenizer, sample)
    assert len(segments) == 1
    assert "Order o_1 shipped yesterday." in segments[0]
    assert "I will look that up." not in segments[0]


def test_conversation_with_no_assistant_turn_yields_nothing(tokenizer):
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert build_trajectory_sample(tokenizer, messages, [], max_seq_len=4096) is None


def test_oversized_trajectory_returns_none(tokenizer):
    assert build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=8) is None


# --------------------------------------------------------------------------------------------------------------
# turn windows
# --------------------------------------------------------------------------------------------------------------


def _long_conversation(n_turns: int = 6) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "You are a support agent."},
                            {"role": "user", "content": "Where is the order for customer c_9?"}]
    for i in range(n_turns):
        messages.append(
            {
                "role": "assistant",
                "content": f"Step {i}: checking the order status for this customer now.",
                "tool_calls": [
                    {"id": f"c{i}", "type": "function",
                     "function": {"name": "search_orders", "arguments": '{"customer_id": "c_9", "limit": 5}'}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": '[{"order_id": "o_1"}]'})
    messages.append({"role": "assistant", "content": "Order o_1 shipped yesterday."})
    return messages


def test_turn_windows_produce_one_sample_per_assistant_turn(tokenizer):
    messages = _long_conversation(4)
    windows = list(build_turn_windows(tokenizer, messages, SAMPLE_TOOLS, max_seq_len=4096, window_turns=4))
    n_assistant = sum(1 for m in messages if m["role"] == "assistant")
    assert len(windows) == n_assistant
    assert all(w.kind == "turn_window" for w in windows)


def test_turn_window_never_starts_on_an_orphan_tool_result(tokenizer):
    """A window beginning with a tool result shows an answer to a call the model cannot see."""
    messages = _long_conversation(6)
    system = [messages[0]]
    body = messages[1:]
    for w in build_turn_windows(tokenizer, messages, SAMPLE_TOOLS, max_seq_len=4096, window_turns=3):
        start = w.meta["window_start"]
        if start > 0:
            assert body[start]["role"] != "tool", f"window starts on an orphan tool result at {start}"
    assert system  # keeps the fixture honest


def test_long_trajectory_falls_back_to_windows(tokenizer):
    trace = {"id": "long", "messages": _long_conversation(8), "tools": SAMPLE_TOOLS}
    samples, note = build_samples_for_trace(tokenizer, trace, max_seq_len=200, window_turns=2)
    assert samples, note
    assert all(s.kind == "turn_window" for s in samples)
    assert "turn windows" in note


def test_windows_disabled_drops_the_trace(tokenizer):
    trace = {"id": "long", "messages": _long_conversation(8), "tools": SAMPLE_TOOLS}
    samples, note = build_samples_for_trace(tokenizer, trace, max_seq_len=200, windows_for_long=False)
    assert samples == []
    assert "windows are disabled" in note


def test_fitting_trajectory_yields_one_trajectory_sample(tokenizer):
    trace = {"id": "short", "messages": SAMPLE_MESSAGES, "tools": SAMPLE_TOOLS}
    samples, note = build_samples_for_trace(tokenizer, trace, max_seq_len=4096)
    assert len(samples) == 1
    assert samples[0].kind == "trajectory"
    assert note == ""


# --------------------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------------------


def test_render_is_prefix_of_full_at_every_assistant_boundary(tokenizer):
    full = render(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS)
    for i, m in enumerate(SAMPLE_MESSAGES):
        if m["role"] == "assistant":
            assert full.startswith(render(tokenizer, SAMPLE_MESSAGES[:i], SAMPLE_TOOLS, add_generation_prompt=True))


def test_sample_hash_is_stable_and_content_sensitive(tokenizer):
    a = build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096)
    b = build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096)
    assert a.hash() == b.hash()
    c = build_trajectory_sample(tokenizer, SAMPLE_MESSAGES, SAMPLE_TOOLS, max_seq_len=4096, target="last_turn")
    assert a.hash() != c.hash(), "a different mask must produce a different sample hash"
