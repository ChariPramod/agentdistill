"""DPO pair rendering and validation.

Most wasted DPO runs come from pairs that cannot teach anything: two sides that do the same thing, or two
continuations of different prompts. These reject them before a GPU is involved.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.train.dpo_data import (
    balance_kinds,
    bos_token_text,
    filter_pairs,
    pair_is_valid,
    render_pair,
)
from tests.conftest import make_call


def pair(chosen_tool: str = "refund_order", rejected_tool: str = "cancel_order", **over) -> dict:
    base = {
        "prompt": [
            {"role": "system", "content": "You are support."},
            {"role": "user", "content": "refund my order"},
        ],
        "chosen": [{"role": "assistant", "content": "Refunding.",
                    "tool_calls": [make_call("c1", chosen_tool, {"order_id": "o_1"})]}],
        "rejected": [{"role": "assistant", "content": "Cancelling.",
                      "tool_calls": [make_call("c1", rejected_tool, {"order_id": "o_1"})]}],
        "tools": [{"type": "function", "function": {"name": chosen_tool, "parameters": {"type": "object"}}},
                  {"type": "function", "function": {"name": rejected_tool, "parameters": {"type": "object"}}}],
        "task_id": "t1",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------------------------


def test_a_well_formed_pair_is_valid():
    assert pair_is_valid(pair()) == (True, "")


def test_same_tool_call_with_different_prose_is_rejected():
    """The most common wasted-run cause: the sides differ in wording, not in what they did."""
    ok, why = pair_is_valid(pair(chosen_tool="refund_order", rejected_tool="refund_order"))
    assert not ok and "only differs in wording" in why


def test_acting_versus_answering_is_a_real_difference():
    p = pair()
    p["rejected"] = [{"role": "assistant", "content": "I am afraid I cannot do that."}]
    assert pair_is_valid(p)[0], "calling a tool instead of declining is a decision worth learning"


def test_different_arguments_to_the_same_tool_is_a_real_difference():
    p = pair(rejected_tool="refund_order")
    p["rejected"][0]["tool_calls"] = [make_call("c1", "refund_order", {"order_id": "o_WRONG"})]
    assert pair_is_valid(p)[0]


def test_formatting_only_difference_is_rejected():
    """Whitespace and key order are not a preference; training on them teaches a serialization style."""
    p = pair()
    p["rejected"] = [{
        "role": "assistant", "content": "Refunding.",
        "tool_calls": [{"id": "z", "type": "function",
                        "function": {"name": "refund_order", "arguments": '{"order_id":   "o_1"}'}}],
    }]
    ok, why = pair_is_valid(p)
    assert not ok and "only differs in wording" in why


def test_prompt_ending_on_an_assistant_turn_is_rejected():
    """The two sides would be continuing different contexts, so the comparison is confounded."""
    p = pair()
    p["prompt"] = [*p["prompt"], {"role": "assistant", "content": "thinking"}]
    ok, why = pair_is_valid(p)
    assert not ok and "different contexts" in why


def test_empty_prompt_is_rejected():
    ok, why = pair_is_valid(pair(prompt=[]))
    assert not ok and "empty" in why


@pytest.mark.parametrize("side", ["chosen", "rejected"])
def test_non_assistant_side_is_rejected(side):
    ok, why = pair_is_valid(pair(**{side: [{"role": "user", "content": "x"}]}))
    assert not ok and "not an assistant turn" in why


@pytest.mark.parametrize("side", ["chosen", "rejected"])
def test_empty_side_is_rejected(side):
    ok, why = pair_is_valid(pair(**{side: []}))
    assert not ok and "empty" in why


def test_malformed_tool_arguments_are_rejected():
    p = pair()
    p["chosen"][0]["tool_calls"][0]["function"]["arguments"] = "{not json"
    ok, why = pair_is_valid(p)
    assert not ok and "malformed" in why


def test_text_only_pairs_are_compared_on_content():
    p = pair()
    p["chosen"] = [{"role": "assistant", "content": "I refunded it."}]
    p["rejected"] = [{"role": "assistant", "content": "I cannot help."}]
    assert pair_is_valid(p)[0]
    p["rejected"] = [{"role": "assistant", "content": "I  refunded   it."}]
    ok, why = pair_is_valid(p)
    assert not ok and "say the same thing" in why


def test_filter_pairs_counts_reasons():
    kept, reasons = filter_pairs([pair(), pair(rejected_tool="refund_order"), pair(prompt=[])])
    assert len(kept) == 1
    assert sum(reasons.values()) == 2


# --------------------------------------------------------------------------------------------------------------
# balancing
# --------------------------------------------------------------------------------------------------------------


def test_teacher_pairs_are_capped_against_rollout_pairs():
    """A set dominated by teacher pairs is SFT wearing a DPO loss."""
    pairs = [pair(pair_kind="rollout") for _ in range(3)] + [pair(pair_kind="teacher") for _ in range(10)]
    kept, counts = balance_kinds(pairs, max_teacher_ratio=1.0)
    assert counts == {"rollout": 3, "teacher": 3, "teacher_dropped": 7}
    assert len(kept) == 6


def test_teacher_ratio_is_configurable():
    pairs = [pair(pair_kind="rollout") for _ in range(4)] + [pair(pair_kind="teacher") for _ in range(10)]
    _, counts = balance_kinds(pairs, max_teacher_ratio=0.5)
    assert counts["teacher"] == 2


def test_teacher_pairs_survive_when_there_are_no_rollouts():
    """Round zero has no student rollouts yet; dropping every teacher pair would leave nothing."""
    pairs = [pair(pair_kind="teacher") for _ in range(5)]
    kept, counts = balance_kinds(pairs)
    assert len(kept) == 5 and counts["teacher_dropped"] == 0


def test_untagged_pairs_count_as_rollout():
    kept, counts = balance_kinds([pair()])
    assert counts["rollout"] == 1 and len(kept) == 1


# --------------------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------------------


def test_render_splits_prompt_from_continuations(tokenizer):
    out = render_pair(tokenizer, pair())
    assert out["prompt"]
    assert out["chosen"] != out["rejected"]
    assert not out["chosen"].startswith(out["prompt"]), "the continuation must exclude the prompt"
    assert "refund_order" in out["chosen"]
    assert "cancel_order" in out["rejected"]


def test_render_carries_the_tool_schemas(tokenizer):
    """TRL may not thread `tools` through the template itself, which is why they go in here."""
    out = render_pair(tokenizer, pair())
    assert "refund_order" in out["prompt"] or "refund_order" in out["chosen"]


def test_render_preserves_metadata(tokenizer):
    out = render_pair(tokenizer, pair(pair_kind="teacher"))
    assert out["task_id"] == "t1" and out["pair_kind"] == "teacher"


def test_render_refuses_a_prefix_unstable_template(tokenizer_factory):
    with pytest.raises(ValueError, match="prefix-stable"):
        render_pair(tokenizer_factory("unstable.jinja"), pair())


def test_render_strips_a_leading_bos_when_asked(tokenizer):
    """TRL adds BOS when it tokenizes the pre-rendered prompt; two of them shift every position."""
    out = render_pair(tokenizer, pair(), strip_bos="<|tools|>")
    assert not out["prompt"].startswith("<|tools|>")


def test_bos_token_text_detects_whether_the_template_emits_one(tokenizer):
    # The fixture template does not emit the tokenizer's BOS, so there is nothing to strip.
    assert bos_token_text(tokenizer) is None


def test_rendered_pair_is_json_serializable(tokenizer):
    json.dumps(render_pair(tokenizer, pair()))
