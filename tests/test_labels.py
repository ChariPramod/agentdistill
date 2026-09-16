"""Turn labels for the confidence gate.

Task success alone is too coarse: a task can succeed despite a bad turn that a later turn repaired. Labelling
that turn "good" would teach the gate to be confident about exactly the mistakes it exists to catch.
"""

from __future__ import annotations

from agentdistill.cascade.labels import (
    call_signature,
    label_mix,
    label_rollouts,
    label_turns,
    prefix_key,
    weak_label_share,
)
from tests.conftest import make_call


def asst(content: str | None = None, tool: str | None = None, args: dict | None = None) -> dict:
    m: dict = {"role": "assistant", "content": content}
    if tool:
        m["tool_calls"] = [make_call("c1", tool, args or {"order_id": "o_1"})]
    return m


def tool_result(content: str = '{"ok": true}') -> dict:
    return {"role": "tool", "tool_call_id": "c1", "content": content}


BASE = [{"role": "system", "content": "sys"}, {"role": "user", "content": "refund my order"}]


# --------------------------------------------------------------------------------------------------------------
# the five label kinds
# --------------------------------------------------------------------------------------------------------------


def test_task_failure_labels_every_turn_bad():
    messages = [*BASE, asst("Looking", "lookup"), tool_result(), asst("Done")]
    labels = label_turns(messages, task_success=False, teacher_messages=messages)
    assert [good for _, good, _ in labels] == [False, False]
    assert {how for _, _, how in labels} == {"task_failed"}


def test_teacher_match_on_the_same_prefix():
    messages = [*BASE, asst("Looking", "lookup"), tool_result(), asst("Done")]
    labels = label_turns(messages, task_success=True, teacher_messages=messages)
    assert all(good for _, good, _ in labels)
    assert {how for _, _, how in labels} == {"teacher_match"}


def test_teacher_mismatch_when_the_student_chose_differently():
    teacher = [*BASE, asst("Refunding", "refund_order"), tool_result(), asst("Done")]
    student = [*BASE, asst("Cancelling", "cancel_order"), tool_result(), asst("Done")]
    labels = label_turns(student, task_success=True, teacher_messages=teacher)
    assert labels[0][1] is False and labels[0][2] == "teacher_mismatch"


def test_corrected_later_when_the_tool_errored():
    messages = [
        *BASE,
        asst("Trying", "refund_order"),
        tool_result('{"error": "order is still processing"}'),
        asst("Explaining instead"),
    ]
    labels = label_turns(messages, task_success=True, teacher_messages=None)
    assert labels[0][2] == "corrected_later" and labels[0][1] is False


def test_corrected_later_when_the_same_tool_is_retried_with_different_arguments():
    messages = [
        *BASE,
        asst("First try", "refund_order", {"order_id": "WRONG"}),
        tool_result(),
        asst("Second try", "refund_order", {"order_id": "o_1"}),
        tool_result(),
        asst("Done"),
    ]
    labels = label_turns(messages, task_success=True, teacher_messages=None)
    assert labels[0][2] == "corrected_later"
    assert labels[1][2] == "uncorrected", "the retry itself was not corrected"


def test_uncorrected_when_nothing_visibly_went_wrong():
    messages = [*BASE, asst("Looking", "lookup"), tool_result(), asst("Done")]
    labels = label_turns(messages, task_success=True, teacher_messages=None)
    assert {how for _, _, how in labels} == {"uncorrected"}
    assert all(good for _, good, _ in labels)


# --------------------------------------------------------------------------------------------------------------
# prefix identity
# --------------------------------------------------------------------------------------------------------------


def test_prefix_identity_includes_argument_values():
    """A call with different arguments leads to a different state, so the teacher's next turn is not a reference."""
    teacher = [*BASE, asst("Looking", "lookup", {"order_id": "o_1"}), tool_result(), asst("Done")]
    student = [*BASE, asst("Looking", "lookup", {"order_id": "o_9"}), tool_result(), asst("Done")]
    labels = label_turns(student, task_success=True, teacher_messages=teacher)
    # The first turn shares the empty prefix, so it is compared; the second does not.
    assert labels[0][2] == "teacher_mismatch"
    assert labels[1][2] == "uncorrected", "no teacher reference exists for a prefix the teacher never saw"


def test_prefix_key_ignores_argument_formatting():
    a = [asst("x", "f", {"a": 1, "b": 2})]
    b = [{"role": "assistant", "content": "x", "tool_calls": [
        {"id": "z", "type": "function", "function": {"name": "f", "arguments": '{"b":2,"a":1}'}}]}]
    assert prefix_key(a, 1) == prefix_key(b, 1)


def test_call_signature_is_order_insensitive():
    a = {"role": "assistant", "tool_calls": [make_call("1", "f", {}), make_call("2", "g", {})]}
    b = {"role": "assistant", "tool_calls": [make_call("9", "g", {}), make_call("8", "f", {})]}
    assert call_signature(a) == call_signature(b)


def test_call_signature_of_a_text_turn_is_empty():
    assert call_signature({"role": "assistant", "content": "hello"}) == frozenset()


def test_unparseable_arguments_do_not_collide_with_valid_ones():
    bad = {"role": "assistant", "tool_calls": [
        {"id": "z", "type": "function", "function": {"name": "f", "arguments": "{broken"}}]}
    good = {"role": "assistant", "tool_calls": [make_call("z", "f", {})]}
    assert call_signature(bad) != call_signature(good)


# --------------------------------------------------------------------------------------------------------------
# the label mix
# --------------------------------------------------------------------------------------------------------------


def test_label_mix_counts_each_rule():
    labels = [(0, True, "teacher_match"), (1, False, "teacher_mismatch"), (2, True, "uncorrected")]
    assert label_mix(labels) == {"teacher_match": 1, "teacher_mismatch": 1, "uncorrected": 1}


def test_weak_share_measures_how_much_rests_on_assumption():
    """A gate trained mostly on `uncorrected` labels has an AUROC that should not be taken at face value."""
    labels = [(0, True, "uncorrected"), (1, True, "uncorrected"), (2, True, "teacher_match")]
    assert weak_label_share(labels) == 2 / 3
    assert weak_label_share([]) == 0.0


def test_label_rollouts_flattens_and_reports_the_mix():
    teacher = {"id": "t1", "task_id": "task1",
               "messages": [*BASE, asst("Refunding", "refund_order"), tool_result(), asst("Done")]}
    rollouts = [
        {"id": "r1", "task_id": "task1", "success": True, "messages": teacher["messages"]},
        {"id": "r2", "task_id": "task1", "success": False,
         "messages": [*BASE, asst("Cancelling", "cancel_order"), tool_result(), asst("Done")]},
    ]
    records, stats = label_rollouts(rollouts, {"task1": teacher})
    assert len(records) == 4
    assert stats["n"] == 4
    assert stats["mix"]["teacher_match"] == 2
    assert stats["mix"]["task_failed"] == 2
    assert stats["positive_rate"] == 0.5
    assert all("message" in r and "good" in r for r in records)


def test_label_rollouts_without_a_teacher_trace():
    rollouts = [{"id": "r1", "task_id": "task1", "success": True,
                 "messages": [*BASE, asst("Looking", "lookup"), tool_result(), asst("Done")]}]
    records, stats = label_rollouts(rollouts, {})
    assert len(records) == 2
    assert stats["weak_share"] == 1.0, "with no teacher reference every label is an assumption"
