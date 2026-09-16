"""The cascade client and its accounting.

The mechanism is simple; what has to be right is the bookkeeping. A cascade that ignores the tokens it generated
and threw away looks cheaper than it is, and that number goes straight into the cost report.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentdistill.cascade.calibrate import assert_feature_order
from agentdistill.cascade.client import CascadeTurnClient, TurnRecord
from agentdistill.cascade.features import DEFAULT_FEATURES
from tests.conftest import make_call

FEATURES = list(DEFAULT_FEATURES)


def choice(text: str = "ok", tool: str | None = None, n_tokens: int = 4, logprob: float = -0.2) -> dict:
    tokens = [{"token": f"t{i}", "logprob": logprob, "top_logprobs": [{"token": f"t{i}", "logprob": logprob}]}
              for i in range(n_tokens)]
    message: dict = {"role": "assistant", "content": text if not tool else None}
    if tool:
        message["tool_calls"] = [make_call("c1", tool, {"order_id": "o_1"})]
    return {"message": message, "logprobs": {"content": tokens}, "text": "".join(t["token"] for t in tokens)}


class StubBackend:
    def __init__(self, reply: dict, label: str) -> None:
        self.reply, self.label = reply, label
        self.calls: list[dict] = []

    def chat(self, messages, tools, n=1, logprobs=False):
        self.calls.append({"n": n, "logprobs": logprobs, "messages": len(messages)})
        return [self.reply for _ in range(n)]


class StubCalibrator:
    """Returns a fixed probability, so routing can be tested without fitting anything."""

    def __init__(self, p: float) -> None:
        self.p = p
        self.seen: list[np.ndarray] = []

    def predict_proba(self, X):
        self.seen.append(X)
        return np.array([[1 - self.p, self.p]])


def build(p: float, threshold: float = 0.5, **kw) -> tuple[CascadeTurnClient, StubBackend, StubBackend]:
    student = StubBackend(choice("student answer", tool="refund_order"), "student")
    teacher = StubBackend(choice("teacher answer", tool="cancel_order"), "teacher")
    client = CascadeTurnClient(
        student=student, teacher=teacher, calibrator=StubCalibrator(p),
        feature_names=FEATURES, threshold=threshold, **kw,
    )
    return client, student, teacher


MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "refund please"}]


# --------------------------------------------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------------------------------------------


def test_confident_turn_stays_with_the_student():
    client, _student, teacher = build(p=0.9, threshold=0.5)
    out = client.next_turn(MESSAGES, [])
    assert out["tool_calls"][0]["function"]["name"] == "refund_order"
    assert teacher.calls == [], "the teacher must not be called when the gate is satisfied"
    assert client.log[0].arm == "student" and not client.log[0].escalated


def test_unconfident_turn_escalates():
    client, _student, teacher = build(p=0.1, threshold=0.5)
    out = client.next_turn(MESSAGES, [])
    assert out["tool_calls"][0]["function"]["name"] == "cancel_order"
    assert len(teacher.calls) == 1
    assert client.log[0].arm == "teacher" and client.log[0].escalated


def test_threshold_is_inclusive():
    client, _, teacher = build(p=0.5, threshold=0.5)
    client.next_turn(MESSAGES, [])
    assert teacher.calls == [], "p >= tau keeps the student"


def test_student_is_sampled_with_extra_samples_and_logprobs():
    """Self-consistency needs k extra samples; the features need logprobs."""
    client, student, _ = build(p=0.9, k_samples=2)
    client.next_turn(MESSAGES, [])
    assert student.calls[0]["n"] == 3
    assert student.calls[0]["logprobs"] is True


def test_teacher_is_called_without_logprobs():
    client, _, teacher = build(p=0.1)
    client.next_turn(MESSAGES, [])
    assert teacher.calls[0]["logprobs"] is False and teacher.calls[0]["n"] == 1


def test_escalate_everything_never_calls_the_student():
    """The documented default when the gate is missing or unusable."""
    client, student, teacher = build(p=0.99, escalate_everything=True)
    client.next_turn(MESSAGES, [])
    assert student.calls == []
    assert len(teacher.calls) == 1
    assert client.log[0].escalated and client.log[0].student_tokens == 0


# --------------------------------------------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------------------------------------------


def test_wasted_tokens_are_counted_on_escalated_turns():
    """The student generated them and they were thrown away; they are still paid for."""
    client, _, _ = build(p=0.1, threshold=0.5)
    client.next_turn(MESSAGES, [])
    s = client.summary()
    assert s["escalations"] == 1
    assert s["wasted_student_tokens"] == 4, "every token of the discarded generation"


def test_kept_turns_do_not_count_as_wasted():
    client, _, _ = build(p=0.9, threshold=0.5)
    client.next_turn(MESSAGES, [])
    s = client.summary()
    assert s["wasted_student_tokens"] == 0
    assert s["student_tokens"] == 4


def test_summary_over_a_mixed_trajectory():
    client, _, _ = build(p=0.9, threshold=0.5)
    client.next_turn(MESSAGES, [])
    client.calibrator = StubCalibrator(0.1)
    client.next_turn(MESSAGES, [])
    client.next_turn(MESSAGES, [])
    s = client.summary()
    assert s["turns"] == 3 and s["escalations"] == 2
    assert s["escalation_rate"] == pytest.approx(2 / 3)
    assert s["wasted_student_tokens"] == 8


def test_summary_of_an_empty_run():
    client, _, _ = build(p=0.9)
    s = client.summary()
    assert s["turns"] == 0 and s["escalation_rate"] == 0.0


def test_reset_clears_the_log_between_tasks():
    client, _, _ = build(p=0.9)
    client.next_turn(MESSAGES, [])
    client.reset()
    assert client.summary()["turns"] == 0


def test_turn_index_advances_with_the_conversation():
    client, _, _ = build(p=0.9)
    client.next_turn(MESSAGES, [])
    client.next_turn([*MESSAGES, {"role": "assistant", "content": "x"}], [])
    assert [r.turn_idx for r in client.log] == [0, 1]


# --------------------------------------------------------------------------------------------------------------
# features reaching the calibrator
# --------------------------------------------------------------------------------------------------------------


def test_the_calibrator_sees_the_configured_feature_order():
    client, _, _ = build(p=0.9)
    client.next_turn(MESSAGES, [])
    assert client.calibrator.seen[0].shape == (1, len(FEATURES))


def test_a_reordered_calibration_is_refused():
    """A reordered vector scores silently and wrongly, so it is caught before scoring."""
    with pytest.raises(ValueError, match="confident nonsense"):
        assert_feature_order({"feature_order": list(reversed(FEATURES))}, FEATURES)


def test_matching_feature_order_passes():
    assert assert_feature_order({"feature_order": FEATURES}, FEATURES) is None


# --------------------------------------------------------------------------------------------------------------
# the harness records what the gate did
# --------------------------------------------------------------------------------------------------------------


def test_harness_copies_escalations_onto_the_outcome():
    from agentdistill.eval.harness import run_task
    from agentdistill.eval.replay import ReplayToolProvider

    trace = {
        "id": "t1", "task_id": "t1", "tools": [],
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "refund please"},
            {"role": "assistant", "content": "done"},
        ],
    }

    class AlwaysEscalates:
        def __init__(self):
            self.inner, _, _ = build(p=0.0, threshold=0.5)
            self.inner.teacher.reply = choice("teacher answer")

        def next_turn(self, messages, tools):
            return self.inner.next_turn(messages, tools)

        def summary(self):
            return self.inner.summary()

        def reset(self):
            self.inner.reset()

    outcome = run_task(trace, AlwaysEscalates(), ReplayToolProvider(trace))
    assert outcome.escalations == 1
    assert outcome.wasted_student_tokens == 4
    assert outcome.to_row()["escalations"] == 1


def test_a_plain_client_reports_no_escalations():
    from agentdistill.eval.clients import RecordedTurnClient
    from agentdistill.eval.harness import run_task
    from agentdistill.eval.replay import ReplayToolProvider

    trace = {"id": "t1", "task_id": "t1", "tools": [],
             "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}]}
    outcome = run_task(trace, RecordedTurnClient(trace), ReplayToolProvider(trace))
    assert outcome.escalations == 0 and outcome.wasted_student_tokens == 0


def test_turn_record_serializes():
    assert TurnRecord("student", 0.9, False, 4, 0).to_dict()["arm"] == "student"
