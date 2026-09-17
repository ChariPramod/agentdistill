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
    assert assert_feature_order({"feature_order": FEATURES}, FEATURES) == FEATURES


def test_a_narrower_stored_order_is_allowed_and_wins():
    """Calibration drops features that had no values; the model expects exactly what it was fitted on."""
    narrowed = [f for f in FEATURES if f != "agreement"]
    assert assert_feature_order({"feature_order": narrowed}, FEATURES) == narrowed


def test_a_feature_the_runtime_cannot_produce_is_refused():
    with pytest.raises(ValueError, match="not configured to produce"):
        assert_feature_order({"feature_order": [*FEATURES, "invented_feature"]}, FEATURES)


def test_an_empty_stored_order_is_refused():
    with pytest.raises(ValueError, match="records no feature order"):
        assert_feature_order({"feature_order": []}, FEATURES)


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


# --------------------------------------------------------------------------------------------------------------
# the NaN invariant
#
# Absent argument features must stay absent. Zero is a plausible logprob, so imputing it would teach the gate
# that a turn with no tool call is a confident one.
# --------------------------------------------------------------------------------------------------------------


def test_a_text_only_turn_has_nan_in_every_argument_slot():
    from agentdistill.cascade.features import as_vector, turn_features

    text_choice = choice("just an answer", tool=None)
    features = turn_features(text_choice, [], [], cluster_prior=0.5, turn_idx=0, prefix_tokens=100)
    assert np.isnan(features.arg_mean_logprob)
    assert np.isnan(features.arg_min_logprob)
    assert features.has_tool_call == 0

    vector = as_vector(features, FEATURES)
    for name in ("arg_mean_logprob", "arg_min_logprob"):
        assert np.isnan(vector[FEATURES.index(name)]), f"{name} must stay NaN, not become 0"
    # Everything that is genuinely present is still finite.
    assert np.isfinite(vector[FEATURES.index("mean_logprob")])


def test_a_real_calibrator_scores_a_nan_vector_without_error():
    """HistGradientBoosting handles NaN natively; this pins that the pipeline does too."""
    from agentdistill.cascade.calibrate import fit_calibrator
    from agentdistill.cascade.features import as_vector, matrix, turn_features

    rng = np.random.default_rng(0)
    task_ids, rows, y = [], [], []
    for t in range(40):
        for i in range(5):
            has_tool = (t + i) % 2 == 0
            conf = rng.normal()
            rows.append({
                "mean_logprob": conf, "min_logprob": conf - 1, "p10_logprob": conf - 0.5,
                # Half the turns are text-only, so the fit sees NaN in these columns.
                "arg_mean_logprob": conf if has_tool else float("nan"),
                "arg_min_logprob": conf - 1 if has_tool else float("nan"),
                "first_tool_token_entropy": 0.3, "n_tokens": 20, "n_tool_calls": int(has_tool),
                "has_tool_call": int(has_tool), "agreement": float("nan"), "cluster_prior": 0.5,
                "turn_idx": i, "prefix_tokens": 100,
            })
            task_ids.append(f"task{t}")
            y.append(int(rng.random() < 1 / (1 + np.exp(-conf))))

    result = fit_calibrator(matrix(rows, FEATURES), np.array(y), task_ids, FEATURES)
    assert result.model is not None

    # `agreement` was NaN for every row, so the gate dropped it and says so.
    assert "agreement" not in result.feature_order
    assert any("agreement" in n for n in result.notes)

    text_features = turn_features(choice("answer", tool=None), [], [], 0.5, 0, 100)
    p = result.model.predict_proba(as_vector(text_features, result.feature_order)[None, :])[0, 1]
    assert 0.0 <= p <= 1.0, "a text-only turn must score without imputation"


def test_a_mask_that_does_not_line_up_yields_no_argument_features():
    """A misaligned mask describes different tokens; no features beat wrong ones."""
    from agentdistill.cascade.features import turn_features

    tool_choice = choice("x", tool="refund_order", n_tokens=4)
    features = turn_features(tool_choice, [True, True], [], 0.5, 0, 100)  # mask shorter than the tokens
    assert np.isnan(features.arg_mean_logprob)


# --------------------------------------------------------------------------------------------------------------
# `cascade:<adapter>:<tau>` as an eval subject
#
# The gateway understood this model name; the eval subject resolver did not, so the GPU day's cascade
# verification stage could not run at all. It is what `--verify-threshold` measures against.
# --------------------------------------------------------------------------------------------------------------


class StubTurnClient:
    """An eval client that reports logprobs and extra samples, the way the real ones now do."""

    def __init__(self, content: str = "hi", samples: int = 2) -> None:
        self.content, self.samples = content, samples

    def next_turn(self, messages, tools):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": None,
            "text": self.content,
            "logprobs": {"content": [
                {"token": c, "logprob": -0.1, "top_logprobs": [{"token": c, "logprob": -0.1}]}
                for c in self.content
            ]},
            "samples": [{"role": "assistant", "content": self.content, "tool_calls": None}
                        for _ in range(self.samples)],
        }


def test_a_turn_client_reshapes_into_the_backend_the_cascade_expects():
    from agentdistill.cascade.client import TurnClientBackend

    backend = TurnClientBackend(StubTurnClient())
    choices = backend.chat([{"role": "user", "content": "x"}], [], n=3, logprobs=True)

    assert len(choices) == 3
    # The transport keys must not leak into the message, or the call signature differs from what serving sees.
    assert set(choices[0]) == {"message", "logprobs", "text"}
    assert "samples" not in choices[0]["message"]
    assert "logprobs" not in choices[0]["message"]
    assert choices[0]["logprobs"]["content"], "the primary choice must carry logprobs"


def test_the_backend_returns_at_least_the_primary_choice():
    """A client with no extra samples still has to answer, or the cascade cannot score a turn at all."""
    from agentdistill.cascade.client import TurnClientBackend

    choices = TurnClientBackend(StubTurnClient(samples=0)).chat([], [], n=3)
    assert len(choices) == 1
    assert choices[0]["message"]["content"] == "hi"


def test_a_cascade_subject_without_a_teacher_is_refused(tmp_path):
    """A cascade escalates to the teacher. Without one it would measure the student and call it a cascade."""
    import typer
    import yaml

    from agentdistill.cli import _resolve_client
    from agentdistill.config import ProjectConfig

    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "t",
        "registry": f"sqlite:///{tmp_path}/r.db",
        "artifacts": str(tmp_path / "a"),
        "reports": str(tmp_path / "rep"),
        "dataset": {"max_seq_len": 512},
        "train": {"base_model": "m", "max_seq_len": 512},
    }))
    cfg = ProjectConfig.load(str(tmp_path / "project.yaml"))

    with pytest.raises(typer.Exit):
        _resolve_client("cascade:some-adapter:auto", cfg, backend="hf")


def test_a_malformed_cascade_subject_is_refused(tmp_path):
    import typer
    import yaml

    from agentdistill.cli import _resolve_client
    from agentdistill.config import ProjectConfig

    (tmp_path / "project.yaml").write_text(yaml.safe_dump({
        "name": "t",
        "registry": f"sqlite:///{tmp_path}/r.db",
        "artifacts": str(tmp_path / "a"),
        "reports": str(tmp_path / "rep"),
        "dataset": {"max_seq_len": 512},
        "train": {"base_model": "m", "max_seq_len": 512},
        "teacher": {"model": "anthropic/claude-sonnet-5"},
    }))
    cfg = ProjectConfig.load(str(tmp_path / "project.yaml"))

    # Two segments rather than three.
    with pytest.raises(typer.Exit):
        _resolve_client("cascade:adapter", cfg, backend="hf")


@pytest.mark.parametrize("tau", ["1.5", "-0.1", "auto-ish"])
def test_an_out_of_range_threshold_is_refused(tau):
    import typer

    from agentdistill.cli import _parse_cascade_tau

    with pytest.raises(typer.Exit):
        _parse_cascade_tau(tau)


def test_a_valid_threshold_parses():
    from agentdistill.cli import _parse_cascade_tau

    assert _parse_cascade_tau("0.72") == pytest.approx(0.72)
    assert _parse_cascade_tau("0") == 0.0
    assert _parse_cascade_tau("1") == 1.0
