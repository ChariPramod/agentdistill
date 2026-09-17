"""Judge calibration.

A judge is a measuring instrument. These tests assert it is treated as one: its error rates are measured, its
bias is corrected where the data supports it, and no code path prints a judge number without that context.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from agentdistill.eval.calibration import (
    MIN_CALIBRATION_ITEMS,
    JudgeCalibration,
    build_judge_prompt,
    calibrate_judge,
    corrected_rate,
    holdout_error,
    make_llm_judge,
    report_line,
    trajectory_summary,
)


def biased_judge(truth: list[bool], fpr: float, fnr: float, seed: int = 0) -> list[bool]:
    """A judge with known error rates, so a correction can be checked against ground truth.

    The seed is offset from the one that built `truth`: drawing from an identically seeded generator replays the
    same sequence, which makes the judge's errors perfectly anti-correlated with the labels and produces a
    flawless judge no matter what error rates are asked for.
    """
    rng = np.random.default_rng(seed + 10_000)
    return [(rng.random() > fnr) if t else (rng.random() < fpr) for t in truth]


def truth_set(n: int = 400, rate: float = 0.7, seed: int = 0) -> list[bool]:
    rng = np.random.default_rng(seed)
    return [bool(rng.random() < rate) for _ in range(n)]


# --------------------------------------------------------------------------------------------------------------
# measuring the judge
# --------------------------------------------------------------------------------------------------------------


def test_perfect_judge_has_no_error():
    truth = truth_set()
    cal = calibrate_judge(truth, truth)
    assert cal.agreement == 1.0
    assert cal.false_positive_rate == 0.0 and cal.false_negative_rate == 0.0
    assert cal.bias == 0.0


def test_generous_judge_is_detected_as_biased():
    """The dangerous direction: a judge that calls failures successes inflates every reported score."""
    truth = truth_set()
    cal = calibrate_judge(biased_judge(truth, fpr=0.4, fnr=0.0), truth)
    assert cal.false_positive_rate > 0.25
    assert cal.false_negative_rate == 0.0
    assert cal.bias > 0.05, "a generous judge reports more successes than exist"


def test_harsh_judge_is_detected():
    truth = truth_set()
    cal = calibrate_judge(biased_judge(truth, fpr=0.0, fnr=0.3), truth)
    assert cal.false_negative_rate > 0.2
    assert cal.bias < 0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="verdicts for"):
        calibrate_judge([True, False], [True])


def test_empty_calibration_is_rejected():
    with pytest.raises(ValueError, match="no items"):
        calibrate_judge([], [])


def test_all_positive_truth_gives_no_false_positive_rate():
    """With no negatives there is nothing to be falsely positive about; it must not divide by zero."""
    cal = calibrate_judge([True, True], [True, True])
    assert cal.false_positive_rate == 0.0


# --------------------------------------------------------------------------------------------------------------
# correcting the bias
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_correction_inverts_exactly_in_sample(seed):
    """Rogan-Gladen inverts exactly when applied to the set its rates were estimated on.

    This is arithmetic, not evidence: the error is zero by construction however bad the judge is. It is kept as
    a sanity check that the formula is implemented right. `test_correction_on_a_disjoint_split` is the test that
    says anything about whether the correction generalizes.
    """
    truth = truth_set(seed=seed)
    cal = calibrate_judge(biased_judge(truth, fpr=0.3, fnr=0.1, seed=seed), truth)
    assert corrected_rate(cal.judge_positive_rate, cal) == pytest.approx(cal.truth_positive_rate, abs=1e-9)


def test_correction_on_a_disjoint_split_beats_no_correction():
    """The claim that matters: fitted on one split, applied to another, it is closer to truth than the raw rate."""
    truth = truth_set(n=1200, seed=3)
    judge = biased_judge(truth, fpr=0.35, fnr=0.05, seed=3)
    h = holdout_error(judge, truth, iters=800)
    assert h["error"] is not None
    assert abs(h["error"]) < abs(h["uncorrected_error"]), (
        f"corrected error {h['error']:+.3f} is no better than uncorrected {h['uncorrected_error']:+.3f}"
    )


def test_holdout_error_reports_an_interval():
    truth = truth_set(n=600, seed=4)
    h = holdout_error(biased_judge(truth, fpr=0.3, fnr=0.1, seed=4), truth, iters=500)
    lo, hi = h["error_ci95"]
    assert lo < hi
    assert h["n_fit"] + h["n_holdout"] == 600
    assert not h["wide_interval"]


def test_holdout_error_warns_on_a_small_labelled_set():
    truth = truth_set(n=80, seed=5)
    h = holdout_error(biased_judge(truth, fpr=0.3, fnr=0.1, seed=5), truth, iters=200)
    assert h["wide_interval"]
    assert "not well established" in h["note"]


def test_holdout_error_refuses_a_set_too_small_to_split():
    truth = truth_set(n=20, seed=6)
    h = holdout_error(biased_judge(truth, 0.3, 0.1, seed=6), truth)
    assert h["error"] is None
    assert "disjoint split needs" in h["note"]


def test_holdout_error_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="verdicts for"):
        holdout_error([True], [True, False])


def test_report_line_includes_the_holdout_check():
    truth = truth_set(n=600, seed=7)
    judge = biased_judge(truth, fpr=0.3, fnr=0.1, seed=7)
    cal = calibrate_judge(judge, truth)
    line = report_line(cal.judge_positive_rate, cal, holdout_error(judge, truth, iters=300))
    assert "holdout check" in line
    assert "uncorrected would be" in line, "the reader must be able to see whether correcting helped"


def test_correction_moves_a_generous_judge_downward():
    truth = truth_set()
    cal = calibrate_judge(biased_judge(truth, fpr=0.4, fnr=0.0), truth)
    assert corrected_rate(0.9, cal) < 0.9


def test_correction_is_refused_on_a_small_set():
    """Below the threshold a correction is noise dressed up as precision."""
    truth = truth_set(n=MIN_CALIBRATION_ITEMS - 1)
    cal = calibrate_judge(biased_judge(truth, fpr=0.2, fnr=0.1), truth)
    assert not cal.usable
    assert corrected_rate(0.8, cal) is None


def test_correction_is_refused_for_a_near_random_judge():
    """Sensitivity + specificity <= 1 means the judge carries no information; the correction is undefined."""
    cal = JudgeCalibration(n=200, agreement=0.5, false_positive_rate=0.5, false_negative_rate=0.5,
                           judge_positive_rate=0.5, truth_positive_rate=0.5)
    assert corrected_rate(0.8, cal) is None


def test_correction_is_clamped_to_a_valid_rate():
    cal = JudgeCalibration(n=200, agreement=0.8, false_positive_rate=0.3, false_negative_rate=0.05,
                           judge_positive_rate=0.7, truth_positive_rate=0.6)
    assert 0.0 <= corrected_rate(0.99, cal) <= 1.0
    assert 0.0 <= corrected_rate(0.01, cal) <= 1.0


# --------------------------------------------------------------------------------------------------------------
# never a bare number
# --------------------------------------------------------------------------------------------------------------


def test_uncalibrated_judge_number_is_labelled_as_such():
    line = report_line(0.82, None)
    assert "82.0%" in line
    assert "UNCALIBRATED" in line
    assert "should not be compared" in line


def test_small_calibration_is_labelled():
    truth = truth_set(n=10)
    cal = calibrate_judge(biased_judge(truth, 0.2, 0.1), truth)
    line = report_line(0.8, cal)
    assert "below the" in line and str(MIN_CALIBRATION_ITEMS) in line


def test_calibrated_line_carries_every_error_rate():
    truth = truth_set()
    cal = calibrate_judge(biased_judge(truth, fpr=0.3, fnr=0.1), truth)
    line = report_line(cal.judge_positive_rate, cal)
    for needle in ["corrected", "agreement", "false-positive", "false-negative", "bias"]:
        assert needle in line, f"the judge line must carry {needle!r}"


def test_near_random_judge_line_says_correction_is_undefined():
    cal = JudgeCalibration(n=200, agreement=0.5, false_positive_rate=0.5, false_negative_rate=0.5,
                           judge_positive_rate=0.5, truth_positive_rate=0.5)
    assert "undefined" in report_line(0.8, cal)


def test_calibration_round_trips_through_disk(tmp_path):
    truth = truth_set()
    cal = calibrate_judge(biased_judge(truth, 0.2, 0.1), truth, judge_model="m", rubric="r.md")
    loaded = JudgeCalibration.load(cal.save(tmp_path / "cal.json"))
    assert loaded.to_dict() == cal.to_dict()


# --------------------------------------------------------------------------------------------------------------
# the judge grader
# --------------------------------------------------------------------------------------------------------------


class StubJudge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[dict] = []

    def next_turn(self, messages, tools):
        self.seen = messages
        return {"role": "assistant", "content": self.reply, "tool_calls": None}


class FakeOutcome:
    def __init__(self, final_text="done", diverged=False):
        self.final_text = final_text
        self.diverged = diverged
        self.messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "issue_refund", "arguments": '{"order_id": "o_1"}'}}]}
        ]


def test_judge_parses_a_verdict():
    grade = make_llm_judge(StubJudge('{"success": true, "reasoning": "refund issued"}'), "rubric", "judge-1")
    success, detail = grade({"task_input": "t"}, FakeOutcome())
    assert success is True
    assert detail["detail"] == "refund issued"
    assert detail["judge_error"] is False


def test_judge_handles_fenced_json():
    grade = make_llm_judge(StubJudge('```json\n{"success": false, "reasoning": "no refund"}\n```'), "r")
    success, detail = grade({"task_input": "t"}, FakeOutcome())
    assert success is False and detail["detail"] == "no refund"


def test_unparseable_judge_output_is_flagged_not_counted_as_failure():
    """Failing closed would bias the score downward and hide that the judge broke."""
    grade = make_llm_judge(StubJudge("I think it went quite well overall!"), "r")
    success, detail = grade({"task_input": "t"}, FakeOutcome())
    assert success is False
    assert detail["judge_error"] is True, "a broken judge must be distinguishable from a failed task"
    assert "unparseable" in detail["detail"]


def test_judge_prompt_contains_task_actions_and_reply():
    stub = StubJudge('{"success": true}')
    make_llm_judge(stub, "THE RUBRIC")({"task_input": "refund my order"}, FakeOutcome("I refunded it"))
    text = json.dumps(stub.seen)
    assert "THE RUBRIC" in text
    assert "refund my order" in text
    assert "I refunded it" in text
    assert "issue_refund" in text


def test_trajectory_summary_omits_tool_results():
    """A judge shown raw tool output grades the output rather than the agent's decisions."""
    summary = trajectory_summary(FakeOutcome())
    assert "issue_refund" in summary
    assert "o_1" in summary
    assert "role" not in summary


def test_trajectory_summary_notes_divergence():
    assert "stopped early" in trajectory_summary(FakeOutcome(diverged=True))


def test_trajectory_summary_with_no_calls():
    class Empty:
        messages = []
        diverged = False

    assert "no tool calls" in trajectory_summary(Empty())


def test_build_judge_prompt_demands_json():
    messages = build_judge_prompt("r", "task", "final", "summary")
    assert messages[0]["role"] == "system"
    assert '"success"' in messages[0]["content"]
    assert "polite reply that failed" in messages[0]["content"], "the judge must be warned about style bias"


# --------------------------------------------------------------------------------------------------------------
# from a stored eval turn to a feature vector
#
# This path had no test and did not work: `calibrate` read `record["features"]`, which `label_rollouts` never
# set, so the command could not fit a gate however good its input was. The CPU rehearsal found it.
# --------------------------------------------------------------------------------------------------------------


def _stored_turn(with_logprobs: bool = True, with_samples: bool = True) -> dict:
    """An assistant message shaped the way the eval harness stores it."""
    call = {"id": "c1", "type": "function",
            "function": {"name": "issue_refund", "arguments": '{"order_id": "A1"}'}}
    message: dict = {"role": "assistant", "content": None, "tool_calls": [call]}
    if with_logprobs:
        text = '{"order_id": "A1"}'
        message["logprobs"] = {"content": [
            {"token": ch, "logprob": -0.1, "top_logprobs": [{"token": ch, "logprob": -0.1}]}
            for ch in text
        ]}
        message["text"] = text
    if with_samples:
        message["samples"] = [{"role": "assistant", "content": None, "tool_calls": [call]}]
    return message


def test_a_stored_turn_becomes_a_scorable_choice():
    from agentdistill.cascade.labels import as_choice

    choice, samples = as_choice(_stored_turn())
    assert set(choice) == {"message", "logprobs", "text"}
    # The extras must not leak into the message, or the call signature they feed would differ from serving.
    assert "logprobs" not in choice["message"]
    assert "samples" not in choice["message"]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "issue_refund"
    assert len(samples) == 1 and "message" in samples[0]


def test_features_are_computed_from_a_stored_turn():
    from agentdistill.cascade.labels import features_for

    features = features_for(_stored_turn(), cluster_prior=0.7, turn_idx=2, prefix_tokens=400)
    assert features.has_tool_call == 1
    assert features.n_tool_calls == 1
    assert features.cluster_prior == pytest.approx(0.7)
    assert features.turn_idx == 2
    assert features.prefix_tokens == 400
    assert not math.isnan(features.mean_logprob)
    # The sample repeated the same call, so agreement is 1.0 rather than missing.
    assert features.agreement == pytest.approx(1.0)


def test_a_disagreeing_sample_lowers_agreement():
    from agentdistill.cascade.labels import features_for

    turn = _stored_turn()
    turn["samples"] = [{"role": "assistant", "content": "I am not sure.", "tool_calls": None}]
    assert features_for(turn).agreement == pytest.approx(0.0)


def test_attach_features_skips_turns_without_logprobs():
    """Dropped rather than filled with NaN: an entirely missing row teaches the calibrator nothing and
    dilutes every metric computed over it."""
    from agentdistill.cascade.labels import attach_features

    records = [
        {"task_id": "t1", "turn_index": 0, "good": True, "message": _stored_turn()},
        {"task_id": "t2", "turn_index": 0, "good": False, "message": _stored_turn(with_logprobs=False)},
    ]
    assert attach_features(records) == 1
    assert records[0].get("features") is not None
    assert records[1].get("features") is None


def test_label_rollouts_carries_the_prefix_each_turn_was_generated_from():
    """`prefix_tokens` is a feature, so a turn's position in a long trajectory has to survive labelling."""
    from agentdistill.cascade.labels import label_rollouts

    messages = [
        {"role": "user", "content": "refund order A1"},
        _stored_turn(),
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"role": "assistant", "content": "Done."},
    ]
    records, _ = label_rollouts([{"id": "r1", "task_id": "t1", "success": True, "messages": messages}], {})
    assert records
    assert all("prefix" in r for r in records)
    first = next(r for r in records if r["turn_index"] == 1)
    assert first["prefix"] == messages[:1]


def test_the_whole_path_produces_a_matrix_the_calibrator_can_fit():
    """End to end: stored turns in, a finite feature matrix out, in the configured column order."""
    import numpy as np

    from agentdistill.cascade.features import DEFAULT_FEATURES, matrix
    from agentdistill.cascade.labels import attach_features, label_rollouts

    rollouts = []
    for i in range(12):
        messages = [{"role": "user", "content": f"task {i}"}, _stored_turn()]
        rollouts.append({"id": f"r{i}", "task_id": f"t{i}", "success": i % 2 == 0, "messages": messages})

    records, stats = label_rollouts(rollouts, {})
    assert attach_features(records) == len(records)

    X = matrix([r["features"] for r in records], list(DEFAULT_FEATURES))
    assert X.shape == (len(records), len(DEFAULT_FEATURES))
    # Some columns are legitimately NaN (no text tokens here); the point is that the matrix builds at all.
    assert np.isfinite(X).any()
    assert stats["n"] == len(records)
