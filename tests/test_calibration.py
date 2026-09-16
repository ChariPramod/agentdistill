"""Judge calibration.

A judge is a measuring instrument. These tests assert it is treated as one: its error rates are measured, its
bias is corrected where the data supports it, and no code path prints a judge number without that context.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agentdistill.eval.calibration import (
    MIN_CALIBRATION_ITEMS,
    JudgeCalibration,
    build_judge_prompt,
    calibrate_judge,
    corrected_rate,
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
def test_correction_recovers_the_true_rate_exactly(seed):
    """Rogan-Gladen is exact in-sample: correcting the judge's own rate must return the truth rate."""
    truth = truth_set(seed=seed)
    cal = calibrate_judge(biased_judge(truth, fpr=0.3, fnr=0.1, seed=seed), truth)
    assert corrected_rate(cal.judge_positive_rate, cal) == pytest.approx(cal.truth_positive_rate, abs=1e-9)


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
