"""Judge calibration.

An LLM judge is a measuring instrument with its own error rate, and that error rate is not symmetric: most
judges are far more willing to call a mediocre trajectory a success than to call a good one a failure. Reporting
a judge's raw agreement as though it were task success hides that bias inside every downstream number.

So the rule, from v0.1 §7: **never report a bare judge number.** A judge score is reported with its agreement
against human or predicate labels, its false-positive and false-negative rates, and — where the calibration set
is large enough — a bias-corrected estimate with an interval that accounts for the judge's own uncertainty.

If no calibration set exists, the report says the number is uncalibrated rather than quietly presenting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Below this many labelled items, a correction is noise. The report says "uncalibrated" instead.
MIN_CALIBRATION_ITEMS = 30

#: Below this, the holdout error's interval is so wide it cannot rule out a useless correction. Warn loudly.
NARROW_CI_ITEMS = 100


@dataclass
class JudgeCalibration:
    """How a judge compares to ground truth on a labelled set.

    `truth` is whatever the project trusts: a predicate outcome, or a human label. The judge is the thing being
    measured, never the reference.
    """

    n: int
    agreement: float
    #: Judge said success, truth said failure. The dangerous direction: it inflates every reported score.
    false_positive_rate: float
    #: Judge said failure, truth said success.
    false_negative_rate: float
    judge_positive_rate: float
    truth_positive_rate: float
    judge_model: str = ""
    rubric: str = ""

    @property
    def usable(self) -> bool:
        return self.n >= MIN_CALIBRATION_ITEMS

    @property
    def bias(self) -> float:
        """How much the judge over-reports success, in rate terms. Positive means it is too generous."""
        return self.judge_positive_rate - self.truth_positive_rate

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "agreement": self.agreement,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "judge_positive_rate": self.judge_positive_rate,
            "truth_positive_rate": self.truth_positive_rate,
            "bias": self.bias,
            "usable": self.usable,
            "judge_model": self.judge_model,
            "rubric": self.rubric,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: str | Path) -> JudgeCalibration:
        d = json.loads(Path(path).read_text())
        return cls(
            n=d["n"],
            agreement=d["agreement"],
            false_positive_rate=d["false_positive_rate"],
            false_negative_rate=d["false_negative_rate"],
            judge_positive_rate=d["judge_positive_rate"],
            truth_positive_rate=d["truth_positive_rate"],
            judge_model=d.get("judge_model", ""),
            rubric=d.get("rubric", ""),
        )


def calibrate_judge(
    judge: list[bool], truth: list[bool], judge_model: str = "", rubric: str = ""
) -> JudgeCalibration:
    """Compare judge verdicts to trusted labels on the same items."""
    if len(judge) != len(truth):
        raise ValueError(f"judge has {len(judge)} verdicts for {len(truth)} labelled items")
    if not judge:
        raise ValueError("no items to calibrate on")
    j = np.array(judge, dtype=bool)
    t = np.array(truth, dtype=bool)
    negatives = int((~t).sum())
    positives = int(t.sum())
    return JudgeCalibration(
        n=len(j),
        agreement=float((j == t).mean()),
        false_positive_rate=float((j & ~t).sum() / negatives) if negatives else 0.0,
        false_negative_rate=float((~j & t).sum() / positives) if positives else 0.0,
        judge_positive_rate=float(j.mean()),
        truth_positive_rate=float(t.mean()),
        judge_model=judge_model,
        rubric=rubric,
    )


def holdout_error(
    judge: list[bool], truth: list[bool], split: float = 0.5, iters: int = 2000, seed: int = 0
) -> dict:
    """How wrong the correction is on data it was not fitted to.

    The in-sample check -- estimate sensitivity and specificity on a set, then correct that same set's rate -- is
    a tautology: Rogan-Gladen inverts exactly, so the error is zero by construction no matter how bad the judge
    is. It says nothing about whether the correction generalizes.

    This fits on one split and applies the correction to a disjoint one, so the reported error reflects the
    sampling noise in the error-rate estimates. The bootstrap resamples the holdout to give that error an
    interval.
    """
    if len(judge) != len(truth):
        raise ValueError(f"judge has {len(judge)} verdicts for {len(truth)} labelled items")
    n = len(judge)
    if n < 2 * MIN_CALIBRATION_ITEMS:
        return {
            "n_fit": 0, "n_holdout": 0, "error": None,
            "note": (
                f"only {n} labelled items; a disjoint split needs at least {2 * MIN_CALIBRATION_ITEMS} "
                f"for either half to support an estimate"
            ),
        }

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(n * split)
    fit_idx, hold_idx = order[:cut], order[cut:]

    j = np.array(judge, dtype=bool)
    t = np.array(truth, dtype=bool)
    fitted = calibrate_judge(j[fit_idx].tolist(), t[fit_idx].tolist())

    observed_holdout = float(j[hold_idx].mean())
    truth_holdout = float(t[hold_idx].mean())
    corrected = corrected_rate(observed_holdout, fitted)
    if corrected is None:
        return {
            "n_fit": len(fit_idx), "n_holdout": len(hold_idx), "error": None,
            "note": "the fitted correction is undefined (the judge is near-random on the fit split)",
        }

    # Bootstrap the holdout to put an interval on the error.
    errors = []
    for _ in range(iters):
        pick = rng.integers(0, len(hold_idx), size=len(hold_idx))
        sub = hold_idx[pick]
        c = corrected_rate(float(j[sub].mean()), fitted)
        if c is not None:
            errors.append(c - float(t[sub].mean()))
    lo, hi = (float(np.percentile(errors, 2.5)), float(np.percentile(errors, 97.5))) if errors else (float("nan"),) * 2

    return {
        "n_fit": len(fit_idx),
        "n_holdout": len(hold_idx),
        "observed": observed_holdout,
        "corrected": corrected,
        "truth": truth_holdout,
        "error": corrected - truth_holdout,
        "error_ci95": [lo, hi],
        "uncorrected_error": observed_holdout - truth_holdout,
        "wide_interval": n < NARROW_CI_ITEMS,
        "note": (
            f"fewer than {NARROW_CI_ITEMS} labelled items, so this interval is wide and the correction is not "
            f"well established"
        ) if n < NARROW_CI_ITEMS else "",
    }


def corrected_rate(observed: float, cal: JudgeCalibration) -> float | None:
    """Rogan-Gladen correction: recover the true rate from a judge's observed rate.

        true = (observed + fnr - 1) / (fnr + tpr - 1)   where tpr = 1 - fnr, and specificity = 1 - fpr

    Returns None when the calibration is too small to trust, or when the judge is so noisy that the correction is
    undefined (sensitivity + specificity <= 1, meaning it carries no more information than a coin).
    """
    if not cal.usable:
        return None
    sensitivity = 1.0 - cal.false_negative_rate
    specificity = 1.0 - cal.false_positive_rate
    denominator = sensitivity + specificity - 1.0
    if denominator <= 1e-6:
        return None
    corrected = (observed + specificity - 1.0) / denominator
    return float(min(max(corrected, 0.0), 1.0))


def report_line(observed: float, cal: JudgeCalibration | None, holdout: dict | None = None) -> str:
    """The one line a judge-graded number is allowed to appear on.

    There is no code path that prints a judge score without this context, which is the point.
    """
    if cal is None:
        return (
            f"judge success {observed:.1%}  [UNCALIBRATED — no labelled set, so this number's error rate is "
            f"unknown and it should not be compared against anything]"
        )
    if not cal.usable:
        return (
            f"judge success {observed:.1%}  [calibration set is only {cal.n} items, below the {MIN_CALIBRATION_ITEMS} "
            f"needed to correct; agreement {cal.agreement:.1%}]"
        )
    corrected = corrected_rate(observed, cal)
    corrected_text = (
        f"corrected {corrected:.1%}" if corrected is not None
        else "correction undefined (the judge is near-random)"
    )
    line = (
        f"judge success {observed:.1%}  ({corrected_text}; agreement {cal.agreement:.1%} on n={cal.n}, "
        f"false-positive {cal.false_positive_rate:.1%}, false-negative {cal.false_negative_rate:.1%}, "
        f"bias {cal.bias:+.1%})"
    )
    if holdout:
        if holdout.get("error") is None:
            line += f"\n  holdout check: {holdout.get('note', 'unavailable')}"
        else:
            lo, hi = holdout["error_ci95"]
            line += (
                f"\n  holdout check: correcting a disjoint split lands {holdout['error']:+.1%} from truth "
                f"[95% CI {lo:+.1%}, {hi:+.1%}] on n={holdout['n_holdout']} "
                f"(uncorrected would be {holdout['uncorrected_error']:+.1%})"
            )
            if holdout.get("wide_interval"):
                line += f"\n  WARNING: {holdout.get('note')}"
    return line


@dataclass
class JudgeResult:
    success: bool
    reasoning: str = ""
    raw: dict = field(default_factory=dict)


def build_judge_prompt(rubric: str, task_input: Any, final_text: str, trajectory_summary: str) -> list[dict]:
    """The judge sees the task, a summary of what the agent did, and the final reply.

    It is shown the *trajectory summary*, not the full transcript: a judge given thousands of tokens of tool
    output tends to grade the tool output rather than the agent's decisions.
    """
    return [
        {
            "role": "system",
            "content": (
                "You grade a customer-support agent's handling of one request. Answer strictly as JSON: "
                '{"success": true|false, "reasoning": "<one sentence>"}. '
                "Judge only against the rubric. Do not reward a polite reply that failed to do the work, and do "
                "not punish a terse reply that did it correctly.\n\nRubric:\n" + rubric
            ),
        },
        {
            "role": "user",
            "content": (
                f"TASK\n{task_input}\n\nWHAT THE AGENT DID\n{trajectory_summary}\n\n"
                f"FINAL REPLY TO THE CUSTOMER\n{final_text}"
            ),
        },
    ]


def trajectory_summary(outcome: Any, limit: int = 20) -> str:
    """Tool calls in order, one per line. Results are omitted deliberately -- see `build_judge_prompt`."""
    from agentdistill.eval.harness import tool_calls_made

    calls = tool_calls_made(outcome)
    lines = [f"{i + 1}. {name}({json.dumps(args, sort_keys=True)})" for i, (name, args) in enumerate(calls[:limit])]
    if len(calls) > limit:
        lines.append(f"... and {len(calls) - limit} more calls")
    if outcome.diverged:
        lines.append("(the run stopped early: the harness had no recorded result for one of these calls)")
    return "\n".join(lines) or "(no tool calls)"


def make_llm_judge(client: Any, rubric: str, judge_model: str = "") -> Any:
    """Build a grader with the runner's signature, backed by a judge model.

    `client` is any `TurnClient`; the judge is just another model call.
    """

    def grade(trace: dict, outcome: Any) -> tuple[bool, dict]:
        messages = build_judge_prompt(
            rubric, trace.get("task_input"), outcome.final_text, trajectory_summary(outcome)
        )
        reply = client.next_turn(messages, [])
        content = (reply.get("content") or "").strip()
        try:
            parsed = json.loads(_strip_fences(content))
            success = bool(parsed["success"])
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, TypeError):
            # A judge that did not answer in the required form has not graded anything. Failing closed would
            # silently count it as a task failure and bias the score downward.
            return False, {"detail": f"judge returned unparseable output: {content[:200]}",
                           "judge_error": True, "judge_model": judge_model}
        return success, {"detail": reasoning, "judge_model": judge_model, "judge_error": False}

    return grade


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        t = t.rsplit("```", 1)[0]
    return t.strip()
