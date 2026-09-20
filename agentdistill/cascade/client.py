"""The cascade as a harness client.

Student first; the teacher only when the gate says the student's turn is not good enough. That is the whole
mechanism, and it sits per *turn* rather than per task: the teacher pays only for hard turns, and a bad tool call
is caught before it executes rather than after the task has failed.

What makes it honest is the accounting. The student generates on every turn, including the ones thrown away, so
those tokens are counted as wasted and reported. A cascade that quietly ignored them would look cheaper than it
is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from agentdistill.cascade.arg_mask import arg_token_mask
from agentdistill.cascade.features import as_vector, turn_features


def score_turn(
    choice: dict,
    extra_samples: list[dict],
    calibrator: Any,
    feature_names: list[str],
    cluster_prior: float,
    turn_idx: int,
    prefix_tokens: int,
) -> float:
    """Probability that this turn is good.

    Shared by the offline cascade client and the gateway, so the gate that is measured is the gate that serves.
    """
    tokens = [t["token"] for t in (choice.get("logprobs") or {}).get("content") or []]
    text = choice.get("text") or "".join(tokens)
    mask = arg_token_mask(tokens, text, choice["message"].get("tool_calls") or [])
    features = turn_features(choice, mask, extra_samples, cluster_prior, turn_idx, prefix_tokens)
    vector = as_vector(features, feature_names)[None, :]
    return float(calibrator.predict_proba(vector)[0, 1])


def prefix_token_estimate(messages: list[dict]) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


def turn_index(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


class SamplingBackend(Protocol):
    def chat(self, messages: list[dict], tools: list[dict], n: int, logprobs: bool) -> list[dict]:
        """Return n OpenAI-style choice dicts: {message, logprobs: {content: [...]}, text}."""
        ...


@dataclass
class TurnRecord:
    arm: str
    p: float
    escalated: bool
    student_tokens: int
    turn_idx: int

    def to_dict(self) -> dict:
        return {
            "arm": self.arm, "p": self.p, "escalated": self.escalated,
            "student_tokens": self.student_tokens, "turn_idx": self.turn_idx,
        }


@dataclass
class CascadeTurnClient:
    """A `TurnClient` that routes each turn through the confidence gate."""

    student: SamplingBackend
    teacher: SamplingBackend
    calibrator: Any
    feature_names: list[str]
    threshold: float
    k_samples: int = 2
    cluster_prior: float = 0.5
    #: When the gate is not usable, escalate everything rather than threshold on meaningless probabilities.
    escalate_everything: bool = False
    log: list[TurnRecord] = field(default_factory=list)

    def reset(self) -> None:
        self.log.clear()

    def score(self, choice: dict, extra: list[dict], turn_idx: int, prefix_tokens: int) -> float:
        return score_turn(
            choice, extra, self.calibrator, self.feature_names, self.cluster_prior, turn_idx, prefix_tokens
        )

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        turn_idx = turn_index(messages)
        prefix_tokens = prefix_token_estimate(messages)

        if self.escalate_everything:
            teacher_choice = self.teacher.chat(messages, tools, n=1, logprobs=False)[0]
            self.log.append(TurnRecord("teacher", float("nan"), True, 0, turn_idx))
            return teacher_choice["message"]

        choices = self.student.chat(messages, tools, n=1 + self.k_samples, logprobs=True)
        primary, extra = choices[0], list(choices[1:])
        student_tokens = len((primary.get("logprobs") or {}).get("content") or [])
        p = self.score(primary, extra, turn_idx, prefix_tokens)

        if p >= self.threshold:
            self.log.append(TurnRecord("student", p, False, student_tokens, turn_idx))
            return primary["message"]

        teacher_choice = self.teacher.chat(messages, tools, n=1, logprobs=False)[0]
        # The student's tokens were generated and thrown away. They are still paid for.
        self.log.append(TurnRecord("teacher", p, True, student_tokens, turn_idx))
        return teacher_choice["message"]

    def summary(self) -> dict:
        n = len(self.log)
        escalated = [r for r in self.log if r.escalated]
        scored = [r.p for r in self.log if not np.isnan(r.p)]
        return {
            "turns": n,
            "escalations": len(escalated),
            "escalation_rate": (len(escalated) / n) if n else 0.0,
            "wasted_student_tokens": sum(r.student_tokens for r in escalated),
            "student_tokens": sum(r.student_tokens for r in self.log),
            "mean_p": float(np.mean(scored)) if scored else float("nan"),
            "threshold": self.threshold,
        }


class FeatureOrderMismatch(ValueError):
    """The runtime cannot build the vector the calibration was fitted on, in the order it was fitted on."""


def scoring_order(
    configured: list[str] | tuple[str, ...],
    report_order: list[str] | None,
    row_order: list[str] | None = None,
) -> list[str]:
    """The feature order to score with, or `FeatureOrderMismatch` saying why there is none.

    The one rule shared by the gateway at boot and the offline cascade, so what is measured is what serves:

    - the stored order is authoritative, because the model expects exactly the columns it was fitted on;
    - it may be a subset of the configured features, because calibration drops columns that had no values;
    - every stored feature must be configured (the runtime cannot produce the others), and the stored features
      must appear in the same relative order as configured. A reordered config means someone edited
      `cascade.features` after calibrating, and the calibration no longer describes the runtime.

    The registry row and calibration.json both record the order; if both are present they must agree, because a
    disagreement means the artifact on disk is not the one the row describes.
    """
    configured_list = list(configured)
    from_row, from_report = list(row_order or []), list(report_order or [])
    if from_row and from_report and from_row != from_report:
        raise FeatureOrderMismatch(
            f"the calibration row records feature order {from_row} but its calibration.json records "
            f"{from_report}; the artifact is not the one the row describes"
        )
    stored = from_row or from_report
    if not stored:
        raise FeatureOrderMismatch("the calibration records no feature order; it cannot be used to score")

    missing = [f for f in stored if f not in configured_list]
    if missing:
        raise FeatureOrderMismatch(
            f"feature order mismatch: the calibration was fitted on {stored} but {missing} "
            f"{'is' if len(missing) == 1 else 'are'} not in the configured cascade.features {configured_list}. "
            f"Recalibrate, or restore the features it was fitted on."
        )
    positions = [configured_list.index(f) for f in stored]
    if positions != sorted(positions):
        in_config_order = [f for f in configured_list if f in stored]
        raise FeatureOrderMismatch(
            f"feature order mismatch: the calibration expects {stored} but cascade.features orders them as "
            f"{in_config_order} (configured: {configured_list}). Recalibrate, or restore the configured order."
        )
    return stored


def from_calibration(
    student: SamplingBackend,
    teacher: SamplingBackend,
    calibration_dir: str,
    configured_features: list[str] | tuple[str, ...],
    threshold: float | None = None,
    k_samples: int = 2,
    cluster_prior: float = 0.5,
    stored_order: list[str] | None = None,
) -> CascadeTurnClient:
    """Build a cascade from a stored calibration, scoring with the order it was fitted on.

    A gate whose calibration is missing or unusable escalates everything. That is the documented default: a
    cascade with a meaningless gate is worse than no cascade, because it escalates the wrong turns while
    reporting a threshold that sounds meaningful.

    A feature-order mismatch raises `FeatureOrderMismatch` rather than escalating everything: an eval run is a
    measurement, and measuring a cascade that is not the one that would serve is worse than not measuring.
    `stored_order` is the registry row's `feature_order`, checked against calibration.json when given.
    """
    from agentdistill.cascade.calibrate import load

    model, report = load(calibration_dir)
    effective = (scoring_order(configured_features, report.get("feature_order"), stored_order)
                 if model is not None else list(configured_features))
    usable = model is not None and report.get("usable", False)
    return CascadeTurnClient(
        student=student,
        teacher=teacher,
        calibrator=model,
        feature_names=effective,
        threshold=threshold if threshold is not None else 1.0,
        k_samples=k_samples,
        cluster_prior=cluster_prior,
        escalate_everything=not usable,
    )


@dataclass
class TurnClientBackend:
    """Adapts an eval `TurnClient` to the `SamplingBackend` the cascade expects.

    The two protocols differ for a reason: an eval client returns one assistant message per call, while the
    cascade needs `n` samples and per-token logprobs to score a turn. The eval clients can produce both, so
    this reshapes what they return rather than introducing a second inference path -- the gate that gets
    measured has to be the gate that serves, and that only holds if both sides see identically shaped input.
    """

    client: Any

    def chat(self, messages: list[dict], tools: list[dict], n: int = 1, logprobs: bool = False) -> list[dict]:
        turn = self.client.next_turn(messages, tools)
        extras = {"logprobs", "samples", "text"}
        core = {k: v for k, v in turn.items() if k not in extras}
        tokens = [t["token"] for t in (turn.get("logprobs") or {}).get("content") or []]

        primary = {
            "message": core,
            "logprobs": turn.get("logprobs"),
            "text": turn.get("text") or "".join(tokens),
        }
        samples = [
            {"message": {k: v for k, v in s.items() if k not in extras}, "logprobs": None, "text": ""}
            for s in (turn.get("samples") or [])
        ]
        return [primary, *samples][: max(n, 1)] or [primary]
