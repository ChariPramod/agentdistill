"""Choosing the escalation threshold.

Grid-search a per-turn threshold against a success-drop budget, then verify the chosen point with the harness.
The analytic pass is for speed; the harness number is what gets reported, because the analytic model makes an
assumption that is not quite true.

The assumption: an escalated turn is as good as the teacher's. It is not. The teacher is answering on a prefix
the *student* built, which may already contain a mistake the teacher would never have made. So the analytic
estimate is optimistic by construction, and the verified points are the honest ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThresholdPoint:
    threshold: float
    cascade_success: float
    escalation_rate: float
    cost_per_turn: float
    wasted_student_tokens: float = 0.0

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "cascade_success": self.cascade_success,
            "escalation_rate": self.escalation_rate,
            "cost_per_turn": self.cost_per_turn,
            "wasted_student_tokens": self.wasted_student_tokens,
        }


@dataclass
class ThresholdChoice:
    chosen: ThresholdPoint | None
    teacher_success: float
    curve: list[ThresholdPoint] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "teacher_success": self.teacher_success,
            "curve": [p.to_dict() for p in self.curve],
            "note": self.note,
        }


def choose_threshold(
    p_turn: np.ndarray,
    good_turn: np.ndarray,
    task_of_turn: np.ndarray,
    teacher_task_success: dict[str, bool],
    cost_student_turn: float = 1.0,
    cost_teacher_turn: float = 20.0,
    max_success_drop_pp: float = 1.0,
    mean_student_tokens: float = 0.0,
    grid: np.ndarray | None = None,
) -> ThresholdChoice:
    """The cheapest threshold whose estimated success stays within the drop budget.

    A task succeeds under the cascade if every turn the student kept was good, and, where anything escalated, the
    teacher would have succeeded on that task. That second clause is the optimistic assumption above.
    """
    grid = grid if grid is not None else np.linspace(0.05, 0.99, 95)
    tasks = np.unique(task_of_turn)
    if len(tasks) == 0:
        return ThresholdChoice(None, float("nan"), [], "no turns to choose a threshold from")

    teacher_rate = float(np.mean([bool(teacher_task_success.get(str(t), False)) for t in tasks]))
    curve: list[ThresholdPoint] = []
    best: ThresholdPoint | None = None

    for tau in grid:
        keep = p_turn >= tau
        ok = []
        for t in tasks:
            in_task = task_of_turn == t
            kept = keep[in_task]
            kept_good = bool(np.all(good_turn[in_task][kept])) if kept.any() else True
            any_escalated = not bool(kept.all())
            ok.append(kept_good and (bool(teacher_task_success.get(str(t), False)) if any_escalated else True))
        rate = float(np.mean(ok))
        escalation = float(1.0 - keep.mean())
        # The student generates on every turn, including the ones that get thrown away, so its cost is paid in
        # full regardless of the escalation rate.
        cost = cost_student_turn + escalation * cost_teacher_turn
        point = ThresholdPoint(
            threshold=float(tau),
            cascade_success=rate,
            escalation_rate=escalation,
            cost_per_turn=cost,
            wasted_student_tokens=escalation * mean_student_tokens,
        )
        curve.append(point)
        if teacher_rate - rate <= max_success_drop_pp / 100 and (best is None or cost < best.cost_per_turn):
            best = point

    if best is None:
        return ThresholdChoice(
            None, teacher_rate, curve,
            f"no threshold keeps the success drop within {max_success_drop_pp} pp; escalate everything and say so",
        )
    return ThresholdChoice(best, teacher_rate, curve, "")


def verification_points(tau: float, spread: float = 0.05) -> list[float]:
    """The thresholds to measure with the harness: the chosen one and a neighbour either side.

    Three points, because a single measurement cannot show whether the choice sits on a cliff.
    """
    return [round(max(0.01, tau - spread), 4), round(tau, 4), round(min(0.99, tau + spread), 4)]


def pick_verified(verified: list[dict], max_success_drop_pp: float, teacher_success: float) -> dict | None:
    """Choose among harness-measured points. These are the numbers that get reported.

    Preference is for the cheapest point still inside the budget; if none is, the one with the highest measured
    success, so the report can say plainly that the budget could not be met.
    """
    if not verified:
        return None
    within = [
        v for v in verified
        if (teacher_success - v.get("success", 0.0)) * 100 <= max_success_drop_pp
    ]
    if within:
        return min(within, key=lambda v: v.get("cost_per_task", float("inf")))
    return max(verified, key=lambda v: v.get("success", 0.0))
