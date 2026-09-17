"""Paired statistics for eval comparisons.

Every claim this project makes is a *paired* comparison: the same tasks, run under two subjects. Pairing is what
makes a small eval set usable — task difficulty varies enormously, and comparing two independent samples would
drown the effect in that variance.

Two mistakes this module exists to prevent:

1. **Treating repeats as independent observations.** Running 40 tasks 5 times is 200 rows but nowhere near 200
   independent samples: repeats of one task are highly correlated. Intervals are computed by resampling *tasks*,
   which is the unit that was actually sampled.
2. **Reporting a p-value per metric without correction.** Testing success, tokens, and turns and announcing
   whichever cleared 0.05 is how noise becomes a finding. Holm-Bonferroni is applied across the family.

Shared with agentreplay: the same rules must produce the same numbers in both, or two reports of the same
experiment will disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_ITERS = 10_000


class TooFewTasks(ValueError):
    """The comparison cannot support an interval. Refusing is better than printing a meaningless one."""


def minimum_n_guard(n_tasks: int, n_per_task: int, min_tasks: int = 8, min_repeats: int = 1) -> None:
    """Refuse comparisons too small to say anything.

    A single repeat per task is allowed — it is the honest N=1 case — but fewer than a handful of *tasks* makes
    a bootstrap interval meaningless, because there is nothing to resample.
    """
    if n_tasks < min_tasks:
        raise TooFewTasks(
            f"only {n_tasks} tasks overlap between the two runs; at least {min_tasks} are needed for a "
            f"task-clustered interval to mean anything. Run the two subjects on the same eval set."
        )
    if n_per_task < min_repeats:
        raise TooFewTasks(f"n_per_task is {n_per_task}; at least {min_repeats} is required")


def _paired_task_means(
    a: dict[str, list[float]], b: dict[str, list[float]]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-task means for the tasks both runs share, in a stable order."""
    tasks = sorted(set(a) & set(b))
    if not tasks:
        raise TooFewTasks("the two runs share no tasks; they cannot be paired")
    va = np.array([float(np.mean(a[t])) for t in tasks])
    vb = np.array([float(np.mean(b[t])) for t in tasks])
    return tasks, va, vb


@dataclass
class Comparison:
    """A paired difference with its interval. `a - b`, so positive favours `a`."""

    n_tasks: int
    mean_a: float
    mean_b: float
    delta: float
    ci95: tuple[float, float]
    iters: int

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero."""
        lo, hi = self.ci95
        return not (lo <= 0.0 <= hi)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "ci95": list(self.ci95),
            "excludes_zero": self.significant,
            "iters": self.iters,
        }


def cluster_bootstrap_diff(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
    iters: int = DEFAULT_ITERS,
    seed: int = 0,
) -> Comparison:
    """Paired bootstrap over tasks.

    Tasks are resampled with replacement and both subjects' means recomputed on the same resample, which keeps
    the pairing intact. Resampling rows instead would break it and inflate the interval.
    """
    tasks, va, vb = _paired_task_means(a, b)
    rng = np.random.default_rng(seed)
    diff = va - vb
    if len(tasks) < 2:
        return Comparison(len(tasks), float(va.mean()), float(vb.mean()), float(diff.mean()),
                          (float("nan"), float("nan")), iters)
    idx = rng.integers(0, len(tasks), size=(iters, len(tasks)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Comparison(
        n_tasks=len(tasks),
        mean_a=float(va.mean()),
        mean_b=float(vb.mean()),
        delta=float(diff.mean()),
        ci95=(float(lo), float(hi)),
        iters=iters,
    )


def mcnemar_paired(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict:
    """Exact McNemar on task-level outcomes.

    A task counts as a success for a subject when it succeeded in the majority of its repeats. Only discordant
    tasks — one succeeded, the other did not — carry information; the exact binomial avoids the chi-square
    approximation, which is unreliable at the discordant counts an eval set of this size produces.
    """
    _tasks, va, vb = _paired_task_means(a, b)
    sa, sb = va > 0.5, vb > 0.5
    a_only = int(np.sum(sa & ~sb))
    b_only = int(np.sum(~sa & sb))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p": 1.0,
                "note": "no task changed outcome between the two subjects"}
    # Two-sided exact binomial at p=0.5.
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    p = min(1.0, 2 * tail)
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p": float(p)}


def wilcoxon_metric(a: dict[str, float], b: dict[str, float]) -> dict:
    """Wilcoxon signed-rank over per-task values, for continuous metrics like tokens and turns.

    Non-parametric on purpose: token counts are heavily skewed, so a t-test's normality assumption does not hold.
    """
    tasks = sorted(set(a) & set(b))
    if not tasks:
        raise TooFewTasks("the two runs share no tasks")
    va = np.array([a[t] for t in tasks], dtype=float)
    vb = np.array([b[t] for t in tasks], dtype=float)
    diff = va - vb
    nonzero = diff[diff != 0]
    if len(nonzero) < 1:
        return {"n_tasks": len(tasks), "median_a": float(np.median(va)), "median_b": float(np.median(vb)),
                "median_delta": 0.0, "ci95": [0.0, 0.0], "p": 1.0, "note": "identical on every task"}
    from scipy.stats import wilcoxon

    try:
        stat, p = wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return {"n_tasks": len(tasks), "median_a": float(np.median(va)), "median_b": float(np.median(vb)),
                "median_delta": float(np.median(diff)), "ci95": list(_median_diff_ci(diff)), "p": 1.0,
                "note": "too few nonzero differences"}
    rel = float(np.median(diff) / np.median(vb)) if np.median(vb) else float("nan")
    return {
        "n_tasks": len(tasks),
        "median_a": float(np.median(va)),
        "median_b": float(np.median(vb)),
        "median_delta": float(np.median(diff)),
        "ci95": list(_median_diff_ci(diff)),
        "relative_delta": rel,
        "statistic": float(stat),
        "p": float(p),
    }


def _median_diff_ci(diff: np.ndarray, iters: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap interval on the median per-task difference.

    A promotion that rests on a point estimate promotes on noise. The interval is what lets a caller ask whether
    the saving is real rather than whether it happened to be negative this run.
    """
    if len(diff) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(iters, len(diff)))
    boots = np.median(diff[idx], axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction.

    Uniformly more powerful than plain Bonferroni and makes no independence assumption, which matters here
    because the metrics in one report are correlated (a student that takes more turns emits more tokens).
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(ordered):
        adjusted = min(1.0, (m - i) * p)
        # Step-down: adjusted p-values must be monotonically non-decreasing.
        running_max = max(running_max, adjusted)
        out[name] = {"p": float(p), "p_adjusted": float(running_max), "significant": running_max <= alpha}
    return out


def success_by_task(rows: list[Any], attr: str = "success") -> dict[str, list[float]]:
    """Group a run's rows into {task_id: [outcome per repeat]}."""
    out: dict[str, list[float]] = {}
    for r in rows:
        value = r[attr] if isinstance(r, dict) else getattr(r, attr)
        task = r["task_id"] if isinstance(r, dict) else r.task_id
        out.setdefault(task, []).append(float(bool(value)) if isinstance(value, bool) else float(value or 0.0))
    return out


def metric_by_task(rows: list[Any], attr: str) -> dict[str, float]:
    """Mean of a numeric metric per task."""
    grouped = success_by_task(rows, attr)
    return {t: float(np.mean(v)) for t, v in grouped.items()}
