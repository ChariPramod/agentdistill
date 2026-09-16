"""Paired statistics, checked by simulation.

Unit-testing a statistical method against hand-computed numbers proves it matches a formula. Simulating from a
known ground truth proves the numbers mean what the report says they mean: that a 95% interval covers the truth
about 95% of the time, that a real effect is detected, and that correction holds the false-positive rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentdistill.eval.stats import (
    TooFewTasks,
    cluster_bootstrap_diff,
    holm,
    mcnemar_paired,
    metric_by_task,
    minimum_n_guard,
    success_by_task,
    wilcoxon_metric,
)


def simulate(
    n_tasks: int, n_repeats: int, p_a: float, p_b: float, rng: np.random.Generator, task_spread: float = 0.0
) -> tuple[dict, dict]:
    """Two runs over the same tasks.

    `task_spread` adds per-task difficulty shared by both subjects, which is what makes repeats correlated and
    pairing worth doing.
    """
    a: dict[str, list[float]] = {}
    b: dict[str, list[float]] = {}
    for i in range(n_tasks):
        shift = rng.normal(0, task_spread)
        pa = min(max(p_a + shift, 0.01), 0.99)
        pb = min(max(p_b + shift, 0.01), 0.99)
        a[f"t{i}"] = [float(rng.random() < pa) for _ in range(n_repeats)]
        b[f"t{i}"] = [float(rng.random() < pb) for _ in range(n_repeats)]
    return a, b


# --------------------------------------------------------------------------------------------------------------
# bootstrap: coverage
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_ci_covers_the_true_delta_about_95_percent_of_the_time():
    """The property that makes a confidence interval worth printing."""
    rng = np.random.default_rng(7)
    true_delta = 0.15
    covered = 0
    trials = 300
    for _ in range(trials):
        a, b = simulate(40, 3, 0.75, 0.75 - true_delta, rng, task_spread=0.12)
        c = cluster_bootstrap_diff(a, b, iters=600, seed=int(rng.integers(1e6)))
        lo, hi = c.ci95
        covered += lo <= true_delta <= hi
    rate = covered / trials
    assert 0.88 <= rate <= 0.99, f"95% interval covered the truth {rate:.0%} of the time"


def test_ci_excludes_zero_for_a_large_effect():
    rng = np.random.default_rng(1)
    a, b = simulate(40, 5, 0.9, 0.5, rng)
    c = cluster_bootstrap_diff(a, b, iters=2000)
    assert c.significant and c.delta > 0.25


def test_ci_includes_zero_when_there_is_no_effect():
    rng = np.random.default_rng(2)
    a, b = simulate(40, 5, 0.7, 0.7, rng)
    assert not cluster_bootstrap_diff(a, b, iters=2000).significant


def test_pairing_is_preserved_across_the_resample():
    """If pairing broke, shared task difficulty would inflate the interval enormously."""
    rng = np.random.default_rng(3)
    a, b = simulate(40, 5, 0.8, 0.6, rng, task_spread=0.25)
    c = cluster_bootstrap_diff(a, b, iters=3000)
    width = c.ci95[1] - c.ci95[0]
    assert width < 0.35, f"interval is {width:.2f} wide; pairing looks broken"
    assert c.significant


def test_delta_sign_favours_a():
    rng = np.random.default_rng(4)
    a, b = simulate(20, 5, 0.9, 0.4, rng)
    assert cluster_bootstrap_diff(a, b, iters=1000).delta > 0
    assert cluster_bootstrap_diff(b, a, iters=1000).delta < 0


def test_bootstrap_is_deterministic_under_a_seed():
    rng = np.random.default_rng(5)
    a, b = simulate(20, 3, 0.8, 0.6, rng)
    assert cluster_bootstrap_diff(a, b, seed=42).ci95 == cluster_bootstrap_diff(a, b, seed=42).ci95


def test_only_shared_tasks_are_compared():
    a = {"t1": [1.0], "t2": [1.0], "extra": [0.0]}
    b = {"t1": [0.0], "t2": [0.0]}
    assert cluster_bootstrap_diff(a, b, iters=200).n_tasks == 2


def test_no_shared_tasks_is_refused():
    with pytest.raises(TooFewTasks, match="share no tasks"):
        cluster_bootstrap_diff({"a": [1.0]}, {"b": [1.0]})


# --------------------------------------------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------------------------------------------


def test_mcnemar_detects_a_consistent_difference():
    rng = np.random.default_rng(11)
    a, b = simulate(40, 5, 0.9, 0.5, rng)
    result = mcnemar_paired(a, b)
    assert result["p"] < 0.01
    assert result["a_only"] > result["b_only"]


def test_mcnemar_reports_no_difference_when_outcomes_match():
    a = {f"t{i}": [1.0] for i in range(10)}
    result = mcnemar_paired(a, dict(a))
    assert result["p"] == 1.0 and result["n_discordant"] == 0
    assert "no task changed outcome" in result["note"]


def test_mcnemar_uses_only_discordant_tasks():
    """Tasks both subjects got right carry no information about which is better."""
    a = {f"t{i}": [1.0] for i in range(20)}
    b = dict(a)
    for i in range(5):
        b[f"t{i}"] = [0.0]
    result = mcnemar_paired(a, b)
    assert result["n_discordant"] == 5 and result["a_only"] == 5 and result["b_only"] == 0


@pytest.mark.slow
def test_mcnemar_false_positive_rate_is_near_alpha():
    rng = np.random.default_rng(13)
    false_positives = 0
    trials = 300
    for _ in range(trials):
        a, b = simulate(30, 3, 0.7, 0.7, rng, task_spread=0.1)
        false_positives += mcnemar_paired(a, b)["p"] < 0.05
    rate = false_positives / trials
    assert rate <= 0.10, f"false-positive rate {rate:.0%} at alpha=0.05"


# --------------------------------------------------------------------------------------------------------------
# Wilcoxon
# --------------------------------------------------------------------------------------------------------------


def test_wilcoxon_detects_a_token_reduction():
    rng = np.random.default_rng(17)
    a = {f"t{i}": 100 + rng.normal(0, 10) for i in range(30)}
    b = {f"t{i}": 160 + rng.normal(0, 10) for i in range(30)}
    result = wilcoxon_metric(a, b)
    assert result["p"] < 0.001
    assert result["median_delta"] < 0
    assert -0.5 < result["relative_delta"] < -0.25


def test_wilcoxon_on_identical_values():
    a = {f"t{i}": 10.0 for i in range(10)}
    result = wilcoxon_metric(a, dict(a))
    assert result["p"] == 1.0 and result["median_delta"] == 0.0


def test_wilcoxon_is_robust_to_a_skewed_outlier():
    """Token counts are skewed; a mean-based test would be dominated by one runaway trajectory."""
    a = {f"t{i}": 100.0 for i in range(20)}
    b = {f"t{i}": 110.0 for i in range(20)}
    b["t0"] = 100_000.0
    result = wilcoxon_metric(a, b)
    assert result["median_delta"] == -10.0, "the outlier must not move the median"
    assert result["p"] < 0.05


# --------------------------------------------------------------------------------------------------------------
# Holm
# --------------------------------------------------------------------------------------------------------------


def test_holm_adjusts_upward_and_preserves_order():
    out = holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert out["a"]["p_adjusted"] >= 0.01
    assert out["a"]["p_adjusted"] <= out["b"]["p_adjusted"] <= out["c"]["p_adjusted"]


def test_holm_keeps_a_strong_result_significant():
    out = holm({"success": 1e-6, "tokens": 0.4, "turns": 0.9})
    assert out["success"]["significant"]
    assert not out["tokens"]["significant"]


def test_holm_suppresses_a_marginal_result_in_a_family():
    """0.04 alone would pass; tested alongside two others it should not."""
    alone = holm({"turns": 0.04})
    family = holm({"success": 0.03, "tokens": 0.035, "turns": 0.04})
    assert alone["turns"]["significant"]
    assert not family["turns"]["significant"]


def test_holm_is_monotone():
    out = holm({"a": 0.001, "b": 0.5, "c": 0.02})
    adjusted = [out[k]["p_adjusted"] for k in sorted(out, key=lambda k: out[k]["p"])]
    assert adjusted == sorted(adjusted), "step-down adjusted p-values must be non-decreasing"


@pytest.mark.slow
def test_holm_family_wise_error_rate_under_the_null():
    """With three null metrics, at most ~5% of reports should flag anything."""
    rng = np.random.default_rng(23)
    any_flagged = 0
    trials = 400
    for _ in range(trials):
        ps = {name: float(rng.random()) for name in ("success", "tokens", "turns")}
        if any(v["significant"] for v in holm(ps).values()):
            any_flagged += 1
    rate = any_flagged / trials
    assert rate <= 0.09, f"family-wise error rate {rate:.0%}"


# --------------------------------------------------------------------------------------------------------------
# guards and grouping
# --------------------------------------------------------------------------------------------------------------


def test_minimum_n_guard_refuses_a_tiny_eval_set():
    with pytest.raises(TooFewTasks, match="at least 8"):
        minimum_n_guard(3, 5)


def test_minimum_n_guard_allows_a_single_repeat():
    minimum_n_guard(20, 1)


def test_success_by_task_groups_repeats():
    rows = [{"task_id": "t1", "success": True}, {"task_id": "t1", "success": False},
            {"task_id": "t2", "success": True}]
    assert success_by_task(rows) == {"t1": [1.0, 0.0], "t2": [1.0]}


def test_metric_by_task_averages_repeats():
    rows = [{"task_id": "t1", "n_turns": 2}, {"task_id": "t1", "n_turns": 4}]
    assert metric_by_task(rows, "n_turns") == {"t1": 3.0}


def test_success_by_task_handles_none():
    assert success_by_task([{"task_id": "t", "success": None}]) == {"t": [0.0]}
