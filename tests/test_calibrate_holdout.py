"""Fitting the confidence gate.

The gate publishes its own reliability, so these check that what it publishes is honest: a task-disjoint split,
holdout metrics rather than in-sample ones, bins that account for every turn, and loud refusals when the fit is
not good enough to threshold on.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentdistill.cascade.calibrate import (
    MAX_ACCEPTABLE_ECE,
    MIN_TURNS,
    MIN_USEFUL_AUROC,
    expected_calibration_error,
    fit_calibrator,
    load,
    reliability_bins,
    save,
    split_by_task,
)


def synth(n_tasks: int = 80, per_task: int = 5, signal: float = 1.0, seed: int = 0):
    """Turns whose goodness is driven by a latent feature, so a fitted gate should find it."""
    rng = np.random.default_rng(seed)
    task_ids, X, y = [], [], []
    for t in range(n_tasks):
        for _ in range(per_task):
            conf = rng.normal()
            task_ids.append(f"task{t}")
            X.append([conf * signal, conf * signal - abs(rng.normal()), rng.normal()])
            y.append(int(rng.random() < 1 / (1 + np.exp(-conf * signal * 2))))
    return task_ids, np.array(X), np.array(y)


NAMES = ["mean_logprob", "min_logprob", "noise"]


# --------------------------------------------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------------------------------------------


def test_split_never_straddles_a_task():
    """Turns within a task are correlated; splitting by turn leaks the task and flatters the gate."""
    task_ids, _, _ = synth()
    fit_idx, hold_idx = split_by_task(task_ids)
    fit_tasks = {task_ids[i] for i in fit_idx}
    hold_tasks = {task_ids[i] for i in hold_idx}
    assert not (fit_tasks & hold_tasks)
    assert len(fit_idx) + len(hold_idx) == len(task_ids)


def test_split_is_deterministic_under_a_seed():
    task_ids, _, _ = synth()
    a, _ = split_by_task(task_ids, seed=3)
    b, _ = split_by_task(task_ids, seed=3)
    assert np.array_equal(a, b)


def test_split_respects_the_fraction():
    task_ids, _, _ = synth(n_tasks=100, per_task=1)
    fit_idx, _ = split_by_task(task_ids, frac=0.6)
    assert 50 <= len(fit_idx) <= 70


# --------------------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------------------


def test_holdout_is_not_better_than_in_sample_on_average():
    """The point of the split: in-sample numbers are optimistic by construction."""
    task_ids, X, y = synth(signal=1.5)
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert res.in_sample["auroc"] >= res.holdout["auroc"] - 0.05


def test_a_learnable_signal_is_found():
    task_ids, X, y = synth(n_tasks=120, signal=2.0)
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert res.holdout["auroc"] > 0.65


def test_pure_noise_produces_a_useless_gate_and_says_so():
    rng = np.random.default_rng(1)
    task_ids = [f"task{i // 5}" for i in range(600)]
    X = rng.normal(size=(600, 3))
    y = (rng.random(600) < 0.5).astype(int)
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert not res.usable
    assert any("not separating" in n or "do not mean what they say" in n for n in res.notes)


def test_reliability_bins_account_for_every_turn():
    p = np.array([0.05, 0.15, 0.55, 0.95, 0.99])
    y = np.array([0, 0, 1, 1, 1])
    bins = reliability_bins(p, y)
    assert len(bins) == 10
    assert sum(b["n"] for b in bins) == len(p)
    assert all(b["mean_predicted"] is None for b in bins if b["n"] == 0)


def test_ece_is_zero_for_a_perfectly_calibrated_gate():
    p = np.array([0.0] * 50 + [1.0] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert expected_calibration_error(p, y) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_a_confidently_wrong_gate():
    p = np.full(100, 0.95)
    y = np.zeros(100, dtype=int)
    assert expected_calibration_error(p, y) > 0.9


def test_a_single_outcome_cannot_support_a_gate():
    """Not a modelling failure: there is nothing to discriminate, and a fitted model would be a constant."""
    task_ids = [f"t{i // 5}" for i in range(200)]
    X = np.random.default_rng(0).normal(size=(200, 3))
    res = fit_calibrator(X, np.ones(200, dtype=int), task_ids, NAMES)
    assert res.model is None and not res.usable
    assert "same outcome" in res.notes[0]


def test_auroc_is_nan_rather_than_misleading_when_a_split_has_one_class():
    """Undefined beats a spurious 0.5 or 1.0."""
    from agentdistill.cascade.calibrate import metrics

    m = metrics(np.array([0.2, 0.8]), np.array([1, 1]))
    assert np.isnan(m["auroc"])


# --------------------------------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------------------------------


def test_too_few_turns_refuses_to_fit():
    task_ids, X, y = synth(n_tasks=5, per_task=2)
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert res.model is None and not res.usable
    assert f"need {MIN_TURNS}" in res.notes[0]
    assert "escalate everything" in res.notes[0]


def test_usable_requires_auroc_and_ece_together():
    task_ids, X, y = synth(n_tasks=120, signal=2.0)
    res = fit_calibrator(X, y, task_ids, NAMES)
    expected = (
        res.n_turns >= MIN_TURNS
        and res.holdout["auroc"] >= MIN_USEFUL_AUROC
        and res.holdout["ece"] <= MAX_ACCEPTABLE_ECE
    )
    assert res.usable == expected


def test_weak_labels_are_called_out():
    task_ids, X, y = synth(n_tasks=100, signal=2.0)
    res = fit_calibrator(X, y, task_ids, NAMES, label_mix={"uncorrected": 400, "teacher_match": 10})
    assert any("uncorrected" in n for n in res.notes)


# --------------------------------------------------------------------------------------------------------------
# the artifact
# --------------------------------------------------------------------------------------------------------------


def test_artifact_round_trips_and_keeps_the_feature_order(tmp_path):
    task_ids, X, y = synth(n_tasks=100, signal=2.0)
    res = fit_calibrator(X, y, task_ids, NAMES)
    model, report = load(save(res, tmp_path / "cal"))
    assert report["feature_order"] == NAMES
    assert model is not None
    assert model.predict_proba(X[:1]).shape == (1, 2)


def test_shipped_artifact_is_refit_on_everything(tmp_path):
    """Holdout numbers are about honesty, not about shipping a weaker model."""
    task_ids, X, y = synth(n_tasks=100, signal=2.0)
    res = fit_calibrator(X, y, task_ids, NAMES)
    p = res.model.predict_proba(X)[:, 1]
    assert res.in_sample["n"] == len(y), "in-sample metrics come from the refit model over all the data"
    assert len(p) == len(y)


def test_unfitted_result_saves_a_report_without_a_model(tmp_path):
    task_ids, X, y = synth(n_tasks=4, per_task=2)
    res = fit_calibrator(X, y, task_ids, NAMES)
    model, report = load(save(res, tmp_path / "cal"))
    assert model is None
    assert report["usable"] is False
    assert report["notes"]


# --------------------------------------------------------------------------------------------------------------
# features with no values
#
# `agreement` is NaN on every turn when self-consistency sampling is off, which is the default in tiny mode.
# HistGradientBoosting cannot bin an all-NaN column and raises, so this must be handled before the fit.
# --------------------------------------------------------------------------------------------------------------


def test_an_all_nan_feature_is_dropped_and_reported():
    task_ids, X, y = synth(n_tasks=80, signal=1.5)
    X[:, 2] = np.nan  # stands in for `agreement` with k_samples = 0
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert res.model is not None, "a dead column must not take down the whole fit"
    assert res.feature_order == NAMES[:2]
    assert any("noise" in n for n in res.notes)


def test_dropping_a_feature_still_produces_holdout_metrics():
    task_ids, X, y = synth(n_tasks=100, signal=2.0)
    X[:, 2] = np.nan
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert res.holdout["n"] > 0
    assert res.holdout["auroc"] == res.holdout["auroc"]  # not NaN


def test_every_feature_empty_cannot_support_a_gate():
    task_ids, X, y = synth(n_tasks=80)
    X[:] = np.nan
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert res.model is None and not res.usable
    assert any("every feature was empty" in n for n in res.notes)


def test_the_saved_artifact_records_the_reduced_order(tmp_path):
    task_ids, X, y = synth(n_tasks=100, signal=2.0)
    X[:, 2] = np.nan
    res = fit_calibrator(X, y, task_ids, NAMES)
    _model, report = load(save(res, tmp_path / "cal"))
    assert report["feature_order"] == NAMES[:2], "the runtime must build exactly the columns the model saw"


def test_a_calibrator_that_cannot_be_fitted_degrades_instead_of_raising(monkeypatch):
    """A GPU-day stage must not die for a data shape the caller cannot fix mid-run."""
    import agentdistill.cascade.calibrate as calibrate_module

    def explode(*a, **kw):
        raise RuntimeError("sklearn said no")

    monkeypatch.setattr(calibrate_module, "_new_model", explode)
    task_ids, X, y = synth(n_tasks=80)
    res = fit_calibrator(X, y, task_ids, NAMES)
    assert res.model is None and not res.usable
    assert "could not be fitted" in res.notes[-1]
