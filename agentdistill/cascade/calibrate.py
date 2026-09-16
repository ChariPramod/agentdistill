"""Fitting the confidence gate.

The gate publishes its own reliability. A gate whose probabilities are not calibrated is worse than no gate: it
escalates the wrong turns and reports a threshold that does not mean what it says.

Two rules govern how it is fitted:

- **Never fit on training tasks.** The student's confidence on prefixes it memorized is not informative about its
  confidence in the field.
- **Report holdout numbers, ship the refit artifact.** Metrics come from a split the model never saw, because
  in-sample calibration error is optimistic by construction. The artifact that ships is refit on everything,
  because throwing away a third of the data to make the artifact match the report would be worse.

The split is by **task**, not by turn: turns within a task are correlated, so a turn-level split leaks the task
across both sides and reports a gate that looks better than it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

#: Below this many labelled turns, the fit is noise and the gate should escalate everything.
MIN_TURNS = 100

#: Above this, the gate's probabilities are not trustworthy enough to threshold on.
MAX_ACCEPTABLE_ECE = 0.05

#: Below this, the gate is not distinguishing good turns from bad ones at all.
MIN_USEFUL_AUROC = 0.6


@dataclass
class CalibrationResult:
    feature_order: list[str]
    holdout: dict
    in_sample: dict
    reliability_bins: list[dict]
    n_turns: int
    n_tasks: int
    label_mix: dict = field(default_factory=dict)
    model: Any = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Is this gate worth thresholding on, rather than escalating everything?"""
        return (
            self.n_turns >= MIN_TURNS
            and self.holdout.get("auroc", 0.0) >= MIN_USEFUL_AUROC
            and self.holdout.get("ece", 1.0) <= MAX_ACCEPTABLE_ECE
        )

    def to_dict(self) -> dict:
        return {
            "feature_order": self.feature_order,
            "holdout_metrics": self.holdout,
            "in_sample_metrics": self.in_sample,
            "reliability_bins": self.reliability_bins,
            "n_turns": self.n_turns,
            "n_tasks": self.n_tasks,
            "label_mix": self.label_mix,
            "usable": self.usable,
            "notes": self.notes,
        }


def split_by_task(task_ids: list[str], frac: float = 0.7, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split turn indices by task. Turns of one task never straddle the split."""
    tasks = sorted(set(task_ids))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(tasks))
    cut = max(1, int(len(shuffled) * frac))
    fit_tasks = set(shuffled[:cut])
    idx = np.arange(len(task_ids))
    in_fit = np.array([t in fit_tasks for t in task_ids], dtype=bool)
    return idx[in_fit], idx[~in_fit]


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Weighted average gap between predicted confidence and observed frequency."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        in_bin = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """The reliability diagram, as data. Empty bins are kept so the bins sum to n."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in pairwise(edges):
        in_bin = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        n = int(in_bin.sum())
        out.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": n,
            "mean_predicted": float(p[in_bin].mean()) if n else None,
            "observed": float(y[in_bin].mean()) if n else None,
        })
    return out


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    out: dict[str, Any] = {"n": len(y), "positive_rate": float(y.mean()) if len(y) else float("nan")}
    # AUROC is undefined when every label is the same, which happens on a small holdout.
    out["auroc"] = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
    out["brier"] = float(brier_score_loss(y, p)) if len(y) else float("nan")
    out["ece"] = expected_calibration_error(p, y) if len(y) else float("nan")
    return out


def _new_model(seed: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    # HistGradientBoosting handles NaN natively, which matters: "no tool call" means genuinely absent argument
    # features, and imputing them would teach the gate that absence is confidence.
    base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)


def fit_calibrator(
    X: np.ndarray,
    y: np.ndarray,
    task_ids: list[str],
    feature_order: list[str],
    seed: int = 0,
    fit_frac: float = 0.7,
    label_mix: dict | None = None,
) -> CalibrationResult:
    """Fit on a task-disjoint split, report on the rest, refit on everything for the artifact."""
    notes: list[str] = []
    n_tasks = len(set(task_ids))

    if len(y) < MIN_TURNS:
        notes.append(
            f"only {len(y)} labelled turns (need {MIN_TURNS}); the gate is not fitted and the cascade should "
            f"escalate everything"
        )
        return CalibrationResult(
            feature_order=feature_order, holdout={}, in_sample={}, reliability_bins=[],
            n_turns=len(y), n_tasks=n_tasks, label_mix=label_mix or {}, model=None, notes=notes,
        )

    if len(set(y.tolist())) < 2:
        # Every turn carries the same label, so there is nothing to discriminate. This is not a modelling
        # failure; it means the label set cannot support a gate, and pretending otherwise would ship a
        # classifier that always returns the same probability.
        notes.append(
            f"every one of the {len(y)} labelled turns has the same outcome, so no gate can be fitted. "
            f"Escalate everything, and check the turn labels before trying again."
        )
        return CalibrationResult(
            feature_order=feature_order, holdout={}, in_sample={}, reliability_bins=[],
            n_turns=len(y), n_tasks=n_tasks, label_mix=label_mix or {}, model=None, notes=notes,
        )

    fit_idx, hold_idx = split_by_task(task_ids, frac=fit_frac, seed=seed)
    if len(hold_idx) == 0 or len(set(y[fit_idx].tolist())) < 2 or len(set(y[hold_idx].tolist())) < 2:
        notes.append(
            "the task split left a side with only one outcome, so the holdout cannot be scored; "
            "metrics below are in-sample and optimistic"
        )
        fit_idx, hold_idx = np.arange(len(y)), np.arange(len(y))

    fitted = _new_model(seed)
    fitted.fit(X[fit_idx], y[fit_idx])
    p_hold = fitted.predict_proba(X[hold_idx])[:, 1]
    holdout = metrics(p_hold, y[hold_idx])
    bins = reliability_bins(p_hold, y[hold_idx])

    # The artifact is refit on everything: reporting holdout numbers is about honesty, not about shipping a
    # weaker model.
    shipped = _new_model(seed)
    shipped.fit(X, y)
    p_in = shipped.predict_proba(X)[:, 1]
    in_sample = metrics(p_in, y)

    if holdout.get("auroc", 0.0) < MIN_USEFUL_AUROC:
        notes.append(
            f"holdout AUROC {holdout.get('auroc'):.3f} is below {MIN_USEFUL_AUROC}: the gate is not separating "
            f"good turns from bad ones. Escalate everything and say so in the report."
        )
    if holdout.get("ece", 1.0) > MAX_ACCEPTABLE_ECE:
        notes.append(
            f"holdout ECE {holdout.get('ece'):.3f} is above {MAX_ACCEPTABLE_ECE}: the probabilities do not mean "
            f"what they say, so a threshold on them does not either."
        )
    if label_mix and label_mix.get("uncorrected", 0) > 0.5 * max(sum(label_mix.values()), 1):
        notes.append(
            "more than half the turn labels are `uncorrected`, which is an assumption rather than evidence; "
            "read the AUROC with that in mind"
        )

    return CalibrationResult(
        feature_order=feature_order,
        holdout=holdout,
        in_sample=in_sample,
        reliability_bins=bins,
        n_turns=len(y),
        n_tasks=n_tasks,
        label_mix=label_mix or {},
        model=shipped,
        notes=notes,
    )


def save(result: CalibrationResult, path: str | Path) -> Path:
    """Persist the model and its report side by side, with the feature order the runtime must assert."""
    import pickle

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if result.model is not None:
        (p / "model.pkl").write_bytes(pickle.dumps(result.model))
    (p / "calibration.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
    return p


def load(path: str | Path) -> tuple[Any, dict]:
    """Load a calibration artifact. Returns (model or None, report)."""
    import pickle

    p = Path(path)
    report = json.loads((p / "calibration.json").read_text())
    model_path = p / "model.pkl"
    model = pickle.loads(model_path.read_bytes()) if model_path.exists() else None
    return model, report


def assert_feature_order(report: dict, configured: list[str] | tuple[str, ...]) -> None:
    """A reordered feature vector scores silently and wrongly, so the runtime checks before it scores."""
    stored = list(report.get("feature_order") or [])
    if stored != list(configured):
        raise ValueError(
            f"the calibration was fitted on features {stored} but the runtime is configured for "
            f"{list(configured)}. Scoring with a different order produces confident nonsense."
        )
