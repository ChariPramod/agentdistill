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

#: Below this holdout AUROC the gate is at chance, and its verdict is `uninformative`. The gateway refuses to
#: load such a calibration, so a coin flip never decides which turns reach the teacher.
MIN_INFORMATIVE_AUROC = 0.55

#: Holdout AUROC from which a gate is reliable enough to threshold on, given an acceptable ECE.
RELIABLE_AUROC = MIN_USEFUL_AUROC

#: The verdicts a calibration row can carry. Only `usable` is ever loaded by the gateway.
VERDICTS = ("usable", "no_threshold", "unreliable", "uninformative", "degenerate_labels")

#: Minimum minority-class samples per calibration fold.
MIN_PER_FOLD = 10

#: Below this many minority-class samples, isotonic overfits and sigmoid is used instead.
ISOTONIC_MIN_MINORITY = 50


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
    min_turns: int = MIN_TURNS
    #: Why no model was fitted, when none was: `too_few_turns`, `no_features`, `single_class`, `fit_error`. Only
    #: `single_class` is a measurement (every turn got the same label, so there is nothing to discriminate); the
    #: rest mean the input could not support a fit at all.
    unfit_reason: str | None = None

    @property
    def usable(self) -> bool:
        """Is this gate worth thresholding on, rather than escalating everything?"""
        return (
            self.n_turns >= self.min_turns
            and (self.holdout.get("auroc") or 0.0) >= MIN_USEFUL_AUROC
            and (1.0 if self.holdout.get("ece") is None else self.holdout["ece"]) <= MAX_ACCEPTABLE_ECE
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
            "min_turns": self.min_turns,
            "unfit_reason": self.unfit_reason,
        }


def verdict_for(n_turns: int, n_positive: int, auroc: float | None, ece: float | None, min_turns: int,
                min_auroc: float = MIN_INFORMATIVE_AUROC, reliable_auroc: float = RELIABLE_AUROC,
                max_ece: float = MAX_ACCEPTABLE_ECE) -> tuple[str, str]:
    """(verdict, reason) for a gate's measurements. Only `usable` is loaded by the gateway.

    The causes are kept apart because their fixes differ. `degenerate_labels` means every turn got the same label
    -- the tasks did not separate, a data problem. `uninformative` means the labels separated and the features
    could not tell them apart -- a feature problem. A random tiny model gets every turn wrong, so tiny mode reports
    the first, which is the honest label for it.
    """
    n_negative = n_turns - n_positive
    if n_turns < min_turns:
        return "unreliable", f"{n_turns} labelled turns below the minimum of {min_turns}"
    if n_positive == 0 or n_negative == 0:
        only = "good" if n_negative == 0 else "bad"
        return "degenerate_labels", f"every one of {n_turns} labelled turns is {only}; AUROC is undefined"
    if auroc is None or auroc != auroc:
        return "unreliable", "AUROC could not be computed"
    if auroc < min_auroc:
        return "uninformative", f"holdout AUROC {auroc:.3f} below {min_auroc}"
    if auroc < reliable_auroc or (ece is not None and ece > max_ece):
        return "unreliable", f"AUROC {auroc:.3f}, ECE {'n/a' if ece is None else round(ece, 3)}"
    return "usable", f"AUROC {auroc:.3f}, ECE {'n/a' if ece is None else round(ece, 3)}"


def gate_verdict(result: CalibrationResult, n_positive: int, threshold_chosen: bool) -> tuple[str, str]:
    """`verdict_for` on a fit, plus the threshold search's say: a usable gate with no threshold inside the
    budget is `no_threshold`."""
    verdict, reason = verdict_for(result.n_turns, n_positive, result.holdout.get("auroc"),
                                  result.holdout.get("ece"), result.min_turns)
    if verdict == "usable" and not threshold_chosen:
        return "no_threshold", f"{reason}; no threshold keeps the success drop inside the budget"
    return verdict, reason


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


def _new_model(seed: int, y: np.ndarray | None = None):
    """The gate's classifier, wrapped in a calibration layer sized to the data.

    HistGradientBoosting handles NaN natively, which is why it is here: "no tool call" means genuinely absent
    argument features, and imputing them would teach the gate that absence is confidence.

    The calibration wrapper is sized to the smaller class. Isotonic regression needs a reasonable number of
    samples per fold, and asking for five folds of a hundred turns where one class has twelve members fails
    inside sklearn rather than producing a bad model -- which is a worse failure, because it happens mid-run.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)
    if y is None:
        return CalibratedClassifierCV(base, method="isotonic", cv=5)

    minority = int(min(np.bincount(y.astype(int), minlength=2)))
    folds = max(2, min(5, minority // MIN_PER_FOLD))
    # Isotonic is a step function fitted per fold; below this it overfits to a handful of points, and sigmoid
    # (a two-parameter fit) is the better-behaved choice.
    method = "isotonic" if minority >= ISOTONIC_MIN_MINORITY else "sigmoid"
    return CalibratedClassifierCV(base, method=method, cv=folds)


def fit_calibrator(
    X: np.ndarray,
    y: np.ndarray,
    task_ids: list[str],
    feature_order: list[str],
    seed: int = 0,
    fit_frac: float = 0.7,
    label_mix: dict | None = None,
    min_turns: int = MIN_TURNS,
) -> CalibrationResult:
    """Fit on a task-disjoint split, report on the rest, refit on everything for the artifact."""
    notes: list[str] = []
    n_tasks = len(set(task_ids))

    if len(y) < min_turns:
        notes.append(
            f"only {len(y)} labelled turns (need {min_turns}); the gate is not fitted and the cascade should "
            f"escalate everything"
        )
        return CalibrationResult(
            feature_order=feature_order, holdout={}, in_sample={}, reliability_bins=[],
            n_turns=len(y), n_tasks=n_tasks, label_mix=label_mix or {}, model=None, notes=notes,
            min_turns=min_turns, unfit_reason="too_few_turns",
        )

    # A feature with no finite value anywhere carries no signal, and HistGradientBoosting cannot bin it -- it
    # raises rather than ignoring the column. This is not hypothetical: `agreement` is NaN for every turn
    # whenever self-consistency sampling is off (`cascade.k_samples: 0`), which is the default in tiny mode.
    keep = [i for i in range(X.shape[1]) if np.isfinite(X[:, i]).any()]
    dropped = [feature_order[i] for i in range(X.shape[1]) if i not in keep]
    if dropped:
        notes.append(
            f"dropped {dropped} from the gate: no turn had a value for them. "
            f"`agreement` appears here when cascade.k_samples is 0, which switches off self-consistency."
        )
        X = X[:, keep]
        feature_order = [feature_order[i] for i in keep]
    if not feature_order:
        notes.append("every feature was empty; no gate can be fitted. Escalate everything.")
        return CalibrationResult(
            feature_order=[], holdout={}, in_sample={}, reliability_bins=[],
            n_turns=len(y), n_tasks=n_tasks, label_mix=label_mix or {}, model=None, notes=notes,
            min_turns=min_turns, unfit_reason="no_features",
        )

    if len(set(y.tolist())) < 2:
        # Every turn carries the same label, so there is nothing to discriminate. This is not a modelling
        # failure; it means the label set cannot support a gate, and pretending otherwise would ship a
        # classifier that always returns the same probability.
        notes.append(
            f"every one of the {len(y)} labelled turns has the same outcome, so no gate can be fitted. "
            f"Escalate everything, and check the turn labels before trying again."
        )
        # Recorded as what it is: a measurement with no variance. AUROC is undefined, which is the definition of
        # a gate that cannot separate anything.
        holdout = {"n": len(y), "positive_rate": float(y.mean()), "auroc": None, "ece": None, "brier": None}
        return CalibrationResult(
            feature_order=feature_order, holdout=holdout, in_sample={}, reliability_bins=[],
            n_turns=len(y), n_tasks=n_tasks, label_mix=label_mix or {}, model=None, notes=notes,
            min_turns=min_turns, unfit_reason="single_class",
        )

    fit_idx, hold_idx = split_by_task(task_ids, frac=fit_frac, seed=seed)
    if len(hold_idx) == 0 or len(set(y[fit_idx].tolist())) < 2 or len(set(y[hold_idx].tolist())) < 2:
        notes.append(
            "the task split left a side with only one outcome, so the holdout cannot be scored; "
            "metrics below are in-sample and optimistic"
        )
        fit_idx, hold_idx = np.arange(len(y)), np.arange(len(y))

    try:
        fitted = _new_model(seed, y[fit_idx])
        fitted.fit(X[fit_idx], y[fit_idx])
        p_hold = fitted.predict_proba(X[hold_idx])[:, 1]
    except Exception as e:
        # A gate that cannot be fitted is a gate that escalates everything. Crashing here would take down a
        # GPU-day stage for a data shape the caller can do nothing about mid-run.
        notes.append(
            f"the calibrator could not be fitted ({type(e).__name__}: {e}). Escalate everything, and check the "
            f"turn labels and feature coverage."
        )
        return CalibrationResult(
            feature_order=feature_order, holdout={}, in_sample={}, reliability_bins=[],
            n_turns=len(y), n_tasks=n_tasks, label_mix=label_mix or {}, model=None, notes=notes,
            min_turns=min_turns, unfit_reason="fit_error",
        )

    holdout = metrics(p_hold, y[hold_idx])
    bins = reliability_bins(p_hold, y[hold_idx])

    # The artifact is refit on everything: reporting holdout numbers is about honesty, not about shipping a
    # weaker model.
    shipped = _new_model(seed, y)
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
        min_turns=min_turns,
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


def assert_feature_order(report: dict, configured: list[str] | tuple[str, ...]) -> list[str]:
    """Check the runtime can build the vector the model was fitted on, and return that order.

    The stored order is authoritative: it may be a subset of the configured features, because calibration drops
    columns that had no values. What must not happen is the runtime building a vector in a different order, or
    containing a feature the model never saw -- either scores silently and wrongly.
    """
    stored = list(report.get("feature_order") or [])
    configured_list = list(configured)
    if not stored:
        raise ValueError("the calibration records no feature order; it cannot be used to score")

    missing = [f for f in stored if f not in configured_list]
    if missing:
        raise ValueError(
            f"the calibration was fitted on {missing}, which the runtime is not configured to produce. "
            f"Configured: {configured_list}. Scoring without them produces confident nonsense."
        )
    # The stored order must be a subsequence of the configured one, or the two disagree about position.
    positions = [configured_list.index(f) for f in stored]
    if positions != sorted(positions):
        raise ValueError(
            f"the calibration's feature order {stored} is not in the configured order {configured_list}. "
            f"Scoring with a different order produces confident nonsense."
        )
    return stored
