"""The retrain loop.

A distilled student decays. Production traffic drifts away from the corpus the student was trained on, a tool
gains an argument, a policy changes, and the adapter that was measured at parity six weeks ago is no longer at
parity. The retrain loop is what turns that from an incident into a scheduled job.

It is a pipeline of stages, each with a gate. The gates are the point. A retrain that runs unattended and
promotes whatever it produced is a mechanism for putting an unmeasured model into production every week; every
stage here has to prove something before the next one starts, and the pipeline stops at the first stage that
cannot. Stopping is a normal outcome, not a failure: most weeks there is not enough new traffic to justify a
retrain, and the right behaviour is to notice that in the first stage and stop.

Nothing here promotes to prod. The last stage promotes to *canary*, and the canary earns prod through
`adapter promote`, which requires a live comparison that only accumulates with real traffic over days.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Gates, in one place, so a reader does not have to reconstruct the policy from eight call sites.
MIN_NEW_REQUESTS = 50
MIN_NEW_SAMPLES = 50
MAX_SUCCESS_REGRESSION_PP = 1.0
MAX_HOLDOUT_ECE = 0.05
MIN_HOLDOUT_AUROC = 0.6
MAX_QUANTIZATION_DROP_PP = 2.0


@dataclass
class Stage:
    name: str
    run: Callable[[dict], dict]
    gate: Callable[[dict], tuple[bool, str]] | None = None
    #: What this stage would do, for `--dry-run`. Takes the context, returns one line.
    describe: Callable[[dict], str] | None = None


@dataclass
class Markers:
    """Per-retrain completion markers, so an interrupted run resumes rather than restarts.

    Scoped by retrain id: a second `retrain` starts fresh rather than skipping everything the previous one did.
    """

    root: Path
    retrain_id: str

    @property
    def dir(self) -> Path:
        return self.root / "retrain" / self.retrain_id

    def done(self, stage: str) -> bool:
        return (self.dir / f"{stage}.done").exists()

    def mark(self, stage: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{stage}.done").touch()


@dataclass
class PipelineResult:
    ctx: dict
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stopped_at: str | None = None
    stop_reason: str | None = None
    gates: dict[str, tuple[bool, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.stopped_at is None

    def summary(self) -> str:
        if self.ok:
            return f"retrain complete: ran {', '.join(self.ran) or 'nothing'}"
        return f"retrain stopped at {self.stopped_at}: {self.stop_reason}"


def run_pipeline(
    stages: list[Stage],
    ctx: dict,
    markers: Markers | None = None,
    log: Callable[[str], None] = logger.info,
    start_from: str | None = None,
    dry_run: bool = False,
) -> PipelineResult:
    """Run stages in order, stopping at the first failed gate.

    `start_from` overrides the markers for the named stage onward: resuming a run means rerunning the stage
    that failed, and a marker from a previous attempt must not skip it.
    """
    names = [s.name for s in stages]
    if start_from and start_from not in names:
        raise ValueError(f"unknown stage {start_from!r}; stages are {', '.join(names)}")

    result = PipelineResult(ctx=ctx)
    started = start_from is None
    for stage in stages:
        if not started:
            started = stage.name == start_from
            if not started:
                log(f"skip {stage.name} (before --from)")
                result.skipped.append(stage.name)
                continue

        if markers and markers.done(stage.name) and start_from is None:
            log(f"skip {stage.name} (done)")
            result.skipped.append(stage.name)
            continue

        if dry_run:
            line = stage.describe(result.ctx) if stage.describe else "no description"
            log(f"would run {stage.name}: {line}")
            result.ran.append(stage.name)
            continue

        log(f"run {stage.name}")
        result.ctx = stage.run(result.ctx)
        result.ran.append(stage.name)

        if stage.gate:
            ok, why = stage.gate(result.ctx)
            result.gates[stage.name] = (ok, why)
            log(f"  gate {stage.name}: {'pass' if ok else 'STOP'} — {why}")
            if not ok:
                result.stopped_at, result.stop_reason = stage.name, why
                result.ctx["stopped_at"] = stage.name
                result.ctx["stop_reason"] = why
                return result

        if markers:
            markers.mark(stage.name)

    return result


# ------------------------------------------------------------------------------------------------------------
# the gates
#
# Each takes the context a stage produced and answers whether the next stage is justified. They are separate
# functions, and separately tested, because a gate that is only exercised through a full pipeline run is a gate
# nobody has checked.
# ------------------------------------------------------------------------------------------------------------


def gate_enough_new_traffic(ctx: dict) -> tuple[bool, str]:
    n = int(ctx.get("n_requests") or 0)
    return (
        n >= MIN_NEW_REQUESTS,
        f"{n} graded requests since the last retrain (need {MIN_NEW_REQUESTS}). Most weeks there is not "
        f"enough new traffic to justify a retrain, and stopping here is the normal outcome."
        if n < MIN_NEW_REQUESTS else f"{n} graded requests since the last retrain",
    )


def gate_enough_new_samples(ctx: dict) -> tuple[bool, str]:
    n = int(ctx.get("n_new_samples") or 0)
    return (
        n >= MIN_NEW_SAMPLES,
        f"{n} new samples after curation (need {MIN_NEW_SAMPLES}); the filters rejected most of the new "
        f"traffic, so there is nothing to learn from"
        if n < MIN_NEW_SAMPLES else f"{n} new samples after curation",
    )


def gate_training_converged(ctx: dict) -> tuple[bool, str]:
    """Eval loss finite.

    A deliberately weak gate: loss is a proxy, and a good loss says nothing about whether the student calls the
    right tool. It exists only to catch a diverged run before spending an eval on it. The real check is two
    stages later.
    """
    loss = ctx.get("eval_loss")
    if loss is None:
        return False, "training produced no eval loss; the run did not complete"
    try:
        value = float(loss)
    except (TypeError, ValueError):
        return False, f"eval loss is not a number: {loss!r}"
    import math

    if not math.isfinite(value):
        return False, f"eval loss is {value}; training diverged"
    return True, f"eval loss {value:.4f} (a proxy; the eval two stages on is the real check)"


def gate_round_promoted(ctx: dict) -> tuple[bool, str]:
    decision = ctx.get("round_decision")
    return (
        decision == "promote",
        f"the on-policy round decided {decision!r}; only 'promote' continues"
        if decision != "promote" else "the on-policy round promoted its candidate",
    )


def gate_not_worse_than_prod(ctx: dict) -> tuple[bool, str]:
    """The candidate is not measurably worse than what is already serving.

    On the low end of the interval rather than the point estimate: a candidate whose success is 0.5 pp below
    prod with an interval spanning 8 points has not been shown to be at parity, it has been measured badly.
    """
    cmp = ctx.get("comparison")
    if not cmp:
        return False, "no paired comparison against prod; there is nothing to promote on"
    if cmp.get("insufficient_power"):
        return False, cmp["insufficient_power"]["reason"]
    lo = cmp["success"]["ci95"][0] * 100
    return (
        lo >= -MAX_SUCCESS_REGRESSION_PP,
        f"success CI low end {lo:+.1f} pp against prod (must be at least -{MAX_SUCCESS_REGRESSION_PP})",
    )


def gate_calibrated(ctx: dict) -> tuple[bool, str]:
    """The gate the cascade will run on is itself measured, on a holdout it did not see.

    Both numbers are required. ECE alone passes a calibrator that outputs the base rate for every turn -- well
    calibrated, and useless, because it never separates a turn the student got right from one it did not.
    AUROC is what catches that.
    """
    metrics = ctx.get("calibration") or {}
    ece, auroc = metrics.get("ece"), metrics.get("auroc")
    if ece is None or auroc is None:
        return False, "calibration produced no holdout metrics; the cascade would escalate everything"
    ok = ece <= MAX_HOLDOUT_ECE and auroc >= MIN_HOLDOUT_AUROC
    detail = f"holdout ECE {ece:.3f} (max {MAX_HOLDOUT_ECE}), AUROC {auroc:.3f} (min {MIN_HOLDOUT_AUROC})"
    if not ok and auroc < MIN_HOLDOUT_AUROC:
        detail += " — a gate that cannot separate right turns from wrong ones saves nothing"
    return ok, detail


def gate_quantization_lossless_enough(ctx: dict) -> tuple[bool, str]:
    """Quantization cost at most two points of success.

    One-sided: quantization scoring higher is evidence that the eval set is too small to resolve the
    difference, not that quantization improved the model, and either way it is not a reason to stop.
    """
    drop = ctx.get("quantization_drop_pp")
    if drop is None:
        return True, "no quantization step, or nothing to compare against"
    return (
        float(drop) <= MAX_QUANTIZATION_DROP_PP,
        f"quantization cost {float(drop):+.1f} pp of success (max {MAX_QUANTIZATION_DROP_PP})",
    )


def gate_promotion_checks_green(ctx: dict) -> tuple[bool, str]:
    checks = ctx.get("promotion_checks") or {}
    if not checks:
        return False, "no promotion checks were run"
    failed = sorted(name for name, check in checks.items() if not _check_ok(check))
    return (
        not failed,
        f"lifecycle checks failed: {', '.join(failed)}" if failed
        else f"all {len(checks)} lifecycle checks green",
    )


def _check_ok(check: Any) -> bool:
    return bool(check.get("ok")) if isinstance(check, dict) else bool(getattr(check, "ok", False))


#: The gate each stage runs, in pipeline order.
GATES: dict[str, Callable[[dict], tuple[bool, str]]] = {
    "ingest_gateway": gate_enough_new_traffic,
    "curate": gate_enough_new_samples,
    "train_sft_continue": gate_training_converged,
    "onpolicy": gate_round_promoted,
    "eval": gate_not_worse_than_prod,
    "calibrate": gate_calibrated,
    "quantize": gate_quantization_lossless_enough,
    "promote_canary": gate_promotion_checks_green,
}

STAGE_ORDER = tuple(GATES)
