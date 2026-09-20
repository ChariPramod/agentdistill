"""Every orchestrated stage's inputs are produced by a stage before it.

The retrain loop's `calibrate` stage called `agentdistill calibrate <adapter>` with no `--from-eval`, because no
stage before it ran the calibration-set eval that flag needs. The flag checks passed -- it was optional to the
parser -- and every retrain stopped at calibrate. These tests would have caught it without running anything:
statically, from each stage's declared `needs` and `provides`, and dynamically, by recording what each stage
actually reads from the context.
"""

from __future__ import annotations

import pytest

from agentdistill.eval.harness import TaskOutcome
from agentdistill.retrain import GATES, Stage
from agentdistill.retrain_stages import build_stages


class Tracking(dict):
    """A context that records every key read, however it is read."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads: set[str] = set()

    def get(self, key, default=None):
        self.reads.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.reads.add(key)
        return super().__getitem__(key)


def _producer_check(stages: list[Stage], initial: frozenset[str] = frozenset()) -> list[str]:
    problems, available = [], set(initial)
    for s in stages:
        missing = [k for k in s.needs if k not in available]
        if missing:
            problems.append(f"{s.name} needs {missing}, which no earlier stage provides")
        available.update(s.provides)
    return problems


def test_the_check_itself_catches_an_unproduced_input():
    stages = [Stage("a", run=lambda c: c, provides=("x",)), Stage("b", run=lambda c: c, needs=("x", "y"))]
    assert _producer_check(stages) == ["b needs ['y'], which no earlier stage provides"]


@pytest.fixture
def stages(project_config, registry):
    project_config.serve.quantization = "fp8"
    project_config.eval.eval_set = "holdout"
    project_config.eval.calib_set = "calib"
    registry.insert_eval_set({"id": "es_calib", "name": "calib", "trace_ids": [], "grader": {}})
    invoked: list[list[str]] = []

    def runner(argv: list[str]) -> str:
        """Pretend each command worked, writing the rows the next stage reads."""
        invoked.append(argv)
        if argv[1:3] == ["eval", "run"] and "--eval-set" in argv:
            subject = argv[3]
            registry.start_eval_run("ev_calib", "es_calib", subject, 3)
            registry.write_eval_result("ev_calib", TaskOutcome(
                task_id="t", repeat_idx=0, messages=[], final_text="", n_turns=1, n_tool_calls=0,
                schema_valid=True, diverged=False, divergence=None, replay_stats={}, latency_ms=1,
                completion_tokens_est=1, stop_reason="answered", success=True))
            registry.finish_eval_run("ev_calib", {"success": 1.0})
        return ""

    built = build_stages(project_config, registry, runner=runner)
    return built, invoked


def test_every_retrain_stage_declares_inputs_produced_earlier(stages):
    built, _ = stages
    assert [s.name for s in built] == list(GATES)
    assert _producer_check(built) == []


def test_calibrate_is_fed_the_calibration_eval_run(stages):
    """The regression itself: calibrate must run with `--from-eval <the run the eval stage produced>`."""
    built, invoked = stages
    ctx: dict = {"candidate_adapter": "ad_x"}
    for s in built:
        if s.name in ("eval", "calibrate"):
            ctx = s.run(ctx)
    calls = [a for a in invoked if a[1] == "calibrate"]
    assert calls, "calibrate was never invoked"
    assert calls[0][calls[0].index("--from-eval") + 1] == "ev_calib"
    assert ctx["calib_eval_run"] == "ev_calib"


def test_every_stage_reads_only_what_it_declares(stages):
    """Declarations that drift from the code are worse than none, so the reads are measured, not trusted."""
    built, _ = stages
    ctx = Tracking()
    for s in built:
        ctx.reads.clear()
        ctx = s.run(ctx) or ctx
        if not isinstance(ctx, Tracking):  # a stage that returned a plain dict
            ctx = Tracking(ctx)
        undeclared = ctx.reads - set(s.needs) - set(s.provides)
        assert not undeclared, f"{s.name} reads {sorted(undeclared)} without declaring them"


def test_without_a_calibration_set_the_gate_stops_and_says_why(project_config, registry):
    project_config.eval.calib_set = None
    built = build_stages(project_config, registry, runner=lambda argv: "")
    ctx: dict = {"candidate_adapter": "ad_x"}
    for s in built:
        if s.name in ("eval", "calibrate"):
            ctx = s.run(ctx)
    ok, why = GATES["calibrate"](ctx)
    assert not ok and "eval.calib_set" in why


def test_a_non_usable_verdict_stops_retrain():
    ok, why = GATES["calibrate"]({"calibration": {"ece": 0.01, "auroc": 0.9}, "calibration_verdict": "no_threshold"})
    assert not ok and "no_threshold" in why
