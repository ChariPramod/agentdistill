"""The retrain loop: the gates, and one full pass through the pipeline.

The gates are what make an unattended weekly retrain acceptable, so each one is tested on its own rather than
only through a pipeline run. A gate exercised only end to end is a gate nobody has checked.

The pipeline test drives all eight stages with a fake runner. Nothing is trained; what is verified is that the
stages run in order, that a failing gate stops the run at that stage, that markers make a rerun resume, and that
`--from` overrides them.
"""

from __future__ import annotations

import pytest

from agentdistill.retrain import (
    MAX_HOLDOUT_ECE,
    MIN_NEW_REQUESTS,
    STAGE_ORDER,
    Markers,
    Stage,
    gate_calibrated,
    gate_enough_new_samples,
    gate_enough_new_traffic,
    gate_not_worse_than_prod,
    gate_promotion_checks_green,
    gate_quantization_lossless_enough,
    gate_round_promoted,
    gate_training_converged,
    run_pipeline,
)

# --------------------------------------------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------------------------------------------


def test_too_little_new_traffic_stops_the_run():
    """The normal weekly outcome. Retraining on twelve requests is not a retrain."""
    ok, why = gate_enough_new_traffic({"n_requests": 12})
    assert not ok
    assert "12 graded requests" in why and "normal outcome" in why


def test_enough_traffic_continues():
    assert gate_enough_new_traffic({"n_requests": MIN_NEW_REQUESTS})[0]


def test_a_missing_count_reads_as_zero_rather_than_passing():
    assert not gate_enough_new_traffic({})[0]


def test_curation_that_rejected_everything_stops_the_run():
    ok, why = gate_enough_new_samples({"n_new_samples": 3})
    assert not ok
    assert "nothing to learn" in why


def test_a_diverged_training_run_stops_the_run():
    assert not gate_training_converged({"eval_loss": float("nan")})[0]
    assert not gate_training_converged({"eval_loss": float("inf")})[0]
    assert not gate_training_converged({"eval_loss": None})[0]
    assert not gate_training_converged({"eval_loss": "n/a"})[0]


def test_a_finite_loss_continues_and_says_it_is_only_a_proxy():
    ok, why = gate_training_converged({"eval_loss": 0.42})
    assert ok
    assert "proxy" in why


def test_only_a_promoted_round_continues():
    assert gate_round_promoted({"round_decision": "promote"})[0]
    for decision in ("discard", "retry", None):
        assert not gate_round_promoted({"round_decision": decision})[0]


def test_the_eval_gate_reads_the_interval_not_the_point_estimate():
    """A candidate 0.5 pp below prod with an eight-point interval has been measured badly, not shown at parity."""
    badly_measured = {"success": {"delta": -0.005, "ci95": (-0.04, 0.03)}}
    assert not gate_not_worse_than_prod({"comparison": badly_measured})[0]

    well_measured = {"success": {"delta": -0.002, "ci95": (-0.008, 0.004)}}
    assert gate_not_worse_than_prod({"comparison": well_measured})[0]


def test_no_comparison_is_a_stop_not_a_pass():
    ok, why = gate_not_worse_than_prod({})
    assert not ok
    assert "nothing to promote on" in why


def test_calibration_needs_both_ece_and_auroc():
    """ECE alone passes a calibrator that outputs the base rate for every turn: well calibrated, and useless."""
    base_rate_predictor = {"calibration": {"ece": 0.01, "auroc": 0.50}}
    ok, why = gate_calibrated(base_rate_predictor)
    assert not ok
    assert "cannot separate" in why

    assert gate_calibrated({"calibration": {"ece": MAX_HOLDOUT_ECE, "auroc": 0.72}})[0]
    assert not gate_calibrated({"calibration": {"ece": 0.20, "auroc": 0.72}})[0]
    assert not gate_calibrated({})[0]


def test_quantization_that_costs_too_much_stops_the_run():
    assert not gate_quantization_lossless_enough({"quantization_drop_pp": 5.0})[0]
    assert gate_quantization_lossless_enough({"quantization_drop_pp": 1.0})[0]


def test_quantization_scoring_higher_is_not_a_stop():
    assert gate_quantization_lossless_enough({"quantization_drop_pp": -3.0})[0]


def test_no_quantization_configured_is_not_a_stop():
    assert gate_quantization_lossless_enough({"quantization_drop_pp": None})[0]


def test_promotion_needs_every_lifecycle_check():
    green = {"a": {"ok": True}, "b": {"ok": True}}
    assert gate_promotion_checks_green({"promotion_checks": green})[0]

    ok, why = gate_promotion_checks_green({"promotion_checks": {"a": {"ok": True}, "calibrated": {"ok": False}}})
    assert not ok
    assert "calibrated" in why


def test_no_checks_at_all_is_a_stop():
    assert not gate_promotion_checks_green({"promotion_checks": {}})[0]


# --------------------------------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------------------------------


def make_stages(fail_at: str | None = None, log: list[str] | None = None) -> list[Stage]:
    def runner(name: str):
        def run(ctx: dict) -> dict:
            (log if log is not None else []).append(name)
            ctx.setdefault("ran", []).append(name)
            return ctx

        return run

    def gate(name: str):
        def check(ctx: dict) -> tuple[bool, str]:
            return (name != fail_at, f"{name} gate")

        return check

    return [
        Stage(name=n, run=runner(n), gate=gate(n), describe=lambda c, n=n: f"would do {n}")
        for n in STAGE_ORDER
    ]


def test_every_stage_runs_in_order():
    result = run_pipeline(make_stages(), {}, log=lambda _: None)
    assert result.ok
    assert result.ran == list(STAGE_ORDER)
    assert result.ctx["ran"] == list(STAGE_ORDER)


def test_a_failing_gate_stops_at_that_stage_and_runs_nothing_after():
    result = run_pipeline(make_stages(fail_at="eval"), {}, log=lambda _: None)
    assert not result.ok
    assert result.stopped_at == "eval"
    assert result.ctx["ran"] == ["ingest_gateway", "curate", "train_sft_continue", "onpolicy", "eval"]
    assert "calibrate" not in result.ran


def test_the_first_stages_gate_stops_the_whole_run():
    """The common case: not enough new traffic, so nothing else should happen."""
    result = run_pipeline(make_stages(fail_at="ingest_gateway"), {}, log=lambda _: None)
    assert result.stopped_at == "ingest_gateway"
    assert result.ctx["ran"] == ["ingest_gateway"]


def test_dry_run_describes_without_running(tmp_path):
    lines: list[str] = []
    ran: list[str] = []
    result = run_pipeline(make_stages(log=ran), {}, log=lines.append, dry_run=True)
    assert ran == []
    assert result.ran == list(STAGE_ORDER)
    assert all("would run" in line for line in lines)
    assert "would do eval" in " ".join(lines)


def test_markers_make_a_rerun_resume(tmp_path):
    markers = Markers(root=tmp_path, retrain_id="r1")
    first = run_pipeline(make_stages(fail_at="calibrate"), {}, markers=markers, log=lambda _: None)
    assert first.stopped_at == "calibrate"

    ran: list[str] = []
    second = run_pipeline(make_stages(log=ran), {}, markers=markers, log=lambda _: None)
    # The four stages before calibrate are marked done; only calibrate onward run again.
    assert ran == ["calibrate", "quantize", "promote_canary"]
    assert second.ok


def test_a_second_retrain_id_starts_fresh(tmp_path):
    run_pipeline(make_stages(), {}, markers=Markers(root=tmp_path, retrain_id="r1"), log=lambda _: None)
    ran: list[str] = []
    run_pipeline(
        make_stages(log=ran), {}, markers=Markers(root=tmp_path, retrain_id="r2"), log=lambda _: None
    )
    assert ran == list(STAGE_ORDER)


def test_from_overrides_the_markers(tmp_path):
    """Resuming means rerunning the stage that failed; a marker from a previous attempt must not skip it."""
    markers = Markers(root=tmp_path, retrain_id="r1")
    run_pipeline(make_stages(), {}, markers=markers, log=lambda _: None)

    ran: list[str] = []
    run_pipeline(make_stages(log=ran), {}, markers=markers, start_from="eval", log=lambda _: None)
    assert ran == ["eval", "calibrate", "quantize", "promote_canary"]


def test_an_unknown_stage_name_is_refused():
    with pytest.raises(ValueError, match="unknown stage"):
        run_pipeline(make_stages(), {}, start_from="trian_sft", log=lambda _: None)


def test_the_stage_order_matches_the_gates():
    from agentdistill.retrain import GATES

    assert list(STAGE_ORDER) == list(GATES)


# --------------------------------------------------------------------------------------------------------------
# the real stages, against a fake runner
# --------------------------------------------------------------------------------------------------------------


def test_the_real_stages_invoke_commands_that_exist(project_config, registry):
    """A typo in a subcommand would surface on a weekly cron at 3am."""
    from typer.main import get_command

    from agentdistill.cli import app
    from agentdistill.retrain_stages import build_stages

    root = get_command(app)
    available = set()
    for name, cmd in root.commands.items():  # type: ignore[attr-defined]
        available.add(name)
        available.update(f"{name} {sub}" for sub in getattr(cmd, "commands", {}))

    invoked: list[list[str]] = []

    def fake_runner(argv: list[str]) -> str:
        invoked.append(argv)
        return ""

    project_config.serve.quantization = "fp8"
    stages = build_stages(project_config, registry, runner=fake_runner)
    ctx: dict = {}
    for stage in stages:
        ctx = stage.run(ctx)

    assert invoked, "no commands were invoked"
    for argv in invoked:
        assert argv[0] == "agentdistill"
        head = argv[1]
        candidate = f"{head} {argv[2]}" if f"{head} {argv[2]}" in available else head
        assert candidate in available, f"unknown command: {' '.join(argv[:3])}"


def test_the_real_stages_accept_the_flags_they_pass(project_config, registry):
    """Every flag the pipeline passes must exist on the command it passes it to."""
    import shlex
    import subprocess
    import sys

    from agentdistill.retrain_stages import build_stages

    invoked: list[list[str]] = []
    project_config.serve.quantization = "fp8"
    stages = build_stages(project_config, registry, runner=lambda argv: invoked.append(argv) or "")
    ctx: dict = {}
    for stage in stages:
        ctx = stage.run(ctx)

    for argv in invoked:
        flags = {a for a in argv if a.startswith("--")}
        help_cmd = [sys.executable, "-m", "agentdistill.cli", *argv[1:3], "--help"]
        proc = subprocess.run(help_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:  # a one-word command; retry without the second token
            help_cmd = [sys.executable, "-m", "agentdistill.cli", argv[1], "--help"]
            proc = subprocess.run(help_cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"could not get help for {shlex.join(argv)}"
        for flag in flags:
            assert flag in proc.stdout, f"{flag} is not a flag of `{' '.join(argv[1:3])}`"


def test_the_real_stages_supply_every_required_argument(project_config, registry):
    """A missing positional is invisible to a flag check and fatal at 3am on a cron.

    Each invocation is parsed by the real CLI with a runner that refuses to do any work, so a command missing a
    required argument fails here rather than after the previous stage has already trained something.
    """
    import subprocess
    import sys

    from agentdistill.retrain_stages import build_stages

    invoked: list[list[str]] = []
    project_config.serve.quantization = "fp8"
    stages = build_stages(project_config, registry, runner=lambda argv: invoked.append(argv) or "")
    ctx: dict = {}
    for stage in stages:
        ctx = stage.run(ctx)

    for argv in invoked:
        proc = subprocess.run(
            [sys.executable, "-m", "agentdistill.cli", *argv[1:], "--help"],
            capture_output=True, text=True, check=False,
        )
        # `--help` parses arguments first, so a missing required one exits 2 with "Missing argument".
        assert "Missing argument" not in proc.stderr + proc.stdout, (
            f"`{' '.join(argv[1:])}` is missing a required argument"
        )


def test_a_stage_command_failing_is_distinct_from_a_gate_refusing(project_config, registry):
    """A gate refusing is a normal outcome; a command exiting nonzero is not, and they must not look alike."""
    from agentdistill.retrain_stages import RetrainStageFailed, build_stages

    def angry_runner(argv: list[str]) -> str:
        raise RetrainStageFailed("vllm ran out of memory")

    stages = build_stages(project_config, registry, runner=angry_runner)
    with pytest.raises(RetrainStageFailed):
        run_pipeline(stages, {}, log=lambda _: None)
