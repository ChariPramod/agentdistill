"""Stages that produce nothing must fail, and must look different from stages that were told to skip.

The rehearsal's one class of error: a stage that wrote nothing was indistinguishable from a stage that wrote
something uninteresting. The calibration returned empty, the cost block skipped silently, and every one of those
exited 0. These tests pin the three outcomes -- wrote, declared skip, silent emptiness -- from the guard itself,
through the CLI commands it wraps, to the GPU-day script's resume markers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from agentdistill.cli import app
from agentdistill.cli_stage import EXIT_EMPTY, StageEmpty, StageOutcome, stage_guard
from agentdistill.eval.harness import TaskOutcome
from tests.conftest import TOKENIZER_DIR

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def out(result) -> str:
    return (result.stdout or "") + (result.stderr or "")


# --------------------------------------------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------------------------------------------


def test_a_written_row_passes(capsys):
    assert stage_guard("eval_base", lambda: StageOutcome(True, "run ev_1 wrote 10 rows")) == 0
    assert "[stage eval_base] ok: run ev_1 wrote 10 rows" in capsys.readouterr().out


def test_a_declared_skip_passes_and_says_skipped(capsys):
    outcome = StageOutcome(False, "", skipped_reason="eval.skip_teacher set")
    assert stage_guard("eval_teach", lambda: outcome, allow_skip=True) == 0
    assert "[stage eval_teach] SKIPPED: eval.skip_teacher set" in capsys.readouterr().out


def test_silent_emptiness_raises():
    with pytest.raises(StageEmpty, match=r"\[stage eval_teach\] wrote no row: teacher backend returned nothing"):
        stage_guard("eval_teach", lambda: StageOutcome(False, "teacher backend returned nothing"))


def test_a_skip_is_only_honoured_where_it_is_allowed():
    """A skip reason on a stage that may not skip is still a hole."""
    with pytest.raises(StageEmpty):
        stage_guard("calibrate", lambda: StageOutcome(False, "nothing", skipped_reason="felt like it"))


def test_skip_and_empty_lines_differ_at_a_glance(capsys):
    stage_guard("s", lambda: StageOutcome(False, "", skipped_reason="x set"), allow_skip=True)
    skipped = capsys.readouterr().out
    with pytest.raises(StageEmpty) as e:
        stage_guard("s", lambda: StageOutcome(False, "x"))
    assert "SKIPPED" in skipped and "SKIPPED" not in str(e.value)
    assert "wrote no row" in str(e.value) and "wrote no row" not in skipped


# --------------------------------------------------------------------------------------------------------------
# gpu_day.sh: markers on exit 0 only
# --------------------------------------------------------------------------------------------------------------


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run_stages(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    script = tmp_path / "day.sh"
    script.write_text(
        "set -euo pipefail\n"
        f'MARKERS="{tmp_path}/markers"; DRY=1; mkdir -p "$MARKERS"\n'
        f'source "{ROOT}/scripts/stage_lib.sh"\n' + body
    )
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, cwd=tmp_path, check=False)


@needs_bash
def test_exit_3_writes_no_marker_and_stops_the_day(tmp_path):
    proc = _run_stages(tmp_path, (
        "first() { echo first ran; }\n"
        "empty() { echo '[stage empty] wrote no row: nothing'; return 3; }\n"
        "after() { echo after ran; }\n"
        "stage first first\nstage empty empty\nstage after after\n"
    ))
    assert proc.returncode == 3
    assert (tmp_path / "markers" / "first.done").exists()
    assert not (tmp_path / "markers" / "empty.done").exists(), "a rerun must retry the empty stage"
    assert "after ran" not in proc.stdout
    assert "WROTE NOTHING (exit 3)" in proc.stderr


@needs_bash
def test_a_declared_skip_writes_the_marker(tmp_path):
    proc = _run_stages(tmp_path, "skip() { echo '[stage skip] SKIPPED: eval.skip_teacher set'; }\nstage skip skip\n")
    assert proc.returncode == 0
    assert (tmp_path / "markers" / "skip.done").exists()


@needs_bash
def test_a_failing_first_command_fails_a_multi_command_stage(tmp_path):
    """`set -e` must hold inside a stage: a stage whose install step fails does not get to run its check."""
    proc = _run_stages(tmp_path, "multi() { false; echo reached; }\nstage multi multi\n")
    assert proc.returncode != 0
    assert "reached" not in proc.stdout
    assert not (tmp_path / "markers" / "multi.done").exists()


@needs_bash
def test_the_stage_name_reaches_the_command(tmp_path):
    proc = _run_stages(tmp_path, 'show() { echo "name=$AGENTDISTILL_STAGE"; }\nstage eval_teach show\n')
    assert "name=eval_teach" in proc.stdout


@needs_bash
def test_gpu_day_uses_the_stage_library():
    text = (ROOT / "scripts" / "gpu_day.sh").read_text()
    assert 'source "$(dirname "$0")/stage_lib.sh"' in text
    assert "stage() {" not in text, "one definition of the marker rule, not two"


# --------------------------------------------------------------------------------------------------------------
# the CLI commands
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "--name", "demo"])
    path = tmp_path / "project.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["train"]["base_model"] = str(TOKENIZER_DIR)
    cfg.setdefault("cascade", {})["min_turns"] = 60
    cfg.setdefault("eval", {})["eval_set"] = "holdout"
    path.write_text(yaml.safe_dump(cfg))
    return tmp_path


def _registry():
    from agentdistill.config import ProjectConfig
    from agentdistill.registry.base import Registry

    return Registry.from_config(ProjectConfig.load("project.yaml"))


def _set_config(project: Path, section: str, key: str, value) -> None:
    path = project / "project.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault(section, {})[key] = value
    path.write_text(yaml.safe_dump(cfg))


def test_skip_teacher_is_a_declared_skip(project):
    _set_config(project, "eval", "skip_teacher", True)
    result = runner.invoke(app, ["eval", "run", "teacher"])
    assert result.exit_code == 0, out(result)
    assert "SKIPPED: eval.skip_teacher set" in out(result)


def test_without_skip_teacher_a_missing_teacher_is_an_error_not_a_skip(project):
    result = runner.invoke(app, ["eval", "run", "teacher"])
    assert result.exit_code != 0
    assert "SKIPPED" not in out(result)


def test_the_report_distinguishes_a_skipped_teacher_from_a_missing_one(project_config, registry):
    from agentdistill.report.assemble import assemble

    project_config.eval.eval_set = "holdout"
    missing = assemble(registry, project_config)
    assert any("no eval run for `teacher`" in w for w in missing.warnings)

    project_config.eval.skip_teacher = True
    skipped = assemble(registry, project_config)
    assert any("skipped by configuration" in w for w in skipped.warnings)
    assert not any("no eval run for `teacher`" in w for w in skipped.warnings)


# ---- calibrate --------------------------------------------------------------------------------------------------


def _seed_logprob_run(n_tasks: int, turns_per_task: int) -> str:
    reg = _registry()
    reg.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                        "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})
    reg.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft", "config": {},
                             "status": "succeeded", "started_at": "2026-09-18T00:00:00+00:00"})
    reg.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "student", "version": 1,
                        "base_model": "m", "path": "/tmp/a"})
    reg.insert_eval_set({"id": "es_calib", "name": "calib", "trace_ids": [], "grader": {}})
    reg.start_eval_run("ev_calib", "es_calib", "student", 1)
    for t in range(n_tasks):
        messages = [{"role": "user", "content": f"task {t}"}]
        messages += [{"role": "assistant", "content": f"turn {k}"} for k in range(turns_per_task)]
        reg.write_eval_result("ev_calib", TaskOutcome(
            task_id=f"t{t}", repeat_idx=0, messages=messages, final_text="", n_turns=turns_per_task,
            n_tool_calls=0, schema_valid=True, diverged=False, divergence=None, replay_stats={}, latency_ms=1,
            completion_tokens_est=10, stop_reason="answered", success=t % 2 == 0,
        ))
    reg.finish_eval_run("ev_calib", {"success": 0.5})
    return "ev_calib"


@pytest.fixture
def synthetic_labels(monkeypatch):
    """Replace turn labelling with synthetic turns whose `mean_logprob` does or does not predict the label."""
    state = {"informative": True, "all_bad": False}

    def label_rollouts(rollouts, teacher_by_task):
        rng = np.random.default_rng(0)
        records = []
        for r in rollouts:
            for i, m in enumerate(r["messages"]):
                if m["role"] != "assistant":
                    continue
                good = False if state["all_bad"] else bool(rng.random() < 0.5)
                records.append({"task_id": r["task_id"], "turn_index": i, "good": good, "how": "synthetic",
                                "message": m, "prefix": []})
        return records, {"n": len(records), "mix": {"synthetic": len(records)}, "weak_share": 0.0}

    def attach_features(records, cluster_priors=None):
        from agentdistill.config import CascadeConfig

        rng = np.random.default_rng(1)
        for r in records:
            # Every configured feature present, most of them empty -- the shape a turn with no tool call has.
            r["features"] = dict.fromkeys(CascadeConfig().features, float("nan"))
            if state["informative"]:
                signal = 1.5 if r["good"] else -1.5
                r["features"].update(mean_logprob=signal + rng.normal(0, 1), min_logprob=rng.normal(0, 1))
            else:
                # Constant, not noise: noise on a 72-turn holdout lands anywhere from 0.35 to 0.65 AUROC by
                # chance, and a test of "at chance" should not itself be a coin flip.
                r["features"].update(mean_logprob=-1.0, min_logprob=-2.0)
        return len(records)

    monkeypatch.setattr("agentdistill.cascade.labels.label_rollouts", label_rollouts)
    monkeypatch.setattr("agentdistill.cascade.labels.attach_features", attach_features)
    return state


def test_calibrate_below_min_turns_exits_3_with_the_count(project, synthetic_labels):
    run_id = _seed_logprob_run(n_tasks=10, turns_per_task=3)
    result = runner.invoke(app, ["calibrate", "student", "--from-eval", run_id])
    assert result.exit_code == EXIT_EMPTY, out(result)
    assert "wrote no row" in out(result)
    assert "30 labelled turns" in out(result) and "cascade.min_turns=60" in out(result)
    from agentdistill.report.registry_views import calibration_for

    assert calibration_for(_registry(), "ad1") is None


def test_calibrate_writes_a_row_the_report_reads(project, synthetic_labels):
    run_id = _seed_logprob_run(n_tasks=40, turns_per_task=6)
    result = runner.invoke(app, ["calibrate", "student", "--from-eval", run_id])
    assert result.exit_code == 0, out(result)
    assert "[stage calibrate] ok: calibration cal_" in out(result)

    from agentdistill.report.registry_views import calibration_for

    row = calibration_for(_registry(), "ad1")
    assert row is not None, "calibrate must write the row the report and gateway read"
    assert row["verdict"] in ("usable", "no_threshold", "unreliable")
    assert row["holdout_metrics"]["auroc"] > 0.55
    assert row["report"]["n_turns"] == 240
    assert row["reliability_bins"]


def test_an_uninformative_gate_still_writes_a_row_with_that_verdict(project, synthetic_labels):
    """A gate at chance is a result, not a hole: the row says `uninformative` and the gateway refuses it."""
    synthetic_labels["informative"] = False
    run_id = _seed_logprob_run(n_tasks=40, turns_per_task=6)
    result = runner.invoke(app, ["calibrate", "student", "--from-eval", run_id])
    assert result.exit_code == 0, out(result)

    from agentdistill.report.registry_views import calibration_for

    row = calibration_for(_registry(), "ad1")
    assert row["verdict"] == "uninformative"
    assert row["threshold"] == 1.0
    assert "every turn" in out(result)


def test_single_class_labels_are_a_measurement_not_a_hole(project, synthetic_labels):
    """Every turn wrong -- what a random tiny model produces -- is the gate being unable to discriminate
    anything. That is recorded as `uninformative`, not treated as a stage that wrote nothing."""
    synthetic_labels["all_bad"] = True
    run_id = _seed_logprob_run(n_tasks=40, turns_per_task=6)
    result = runner.invoke(app, ["calibrate", "student", "--from-eval", run_id])
    assert result.exit_code == 0, out(result)

    from agentdistill.report.registry_views import calibration_for

    row = calibration_for(_registry(), "ad1")
    assert row["verdict"] == "uninformative"
    assert row["holdout_metrics"]["n"] == 240 and row["holdout_metrics"]["positive_rate"] == 0.0
    assert row["threshold"] == 1.0


# ---- the verdict, end to end --------------------------------------------------------------------------------


def test_gate_verdict_thresholds():
    from agentdistill.cascade.calibrate import CalibrationResult, gate_verdict

    def result(auroc, ece=0.01, n=500):
        return CalibrationResult(feature_order=["x"], holdout={"auroc": auroc, "ece": ece}, in_sample={},
                                 reliability_bins=[], n_turns=n, n_tasks=10)

    assert gate_verdict(result(0.52), True) == "uninformative"
    assert gate_verdict(result(float("nan")), True) == "uninformative"
    assert gate_verdict(result(0.58), True) == "unreliable", "above chance but below the useful floor"
    assert gate_verdict(result(0.8, ece=0.2), True) == "unreliable"
    assert gate_verdict(result(0.8), False) == "no_threshold"
    assert gate_verdict(result(0.8), True) == "usable"


def _calibration_row(registry, verdict: str, tmp_path: Path) -> str:
    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": "2026-09-18T00:00:00+00:00"})
    registry.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "student", "version": 1,
                             "base_model": "m", "path": "/tmp/a", "status": "prod"})
    registry.insert_eval_set({"id": "es_x", "name": "x", "trace_ids": [], "grader": {}})
    registry.start_eval_run("ev_x", "es_x", "student", 1)
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    (cal_dir / "calibration.json").write_text(json.dumps({"usable": True, "feature_order": ["mean_logprob"]}))
    (cal_dir / "model.pkl").write_bytes(__import__("pickle").dumps("a model"))
    return registry.insert_calibration({
        "adapter_id": "ad1", "eval_run_id": "ev_x", "feature_order": ["mean_logprob"], "model_path": str(cal_dir),
        "threshold": 0.7, "holdout_metrics": {"auroc": 0.52, "ece": 0.02}, "verdict": verdict,
        "report": {"predicted_escalation_rate": 0.3},
    })


def test_the_gateway_refuses_a_non_usable_calibration(registry, project_config, tmp_path):
    from agentdistill.gateway.state import _load_calibration

    _calibration_row(registry, "uninformative", tmp_path)
    notes: list[str] = []
    assert _load_calibration(project_config, registry, "ad1", notes) is None
    assert any("uninformative" in n and "refusing" in n for n in notes)


def test_verification_stores_a_point_the_report_uses(registry, project_config, tmp_path):
    from agentdistill.cli import _report_threshold_verification
    from agentdistill.report.registry_views import calibration_for

    cal_id = _calibration_row(registry, "usable", tmp_path)
    run = {"id": "ev_cascade", "metrics": {"escalation_rate": 0.35, "success": 0.8, "cascade_threshold": 0.7,
                                           "wasted_student_tokens_median": 12.0}}
    point = _report_threshold_verification(registry, run, "cascade:student:auto")
    assert point is not None and point["run_id"] == "ev_cascade"

    row = calibration_for(registry, "ad1")
    assert row["id"] == cal_id
    assert row["verified"] == [point]

    # A remeasurement at the same threshold supersedes rather than accumulates.
    run["metrics"]["escalation_rate"] = 0.4
    _report_threshold_verification(registry, run, "cascade:student:auto")
    assert [p["escalation_rate"] for p in calibration_for(registry, "ad1")["verified"]] == [0.4]


def test_verification_of_a_non_cascade_run_measures_nothing(registry, tmp_path):
    from agentdistill.cli import _report_threshold_verification

    _calibration_row(registry, "usable", tmp_path)
    assert _report_threshold_verification(registry, {"id": "ev", "metrics": {}}, "student") is None


def test_a_cascade_run_records_run_level_escalation():
    from agentdistill.eval.runner import cascade_metrics

    rows = [{"n_turns": 4, "escalations": 1, "wasted_student_tokens": 10},
            {"n_turns": 6, "escalations": 3, "wasted_student_tokens": 30}]
    m = cascade_metrics(rows, 0.7)
    assert m["escalation_rate"] == pytest.approx(0.4)
    assert m["escalations"] == 4
    assert m["cascade_threshold"] == 0.7
