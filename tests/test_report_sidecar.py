"""Warning codes, the report.json sidecar, provenance in the reproduce block, and `assert_report`.

The clean rehearsal asserts on the report's data, not its prose, so each warning carries a stable code and the
`report` command writes the data it rendered next to the HTML. These tests also pin the things the first
rehearsal got right (run ids on every row, the unclaimable things printed, lineage through DPO) so a later
refactor cannot quietly drop them.
"""

from __future__ import annotations

import json
from html import escape

import pytest
from sqlalchemy import text

from agentdistill.registry.base import dumps
from agentdistill.report.assemble import ReportData, assemble
from agentdistill.report.html import render
from agentdistill.report.markdown import full_markdown, results_block
from agentdistill.tools.assert_report import check, main
from tests.test_report import seed


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch):
    """Rows record the git state of the tree the tests run in; pin it clean unless a test says otherwise."""
    import agentdistill.provenance as prov

    monkeypatch.setattr(prov, "git_state", lambda cwd=None: {"commit": "abc1234", "dirty": False})


@pytest.fixture
def cfg(project_config):
    from agentdistill.config import TeacherConfig

    project_config.eval.eval_set = "holdout"
    project_config.name = "support-agent"
    project_config.teacher = TeacherConfig(
        model="frontier-v1", provider="anthropic", input_per_mtok=3.0, output_per_mtok=15.0
    )
    return project_config


def codes(r: ReportData) -> set[str]:
    assert len(r.warning_codes) == len(r.warnings), "every warning needs exactly one code"
    return set(r.warning_codes)


def set_metrics(registry, run_id: str, **extra) -> None:
    run = registry.get_eval_run(run_id)
    metrics = {**(run.get("metrics") or {}), **extra}
    with registry.engine.begin() as conn:
        conn.execute(text("UPDATE eval_runs SET metrics = :m WHERE id = :i"), {"m": dumps(metrics), "i": run_id})


# --------------------------------------------------------------------------------------------------------------
# warning codes
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"with_calibration": False}, "no_calibration"),
        ({"with_teacher": False}, "no_teacher_run"),
        ({"with_teacher": False}, "no_run_found"),
        ({"with_throughput": False}, "no_throughput"),
        ({}, "quantization_missing"),
    ],
)
def test_each_unclaimable_thing_has_a_code_and_is_printed(registry, cfg, kwargs, code):
    """The report names what it cannot claim, in both renderers, under a code tooling can assert on."""
    r = assemble(seed(registry, **kwargs), cfg, tag_glob="gpu-day")
    assert code in codes(r)
    message = r.warnings[r.warning_codes.index(code)]
    assert message in results_block(r)
    assert escape(message) in render(r)


def test_missing_pricing_and_teacher_config_have_codes(registry, cfg):
    seed(registry, with_quantized=True)
    cfg.teacher.input_per_mtok = cfg.teacher.output_per_mtok = None
    assert "no_pricing" in codes(assemble(registry, cfg, tag_glob="gpu-day"))
    cfg.teacher = None
    assert "no_teacher_config" in codes(assemble(registry, cfg, tag_glob="gpu-day"))


def test_a_skipped_teacher_is_coded_differently_from_a_missing_one(registry, cfg):
    cfg.eval.skip_teacher = True
    r = assemble(seed(registry, with_teacher=False), cfg, tag_glob="gpu-day")
    assert "teacher_skipped" in codes(r)
    assert "no_run_found" not in codes(r)


def test_tiny_mode_and_no_eval_set_are_coded(registry, cfg, tmp_path):
    cfg.source_path = tmp_path / "project.tiny.yaml"
    cfg.eval.eval_set = None
    r = assemble(registry, cfg)
    assert codes(r) == {"tiny_mode", "no_eval_set"}


def test_an_uninformative_gate_is_coded_and_says_so(registry, cfg):
    seed(registry, with_quantized=True)
    with registry.engine.begin() as conn:
        conn.execute(text("UPDATE calibrations SET verdict = 'uninformative'"))
    r = assemble(registry, cfg, tag_glob="gpu-day")
    assert "gate_not_usable" in codes(r)
    assert "uninformative" in r.warnings[r.warning_codes.index("gate_not_usable")]
    assert r.cascade["escalate_everything"] is True


def test_a_degenerate_gate_has_its_own_code_and_renders_an_undefined_auroc(registry, cfg):
    """All-one-class labels are a data problem, coded apart from an uninformative gate so tiny mode can allow it
    while the GPU day forbids it. AUROC is None there, and must render rather than crash."""
    seed(registry, with_quantized=True)
    reason = "every one of 60 labelled turns is bad; AUROC is undefined"
    with registry.engine.begin() as conn:
        conn.execute(text("UPDATE calibrations SET verdict = 'degenerate_labels', auroc = NULL, "
                          "holdout_metrics = :h, report = :r"),
                     {"h": dumps({"auroc": None, "ece": 0.02, "brier": 0.1, "n": 60}),
                      "r": dumps({"verdict_reason": reason})})
    r = assemble(registry, cfg, tag_glob="gpu-day")
    assert "gate_degenerate" in codes(r)
    assert "gate_not_usable" not in codes(r)
    message = r.warnings[r.warning_codes.index("gate_degenerate")]
    assert "degenerate labels" in message and reason in message
    assert r.calibration["verdict_reason"] == reason
    assert r.cascade["escalate_everything"] is True
    block, html = results_block(r), render(r)
    assert "AUROC n/a" in block and "AUROC n/a" in html
    assert block.count(reason) >= 2, "the gate line prints the reason as well as the warning"
    assert escape(reason) in html.split("<h2>Gate</h2>")[1]
    assert "AUROC nan" not in block and "AUROC nan" not in html


def test_a_quantized_artifact_never_evaluated_keeps_its_own_code(registry, cfg):
    seed(registry)
    registry.insert_adapter({"id": "ad1q", "training_run_id": "tr1", "name": "support-fp8", "version": 2,
                             "base_model": "m", "path": "/tmp/ad1q", "quantization": "fp8",
                             "parent_adapter_id": "ad1"})
    r = assemble(registry, cfg, tag_glob="gpu-day")
    assert "quantized_unevaluated" in codes(r)
    assert "quantization_missing" not in codes(r)


def test_no_quantization_configured_means_no_quantization_warning(registry, cfg):
    cfg.serve.quantization = None
    r = assemble(seed(registry), cfg, tag_glob="gpu-day")
    assert r.quantization == {}
    assert "quantization_missing" not in codes(r)


# --------------------------------------------------------------------------------------------------------------
# replay teacher
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pricing", [True, False])
def test_a_replay_teacher_is_always_disclosed(registry, cfg, pricing):
    """A replay stub is a perfect teacher by construction. Its numbers never appear without saying so."""
    seed(registry, with_quantized=True)
    set_metrics(registry, "ev_teacher", teacher_backend="replay", prompt_tokens_median=1200,
                completion_tokens_median=90)
    if not pricing:
        cfg.teacher.input_per_mtok = cfg.teacher.output_per_mtok = None
    r = assemble(registry, cfg, tag_glob="gpu-day")
    assert "replay_teacher" in codes(r)
    assert "replay stub" in r.warnings[r.warning_codes.index("replay_teacher")]
    if pricing:
        # The stub's fixed token counts drive the cost block: 1200 in at $3/Mtok + 90 out at $15/Mtok.
        assert r.cost["teacher_cost_per_task"] == pytest.approx((1200 * 3.0 + 90 * 15.0) / 1e6)
        assert "no_prompt_tokens" not in codes(r)


def test_a_real_teacher_is_not_flagged_as_replay(registry, cfg):
    r = assemble(seed(registry, with_quantized=True), cfg, tag_glob="gpu-day")
    assert "replay_teacher" not in codes(r)


# --------------------------------------------------------------------------------------------------------------
# provenance in the reproduce block
# --------------------------------------------------------------------------------------------------------------


def set_provenance(registry, table: str, row_id: str, prov: dict | None) -> None:
    with registry.engine.begin() as conn:
        conn.execute(text(f"UPDATE {table} SET provenance = :p WHERE id = :i"),
                     {"p": dumps(prov) if prov is not None else None, "i": row_id})


def test_reproduce_block_prints_commit_and_config_beneath_each_command(registry, cfg):
    seed(registry, with_quantized=True)
    set_provenance(registry, "training_runs", "tr1", {
        "command": "agentdistill train sft support", "commit": "deadbeef", "dirty": False,
        "config_path": "project.yaml", "config_hash": "0123456789ab",
    })
    r = assemble(registry, cfg, tag_glob="gpu-day")
    entry = next(c for c in r.commands if c["id"] == "tr1")
    assert entry["provenance"]["commit"] == "deadbeef"

    md = full_markdown(r)
    block = md[md.index("## How to reproduce"):]
    lines = block.splitlines()
    i = lines.index("agentdistill train sft support")
    assert lines[i + 1] == "# commit deadbeef config project.yaml@0123456789ab"
    assert "# commit deadbeef config project.yaml@0123456789ab" in render(r)
    assert "dirty_tree" not in codes(r)


def test_rows_without_provenance_print_the_bare_command(registry, cfg):
    seed(registry, with_quantized=True)
    set_provenance(registry, "training_runs", "tr1", None)
    r = assemble(registry, cfg, tag_glob="gpu-day")
    entry = next(c for c in r.commands if c["id"] == "tr1")
    assert entry["provenance"] is None
    lines = full_markdown(r).splitlines()
    i = lines.index("agentdistill train sft support")
    assert not lines[i + 1].startswith("# commit")


def test_a_dirty_tree_is_flagged(registry, cfg):
    seed(registry, with_quantized=True)
    set_provenance(registry, "training_runs", "tr1", {
        "command": "agentdistill train sft support", "commit": "deadbeef", "dirty": True,
        "config_path": "project.yaml", "config_hash": "0123456789ab",
    })
    r = assemble(registry, cfg, tag_glob="gpu-day")
    assert "dirty_tree" in codes(r)
    assert r.warnings[r.warning_codes.index("dirty_tree")].startswith("1 run(s) were recorded from a dirty")
    assert "# commit deadbeef (dirty) config project.yaml@0123456789ab" in full_markdown(r)


# --------------------------------------------------------------------------------------------------------------
# section 1 pins
# --------------------------------------------------------------------------------------------------------------


def test_every_rendered_subject_row_carries_its_run_id(registry, cfg):
    r = assemble(seed(registry, with_quantized=True), cfg, tag_glob="gpu-day")
    html = render(r)
    for name, s in r.subjects.items():
        assert s["run_id"], name
        assert f"`{s['run_id']}`" in results_block(r)
        assert f"<code>{s['run_id']}</code>" in html


def test_lineage_chains_through_dpo_to_the_original_adapter(registry, cfg):
    from agentdistill.registry.base import utcnow
    from agentdistill.report.registry_views import lineage

    seed(registry)
    for n, parent in ((2, "ad1"), (3, "ad2")):
        registry.insert_training_run({"id": f"tr{n}", "dataset_id": "ds1", "base_model": "m", "method": "dpo",
                                      "parent_adapter_id": parent, "config": {}, "status": "succeeded",
                                      "started_at": utcnow(), "command": f"agentdistill train dpo round {n}"})
        registry.insert_adapter({"id": f"ad{n}", "training_run_id": f"tr{n}", "name": "support-dpo",
                                 "version": n, "base_model": "m", "path": f"/tmp/ad{n}",
                                 "parent_adapter_id": parent})
    lin = lineage(registry, "ad3")
    assert lin["training_run"]["method"] == "dpo"
    assert [p["id"] for p in lin["parents"]] == ["ad2", "ad1"]


# --------------------------------------------------------------------------------------------------------------
# the sidecar
# --------------------------------------------------------------------------------------------------------------


def test_the_sidecar_round_trips_nan():
    r = ReportData(generated_at="t", project="p", eval_set="e", calibration={"auroc": float("nan")})
    r.warn("no_calibration", "x")
    loaded = json.loads(json.dumps(r.to_dict(), sort_keys=True, indent=2))
    assert loaded["calibration"]["auroc"] != loaded["calibration"]["auroc"]
    assert loaded["warning_codes"] == ["no_calibration"]


@pytest.mark.parametrize("fmt", ["html", "md"])
def test_report_writes_the_sidecar_even_when_it_exits_3(tmp_path, monkeypatch, fmt):
    from typer.testing import CliRunner

    from agentdistill.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["init", "--name", "demo"]).exit_code == 0
    result = runner.invoke(app, ["report", "--format", fmt])
    assert result.exit_code == 3
    data = json.loads((tmp_path / "reports" / "report.json").read_text())
    assert data["subjects"] == {}
    assert len(data["warning_codes"]) == len(data["warnings"])
    assert "no_student" in data["warning_codes"]


# --------------------------------------------------------------------------------------------------------------
# assert_report
# --------------------------------------------------------------------------------------------------------------


def tiny_report(registry, cfg, tmp_path) -> dict:
    cfg.source_path = tmp_path / "project.tiny.yaml"
    seed(registry, with_quantized=True)
    set_metrics(registry, "ev_teacher", teacher_backend="replay")
    with registry.engine.begin() as conn:
        conn.execute(text("UPDATE calibrations SET verdict = 'uninformative'"))
    return json.loads(json.dumps(assemble(registry, cfg, tag_glob="gpu-day").to_dict()))


ALLOW = ["tiny mode", "replay stub", "uninformative"]
SECTIONS = ["calibration", "cascade", "cost", "quantization"]


def test_a_complete_tiny_report_passes(registry, cfg, tmp_path):
    data = tiny_report(registry, cfg, tmp_path)
    assert set(data["warning_codes"]) == {"tiny_mode", "replay_teacher", "gate_not_usable"}
    failures = check(data, ["base", "student", "teacher"], SECTIONS,
                     forbid=["no run found", "no calibration"], allow=ALLOW)
    assert failures == []


def test_an_unexpected_warning_fails_when_allow_is_given(registry, cfg, tmp_path):
    data = tiny_report(registry, cfg, tmp_path)
    data["warnings"].append("the teacher eval did not record prompt-token usage")
    data["warning_codes"].append("no_prompt_tokens")
    assert check(data, allow=ALLOW) == [
        "unexpected warning [no_prompt_tokens]: the teacher eval did not record prompt-token usage"
    ]
    assert check(data) == [], "without --allow-warning, unforbidden warnings are tolerated"


def test_forbidden_warnings_match_by_code_or_prose():
    data = {"warnings": ["no eval run for `teacher` on `holdout`"], "warning_codes": ["no_run_found"]}
    assert check(data, forbid=["no run found"])
    assert check(data, forbid=["no_run_found"])
    assert check(data, forbid=["eval run for `teacher`"])
    assert not check(data, forbid=["no calibration"])


def test_missing_subjects_and_empty_sections_fail():
    data = {"subjects": {"base": {"run_id": "ev1"}, "student": {"run_id": None}},
            "calibration": {"holdout": {"auroc": 0.5}}, "cascade": {"verified": [], "threshold": None},
            "cost": {}, "quantization": {"delta_pp": None}}
    failures = check(data, ["base", "student", "teacher"], SECTIONS)
    assert failures == [
        "subject student has no run id",
        "subject teacher is missing",
        "calibration has no verdict",
        "cascade has no verified point",
        "cost has no teacher_cost_per_task",
        "quantization has no delta_pp",
    ]


def test_a_cascade_needs_a_measured_point_even_when_it_escalates_everything():
    """An escalate-everything decision with no harness measurement is an estimate, not a cascade result."""
    point = {"threshold": 1.0, "escalation_rate": 1.0, "success": 0.5, "escalate_everything": True}
    assert check({"cascade": {"verified": [], "escalate_everything": True, "threshold": 1.0}}, [], ["cascade"])
    assert check({"cascade": {"verified": [point], "escalate_everything": True, "threshold": 1.0}},
                 [], ["cascade"]) == []
    assert check({"cascade": {"verified": [point], "threshold": None}}, [], ["cascade"])


def test_main_reads_the_sidecar_next_to_the_html(registry, cfg, tmp_path, capsys):
    data = tiny_report(registry, cfg, tmp_path)
    (tmp_path / "report.json").write_text(json.dumps(data))
    html = tmp_path / "report.html"
    html.write_text("<!doctype html>")
    argv = [str(html), "--require-subjects", "base,student,teacher", "--require-sections", ",".join(SECTIONS),
            "--forbid-warning", "no run found"]
    for a in ALLOW:
        argv += ["--allow-warning", a]
    assert main(argv) == 0
    assert capsys.readouterr().out.startswith("report ok:")
    assert main([str(html), "--allow-warning", "tiny mode", "--allow-warning", "replay stub"]) == 1
    assert capsys.readouterr().out.startswith("unexpected warning [gate_not_usable]")
    assert main([str(tmp_path / "nowhere" / "report.html")]) == 2
