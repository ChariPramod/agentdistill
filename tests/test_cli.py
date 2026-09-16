"""CLI contract: the commands a user actually types, end to end on a temp project."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from agentdistill.cli import app
from tests.conftest import TOKENIZER_DIR, make_trace

runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temp project directory with traces on disk, cwd set to it."""
    monkeypatch.chdir(tmp_path)
    traces = [
        make_trace(
            f"t{i}",
            task=f"Where is order {i} for my account, I have been waiting some days now",
            closing=" ".join(f"Point {j} of case {i} is confirmed as {i * 17 + j}" for j in range(6)),
        )
        for i in range(8)
    ]
    (tmp_path / "traces.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    (tmp_path / "evals.jsonl").write_text(json.dumps(traces[0]) + "\n")
    return tmp_path


def run(*args):
    return runner.invoke(app, list(args))


def out(result) -> str:
    """Errors go to stderr; CliRunner reports the streams separately, so assertions look at both."""
    return (result.stdout or "") + (result.stderr or "")


def test_version():
    result = run("--version")
    assert result.exit_code == 0
    assert "0.1.0" in out(result)


def test_init_writes_config_and_registry(project):
    result = run("init", "--name", "demo")
    assert result.exit_code == 0, out(result)
    assert (project / "project.yaml").exists()
    assert (project / ".agentdistill" / "registry.db").exists()
    assert "name: demo" in (project / "project.yaml").read_text()


def test_init_refuses_to_clobber(project):
    run("init")
    result = run("init")
    assert result.exit_code == 1
    assert "already exists" in out(result)


def test_init_force_overwrites(project):
    run("init")
    assert run("init", "--force").exit_code == 0


def _init(project):
    run("init", "--name", "demo")
    cfg = (project / "project.yaml").read_text()
    cfg = cfg.replace("base_model: <org>/<model-8b-instruct>", f"base_model: {TOKENIZER_DIR}")
    cfg = cfg.replace("max_seq_len: 8192", "max_seq_len: 2048")
    cfg = cfg.replace("clusters: 32", "clusters: 3").replace("cap_per_cluster: 400", "cap_per_cluster: 50")
    (project / "project.yaml").write_text(cfg)


def test_missing_config_names_the_fix(project):
    result = run("curate")
    assert result.exit_code == 1
    assert "agentdistill init" in out(result)


def test_ingest_then_curate(project):
    _init(project)
    result = run("ingest", "jsonl", "traces.jsonl")
    assert result.exit_code == 0, out(result)
    assert "added 8" in out(result)

    result = run("curate")
    assert result.exit_code == 0, out(result)
    assert "curation report" in out(result)
    assert "dataset" in out(result)
    reports = list((project / "reports").glob("curation-*.md"))
    assert len(reports) == 1
    assert "## What each filter dropped" in reports[0].read_text()


def test_curate_is_idempotent(project):
    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    run("curate")
    result = run("curate")
    assert result.exit_code == 0, out(result)
    assert "identical to" in out(result)

    listed = run("dataset", "list")
    assert out(listed).count("demo") == 1


def test_ingest_duplicate_traces_are_skipped(project):
    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    result = run("ingest", "jsonl", "traces.jsonl")
    assert "added 0" in out(result)
    assert "skipped 8" in out(result)


def test_curate_without_traces_is_a_clear_error(project):
    _init(project)
    result = run("curate")
    assert result.exit_code == 1
    assert "no traces" in out(result)


def test_dry_run_writes_nothing(project):
    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    result = run("curate", "--dry-run")
    assert result.exit_code == 0
    assert "dry run" in out(result)
    assert not (project / "reports").exists()


def test_evalset_add_then_decontaminate(project):
    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    result = run("evalset", "add", "holdout", "evals.jsonl")
    assert result.exit_code == 0, out(result)
    assert "8 tasks" not in out(result) and "1 tasks" in out(result)

    listed = run("evalset", "list")
    assert "holdout" in out(listed) and "frozen" in out(listed)

    result = run("curate")
    assert result.exit_code == 0
    report = next((project / "reports").glob("curation-*.md")).read_text()
    assert "decontaminate" in report


def test_evalset_is_frozen_against_replacement(project):
    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    run("evalset", "add", "holdout", "evals.jsonl")
    result = run("evalset", "add", "holdout", "evals.jsonl")
    assert result.exit_code == 1
    assert "already exists" in out(result)


def test_dataset_verify_and_inspect(project):
    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    run("curate")
    path = str(next((project / "artifacts" / "datasets").iterdir()))

    result = run("dataset", "verify", path)
    assert result.exit_code == 0 and "ok" in out(result)

    result = run("dataset", "inspect", path, "--row", "0")
    assert result.exit_code == 0, out(result)
    assert "trained on" in out(result)


def test_dataset_inspect_out_of_range(project):
    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    run("curate")
    path = str(next((project / "artifacts" / "datasets").iterdir()))
    result = run("dataset", "inspect", path, "--row", "999")
    assert result.exit_code == 1
    assert "out of range" in out(result)


def test_base_check_passes_on_the_fixture_tokenizer(project):
    result = run("base-check", str(TOKENIZER_DIR))
    assert result.exit_code == 0, out(result)
    assert "usable as a base model" in out(result)
    assert "prefix_stable" in out(result)


def test_base_check_json_output(project):
    result = run("base-check", str(TOKENIZER_DIR), "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert {c["name"] for c in payload["checks"]} >= {"has_template", "accepts_tools", "prefix_stable"}


def test_base_check_on_a_missing_model(project):
    result = run("base-check", "definitely/not-a-real-model-xyz")
    assert result.exit_code == 1
    assert "could not load a tokenizer" in out(result)


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (("train", "sft", "ds"), "milestone 2"),
        (("eval", "run", "student"), "milestone 3"),
        (("train", "onpolicy", "a"), "milestone 4"),
        (("calibrate", "a"), "milestone 5"),
        (("serve",), "milestone 6"),
        (("retrain",), "milestone 7"),
        (("report",), "milestone 8"),
        (("ingest", "gateway"), "milestone 6"),
    ],
)
def test_unbuilt_commands_say_so_clearly(project, args, fragment):
    """An unimplemented command must exit 2 with a pointer, never a stack trace or a silent no-op."""
    result = run(*args)
    assert result.exit_code == 2
    assert "not built yet" in out(result)
    assert fragment in out(result)
