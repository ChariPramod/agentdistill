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
        (("ingest", "otel", "trace.json"), "milestone 1 stretch"),
    ],
)
def test_unbuilt_commands_say_so_clearly(project, args, fragment):
    """An unimplemented command must exit 2 with a pointer, never a stack trace or a silent no-op."""
    result = run(*args)
    assert result.exit_code == 2
    assert "not built yet" in out(result)
    assert fragment in out(result)


def test_retrain_is_implemented(project):
    """`retrain` is built; a dry run against an empty registry prints the plan and changes nothing."""
    _init(project)
    result = run("retrain", "--dry-run")
    combined = out(result)
    assert "not built yet" not in combined
    assert result.exit_code == 0
    assert "would run ingest_gateway" in combined
    assert "would run promote_canary" in combined


def test_report_is_implemented(project):
    """`report` is built; with an empty registry it must still render, with warnings -- and exit 3, because a
    report with no subjects is a pipeline that produced nothing."""
    _init(project)
    result = run("report", "--format", "md")
    combined = out(result)
    assert "not built yet" not in combined
    assert "agentdistill:results:begin" in combined
    assert (project / "reports" / "report.md").exists() or "wrote" in combined
    assert result.exit_code == 3
    assert "[stage report] wrote no row" in combined


def test_report_injects_into_the_readme(project):
    _init(project)
    readme = project / "README.md"
    readme.write_text("# My agent\n\nIntro.\n")
    # Exit 3: an empty registry has no subjects. The injection still happens, and says so.
    assert run("report", "--format", "md", "--inject", str(readme)).exit_code == 3
    once = readme.read_text()
    assert "## Results" in once
    run("report", "--format", "md", "--inject", str(readme))
    assert readme.read_text().count("agentdistill:results:begin") == 1, "injection must be idempotent"


def test_report_rejects_an_unknown_format(project):
    _init(project)
    result = run("report", "--format", "pdf")
    assert result.exit_code == 1
    assert "html or md" in out(result)


def test_adapter_promote_is_implemented(project):
    """`adapter promote` is built; an unknown adapter must be the error, not an unbuilt command."""
    _init(project)
    result = run("adapter", "promote", "no-such-adapter", "--to", "canary")
    combined = out(result)
    assert "not built yet" not in combined
    assert "no adapter" in combined


def test_serve_is_implemented(project):
    """`serve --dry-run` must load and report, not claim to be unbuilt."""
    _init(project)
    result = run("serve", "--dry-run")
    combined = out(result)
    assert "not built yet" not in combined
    assert "dry run" in combined or "gateway" in combined


def test_calibrate_is_implemented(project):
    """`calibrate` is built; it must explain what it needs rather than report itself unbuilt."""
    _init(project)
    result = run("calibrate", "some-adapter")
    combined = out(result)
    assert "not built yet" not in combined
    assert "--from-eval is required" in combined


def test_train_onpolicy_is_implemented(project):
    """`train onpolicy` is built; --dry-run must plan rather than report it unbuilt."""
    _init(project)
    result = run("train", "onpolicy", "some-adapter", "--dry-run")
    combined = out(result)
    assert "not built yet" not in combined
    # No traces ingested in this fixture, so it should say that rather than claim the command is missing.
    assert "no training traces" in combined or "dry run" in combined


def test_eval_run_is_implemented(project):
    """`eval run` is built; it must fail on a missing eval set, not on being unimplemented."""
    _init(project)
    result = run("eval", "run", "base", "--eval-set", "nope")
    assert "not built yet" not in out(result)
    assert result.exit_code == 1


def test_train_sft_is_implemented(project):
    """`train sft` is built; it must fail on a missing dataset rather than on being unimplemented."""
    _init(project)
    result = run("train", "sft", "no-such-dataset")
    assert "not built yet" not in out(result)
    assert result.exit_code == 1
    assert "no dataset named" in out(result)


def test_curate_saves_the_cluster_model_the_gateway_loads(project):
    """Curation used to compute centroids and discard them, so the gateway could never place a request."""
    from agentdistill.config import ProjectConfig
    from agentdistill.registry.base import Registry
    from agentdistill.router.clusters import load_cluster_assigner

    _init(project)
    run("ingest", "jsonl", "traces.jsonl")
    result = run("curate")
    assert result.exit_code == 0, out(result)
    assert "cluster model" in out(result)
    assigner = load_cluster_assigner(Registry.from_config(ProjectConfig.load("project.yaml")))
    assert assigner.describe()["state"] == "loaded"
    assert assigner.assign([{"role": "user", "content": "Where is order 3 for my account"}]) >= 0
