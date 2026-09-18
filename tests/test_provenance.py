"""Provenance: every result row says which command, commit, tree state, and config produced it.

These pin the reproduction block the report prints. A later refactor that drops the commit or stops normalizing
argv[0] would not fail anything else, and the report would quietly become something nobody can paste.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from agentdistill import provenance as prov

KEYS = {"command", "commit", "dirty", "config_path", "config_hash"}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(prov, "_active_config", None)
    prov.git_state.cache_clear()
    yield
    prov.git_state.cache_clear()


@pytest.mark.parametrize(
    "argv0",
    [
        "/Users/someone/.venv/lib/python3.13/site-packages/agentdistill/cli.py",
        "/Users/someone/project/agentdistill/__main__.py",
        "__main__",
        "/usr/local/bin/agentdistill",
        "/Users/someone/.venv/bin/pytest",
    ],
)
def test_known_entrypoints_normalize_to_agentdistill(monkeypatch, argv0):
    monkeypatch.setattr(sys, "argv", [argv0, "eval", "run", "--config", "project.yaml"])
    assert prov.normalized_argv() == ["agentdistill", "eval", "run", "--config", "project.yaml"]


@pytest.mark.parametrize("argv0", ["/usr/bin/python3", "some_script.py", "/opt/wrapper.sh"])
def test_an_unknown_argv0_is_left_untouched(monkeypatch, argv0):
    """The normalization refuses to guess: a wrapper may do work of its own."""
    monkeypatch.setattr(sys, "argv", [argv0, "curate"])
    assert prov.normalized_argv() == [argv0, "curate"]


def test_invocation_still_maps_pytest_and_cli_py(monkeypatch):
    from agentdistill.registry.base import invocation

    monkeypatch.setattr(sys, "argv", ["/x/.venv/bin/pytest", "-x"])
    assert invocation() == "agentdistill -x"
    monkeypatch.setattr(sys, "argv", ["/x/agentdistill/cli.py", "curate"])
    assert invocation() == "agentdistill curate"


def test_config_state_hash_is_stable_and_tracks_content(tmp_path):
    cfg = tmp_path / "project.yaml"
    cfg.write_text("name: demo\n")
    first = prov.config_state(str(cfg))
    assert first == prov.config_state(str(cfg))
    assert first["config_path"] == str(cfg)
    assert len(first["config_hash"]) == 12
    cfg.write_text("name: demo2\n")
    assert prov.config_state(str(cfg))["config_hash"] != first["config_hash"]


def test_config_state_of_a_missing_file_has_no_hash(tmp_path):
    missing = str(tmp_path / "nope.yaml")
    assert prov.config_state(missing) == {"config_path": missing, "config_hash": None}
    assert prov.config_state(None) == {"config_path": None, "config_hash": None}


def test_the_config_path_is_stored_as_given(tmp_path, monkeypatch):
    (tmp_path / "project.yaml").write_text("name: demo\n")
    monkeypatch.chdir(tmp_path)
    state = prov.config_state("project.yaml")
    assert state["config_path"] == "project.yaml"
    assert state["config_hash"] is not None


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_state_reports_commit_and_dirty_flag(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "project.yaml").write_text("name: demo\n")
    _git(repo, "add", "project.yaml")
    _git(repo, "commit", "-q", "-m", "init")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()

    assert prov.git_state(str(repo)) == {"commit": sha, "dirty": False}

    (repo / "project.yaml").write_text("name: edited\n")
    # Cached per process: the edit is not seen until the cache is cleared.
    assert prov.git_state(str(repo))["dirty"] is False
    prov.git_state.cache_clear()
    assert prov.git_state(str(repo)) == {"commit": sha, "dirty": True}

    # provenance() asks the repository that holds the config, not the process cwd.
    p = prov.provenance(str(repo / "project.yaml"))
    assert p["commit"] == sha and p["dirty"] is True


def test_git_state_outside_a_repository_is_none_not_an_error(tmp_path, monkeypatch):
    # An empty ceiling makes git stop at tmp_path even if the temp dir happens to sit inside a checkout.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert prov.git_state(str(tmp_path)) == {"commit": None, "dirty": None}
    prov.git_state.cache_clear()
    assert prov.git_state(str(tmp_path / "does-not-exist")) == {"commit": None, "dirty": None}


def _stored(registry, table: str, row_id: str):
    from sqlalchemy import text

    # Adapters have no command column of their own; their provenance carries it.
    cols = "provenance" if table == "adapters" else "command, provenance"
    with registry.engine.connect() as conn:
        return conn.execute(text(f"SELECT {cols} FROM {table} WHERE id = :id"), {"id": row_id}).first()


def test_registry_rows_store_provenance(registry, tmp_path, monkeypatch):
    """Eval runs, training runs, adapters and calibrations all carry the same parseable provenance."""
    cfg = tmp_path / "project.yaml"
    cfg.write_text("name: demo\n")
    prov.set_active_config(str(cfg))
    expected_hash = prov.config_state(str(cfg))["config_hash"]
    monkeypatch.setattr(sys, "argv", ["/x/agentdistill/cli.py", "eval", "run", "--config", str(cfg)])

    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": "2026-09-18T00:00:00+00:00"})
    registry.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "student", "version": 1,
                             "base_model": "m", "path": "/tmp/a", "status": "prod"})
    registry.insert_eval_set({"id": "es_x", "name": "x", "trace_ids": [], "grader": {}})
    registry.start_eval_run("ev_x", "es_x", "student", 1)
    cal_id = registry.insert_calibration({
        "adapter_id": "ad1", "eval_run_id": "ev_x", "feature_order": ["mean_logprob"], "model_path": "/tmp/c",
        "threshold": 0.7, "holdout_metrics": {"auroc": 0.52}, "verdict": "usable",
    })

    command = f"agentdistill eval run --config {cfg}"
    for table, row_id in [("training_runs", "tr1"), ("adapters", "ad1"), ("eval_runs", "ev_x"),
                          ("calibrations", cal_id)]:
        row = _stored(registry, table, row_id)
        p = json.loads(row.provenance)
        assert set(p) == KEYS, table
        assert p["command"] == command, table
        assert p["config_path"] == str(cfg) and p["config_hash"] == expected_hash, table
        if table != "adapters":
            assert row.command == command, table


def test_an_explicit_command_and_provenance_are_respected(registry):
    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": "2026-09-18T00:00:00+00:00",
                                  "command": "agentdistill train sft demo"})
    row = _stored(registry, "training_runs", "tr1")
    assert row.command == "agentdistill train sft demo"
    assert json.loads(row.provenance)["command"] == "agentdistill train sft demo"

    given = {"command": "c", "commit": "abc", "dirty": False, "config_path": "p.yaml", "config_hash": "0" * 12}
    registry.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "student", "version": 1,
                             "base_model": "m", "path": "/tmp/a", "provenance": given})
    assert json.loads(_stored(registry, "adapters", "ad1").provenance) == given


def test_cli_load_records_the_active_config(tmp_path):
    from agentdistill.cli import _load

    path = tmp_path / "project.yaml"
    path.write_text("name: demo\n")
    _load(str(path))
    assert prov.active_config() == str(path)
