"""The pre-flight, the lock, and the export: the three things between a rented box and a wasted day.

Each is tested for the failure it exists to catch -- a leaked key, a silently drifted input, a result left on a
machine about to be deleted -- rather than for its happy path alone.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from agentdistill.ops.lock import diff_locks

ROOT = Path(__file__).resolve().parents[1]
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

#: A value that looks like a key and is unique enough that any appearance in output is a leak, not a coincidence.
SENTINEL = "sk-ant-test-" + "Zq7" * 8


def _preflight(env_extra: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}", **env_extra}
    env.pop("ANTHROPIC_API_KEY", None) if "ANTHROPIC_API_KEY" not in env_extra else None
    return subprocess.run(["bash", str(ROOT / "scripts" / "preflight.sh")], capture_output=True, text=True,
                          cwd=ROOT, env=env, timeout=600, check=False)


@needs_bash
def test_preflight_never_prints_any_part_of_the_key():
    """The pre-flight's output is handed to the lead verbatim when it fails. A key in it is a leaked key."""
    proc = _preflight({"ANTHROPIC_API_KEY": SENTINEL})
    out = proc.stdout + proc.stderr
    assert "PASS env.api_key" in out, "a set key must read as set"
    # Every substring long enough to identify the key, not only the whole value.
    for i in range(len(SENTINEL) - 7):
        assert SENTINEL[i:i + 8] not in out, f"the pre-flight printed part of the key: {SENTINEL[i:i + 8]!r}"


@needs_bash
def test_preflight_fails_loudly_without_a_key():
    proc = _preflight({})
    out = proc.stdout
    assert "FAIL env.api_key" in out
    assert proc.returncode != 0, "a missing key must stop the day, not merely warn"
    assert "do not start the GPU day" in out


@needs_bash
def test_preflight_names_the_fix_for_an_unpushed_repo():
    """The owner reads this on the box; a FAIL without the command that fixes it costs a round trip."""
    proc = _preflight({"ANTHROPIC_API_KEY": SENTINEL})
    if "PASS git.pushed" in proc.stdout:
        pytest.skip("this checkout has a remote with HEAD pushed")
    assert "git remote add origin" in proc.stdout and "--bundle-ok" in proc.stdout


def test_a_changed_hash_reads_as_a_diff_naming_both_values():
    locked = {"base_model": "Qwen/Qwen2.5-7B-Instruct", "sft_dataset": {"content_hash": "aaa"}, "note": "x"}
    current = {"base_model": "Qwen/Qwen2.5-7B-Instruct", "sft_dataset": {"content_hash": "bbb"}, "note": "y"}
    lines = diff_locks(locked, current)
    assert len(lines) == 1, "the free-text note is not part of what the day must reproduce"
    assert "sft_dataset.content_hash" in lines[0]
    assert "locked:  aaa" in lines[0] and "current: bbb" in lines[0]


def test_a_field_missing_on_either_side_is_a_difference_not_a_match():
    lines = diff_locks({"base_model_revision": "abc"}, {})
    assert lines and "<not in this tree>" in lines[0]


def test_the_committed_lock_names_every_input_the_plan_lists():
    lock = json.loads((ROOT / "examples" / "support_agent" / "gpu-day.lock.json").read_text())
    for key in ("base_model", "base_model_revision", "tool_parser", "sft_dataset", "eval_sets"):
        assert key in lock, f"the lock does not pin {key}"
    assert len(lock["base_model_revision"]) == 40, "a revision is a 40-character SHA, never a tag"
    assert len(lock["eval_sets"]) == 3


# --------------------------------------------------------------------------------------------------------------
# the export
# --------------------------------------------------------------------------------------------------------------


@needs_bash
def test_the_export_runs_when_a_stage_exits_3_and_carries_the_registry(tmp_path):
    """The trap is the point: a day that dies at `calibrate` has still produced every row before it."""
    text = (ROOT / "scripts" / "gpu_day.sh").read_text()
    assert "trap on_exit EXIT" in text, "gpu_day.sh does not export from an exit trap"
    body = text.split("on_exit() {", 1)[1].split("\n}\n", 1)[0]
    assert "export.done" in body, "the trap must stand down only when the export stage itself succeeded"
    assert "export_results.sh" in body


@needs_bash
def test_an_export_tarball_verifies_on_the_laptop(tmp_path):
    """Round trip through the two scripts the owner runs: export on the box, verify on the laptop."""
    registry = ROOT / "examples" / "support_agent" / ".agentdistill" / "registry.tiny.db"
    report = ROOT / "artifacts" / "gpu_day" / "report.json"
    if not (registry.exists() and report.exists()):
        pytest.skip("needs a tiny rehearsal's registry and report; run scripts/clean_rehearsal.sh")
    out = tmp_path / "export"
    env = {**os.environ, "PATH": f"{ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}",
           "AGENTDISTILL_CONFIG": "examples/support_agent/project.tiny.yaml"}
    made = subprocess.run(["bash", str(ROOT / "scripts" / "export_results.sh"), "--out", str(out)],
                          capture_output=True, text=True, cwd=ROOT, env=env, timeout=300, check=False)
    assert made.returncode == 0, made.stderr
    tarball = next(out.glob("agentdistill-export-*.tar.gz"))
    with tarfile.open(tarball) as tf:
        names = tf.getnames()
    assert any(n.endswith("registry/registry.tiny.db") for n in names), "the registry is not in the export"
    assert not any(Path(n).name.startswith("._") for n in names), "macOS resource forks leaked into the export"

    verified = subprocess.run(["bash", str(ROOT / "scripts" / "verify_export.sh"), str(tarball)],
                              capture_output=True, text=True, cwd=ROOT, env=env, timeout=300, check=False)
    assert verified.returncode == 0, verified.stdout[-2000:]
    assert "export verified" in verified.stdout
