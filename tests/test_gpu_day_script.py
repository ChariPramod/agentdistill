"""The GPU-day script.

It cannot be executed here, so what is tested is that it is syntactically valid, that every stage plans, that
the resume markers work, and that every `agentdistill` subcommand it invokes actually exists. A typo in a
subcommand name would otherwise be found on rented hardware.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gpu_day.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def dry_run(tmp_path: Path) -> str:
    """Run the script with every agentdistill call echoed, in a scratch copy of the repo's script."""
    # A private marker directory: the suite must never delete a real session's markers, because a GPU day
    # that loses them redoes work that was paid for.
    env = {
        "AGENTDISTILL_DRY_RUN": "1",
        "AGENTDISTILL_MARKER_DIR": str(tmp_path / "markers"),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# There is no cleanup fixture on purpose. Every run here writes its markers under its own tmp_path, so nothing
# in the repo is touched -- an earlier version deleted artifacts/gpu_day, which is a real session's resume state.


def test_script_is_valid_bash():
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_every_stage_plans(tmp_path):
    out = dry_run(tmp_path)
    expected = [
        "env", "base_check", "sft", "merge", "eval_base", "eval_sft", "eval_teach", "cmp_sft",
        "onpolicy", "eval_r1", "unseen", "logprobs", "calibrate", "cascade_ver", "quantize",
        "eval_quant", "serve_smoke", "report",
    ]
    for stage in expected:
        assert f"== {stage}" in out, f"stage {stage} did not run"
    assert "== done" in out


def test_rerun_skips_completed_stages(tmp_path):
    dry_run(tmp_path)
    second = dry_run(tmp_path)
    assert "== skip sft (done)" in second
    assert "== skip report (done)" in second


def test_removing_a_marker_reruns_only_that_stage(tmp_path):
    dry_run(tmp_path)
    (tmp_path / "markers" / "sft.done").unlink()
    out = dry_run(tmp_path)
    assert "== sft" in out and "== skip sft" not in out
    assert "== skip merge (done)" in out


def test_every_subcommand_invoked_exists(tmp_path):
    """A typo in a subcommand name would surface on rented hardware; catch it here."""
    from typer.main import get_command

    from agentdistill.cli import app

    root = get_command(app)
    groups: dict[str, set[str]] = {}
    available: set[str] = set()
    for name, cmd in root.commands.items():  # type: ignore[attr-defined]
        available.add(name)
        subs = set(getattr(cmd, "commands", {}))
        if subs:
            groups[name] = subs
            available.update(f"{name} {sub}" for sub in subs)

    out = dry_run(tmp_path)
    invoked = set()
    for line in out.splitlines():
        if not line.startswith("agentdistill "):
            continue
        parts = line.split()[1:]
        head = parts[0]
        # Only a known group takes a second word; otherwise the next token is a flag's value, not a subcommand.
        invoked.add(f"{head} {parts[1]}" if head in groups and len(parts) > 1 else head)

    unknown = {c for c in invoked if c not in available}
    assert not unknown, f"script calls commands that do not exist: {sorted(unknown)} (have: {sorted(available)})"


def test_selectors_used_by_the_script_are_real_commands():
    """The `$( ... )` substitutions are what produce the ids the next stage consumes."""
    text = SCRIPT.read_text()
    for selector in ["dataset latest", "adapter latest", "adapter best", "eval latest", "config get"]:
        assert selector in text, f"the script no longer uses {selector!r}"


def test_script_documents_how_to_force_a_stage():
    text = SCRIPT.read_text()
    assert "rm artifacts/gpu_day" in text, "the runbook must say how to rerun one stage"
    assert "AGENTDISTILL_DRY_RUN" in text


def test_no_stage_hardcodes_the_base_model():
    """Settings live in project.yaml; a second copy in the script is a second thing to keep in sync."""
    text = SCRIPT.read_text()
    assert "config get train.base_model" in text
    assert not re.search(r"--base-model\s+\S+/\S+", text)


def test_every_flag_the_script_passes_exists(tmp_path):
    """A flag that does not exist fails the stage, on the box, after the stages before it have already run.

    The subcommand check above does not catch this: `eval run --logprobs` names a command that exists and a
    flag that did not. Four such flags were found this way, including the two the calibration stage depends on.
    """
    import re
    import sys

    from typer.main import get_command

    from agentdistill.cli import app

    root = get_command(app)
    groups = {n: set(getattr(c, "commands", {})) for n, c in root.commands.items()}  # type: ignore[attr-defined]

    out = dry_run(tmp_path)
    missing, checked = [], set()
    for line in out.splitlines():
        if not line.startswith("agentdistill "):
            continue
        parts = line.split()[1:]
        head = parts[0]
        cmd = parts[:2] if head in groups and len(parts) > 1 and parts[1] in groups[head] else parts[:1]
        flags = frozenset(p for p in parts if p.startswith("--"))
        if (tuple(cmd), flags) in checked:
            continue
        checked.add((tuple(cmd), flags))

        help_text = subprocess.run(
            [sys.executable, "-m", "agentdistill.cli", *cmd, "--help"],
            capture_output=True, text=True, check=False, env={**os.environ, "COLUMNS": "200"},
        ).stdout
        # Rich wraps help output, so collapse whitespace before looking for a flag.
        collapsed = re.sub(r"\s+", " ", help_text)
        missing += [f"`{' '.join(cmd)}` has no {f}" for f in sorted(flags) if f not in collapsed]

    assert not missing, "the GPU day script passes flags that do not exist:\n  " + "\n  ".join(missing)


def test_stages_that_load_a_local_model_pin_the_backend():
    """Every stage that loads weights locally must take `--backend "$BACKEND"`, or tiny mode reaches for vLLM.

    Three stages were missed at different times -- `eval run`, then `adapter merge`, then `train onpolicy` --
    and each failed the rehearsal only after the stages before it had already run. `eval run teacher` is
    exempt: the teacher is an API, and it resolves to a LiteLLM client rather than to local weights.
    """
    text = SCRIPT.read_text()
    checked = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("s_") or "{ ad " not in stripped:
            continue
        loads_weights = (
            ("ad eval run" in stripped and "$TEACHER_SUBJECT" not in stripped)
            or "ad adapter merge" in stripped
            or "ad train onpolicy" in stripped
        )
        if not loads_weights:
            continue
        checked += 1
        assert '--backend "$BACKEND"' in stripped, (
            f"this stage loads a model but pins no backend, so tiny mode will reach for vLLM:\n  {stripped}"
        )
    assert checked >= 6, f"expected several model-loading stages, found {checked}"


def test_the_teacher_stage_does_not_pin_a_local_backend():
    """The teacher is an API. A `--backend` there would suggest it loads weights locally."""
    for line in SCRIPT.read_text().splitlines():
        if "$TEACHER_SUBJECT" in line and "ad eval run" in line:
            assert "--backend" not in line
            return
    raise AssertionError("the script no longer evaluates the teacher")


def test_tiny_mode_uses_the_replay_teacher_instead_of_calling_one():
    """A rehearsal must not spend money, and must not skip the teacher either: a skipped teacher row is how the
    cost block and the cascade went dark in the first rehearsal. Tiny mode evaluates `teacher` against the replay
    stub its config declares, and the report discloses that the numbers are structural."""
    from agentdistill.config import ProjectConfig

    text = SCRIPT.read_text()
    assert "recorded}" not in text
    assert text.count('TEACHER_SUBJECT="${TEACHER_SUBJECT:-teacher}"') == 2
    tiny = ProjectConfig.load(str(ROOT / "examples" / "support_agent" / "project.tiny.yaml"))
    assert tiny.teacher is not None and tiny.teacher.backend == "replay"
    assert tiny.teacher.input_per_mtok is not None, "the cost block needs a price, even a structural one"


def test_tiny_mode_no_longer_skips_the_cascade():
    body = SCRIPT.read_text().split("s_cascade_ver() {", 1)[1].split("\n}\n", 1)[0]
    assert "TINY" not in body and "--verify-threshold" in body


def test_every_flag_inside_a_command_substitution_exists():
    """The selectors inside `$( ... )` are what produce the ids the next stage consumes.

    The dry run replaces them with a placeholder rather than echoing them, so the flag check above never sees
    them -- which is how `eval latest --subject-tag` survived. This one reads the script text instead.
    """
    import re
    import sys

    from typer.main import get_command

    from agentdistill.cli import app

    root = get_command(app)
    groups = {n: set(getattr(c, "commands", {})) for n, c in root.commands.items()}  # type: ignore[attr-defined]

    missing, checked = [], 0
    # `cap <label> <command...>`: the label is consumed by the dry-run placeholder, the rest is the command.
    for m in re.finditer(r"\$\(cap\s+\S+\s+([^)]*)\)", SCRIPT.read_text()):
        parts = [p for p in m.group(1).split() if not p.startswith(('"', "$", "}"))]
        if not parts:
            continue
        head = parts[0]
        cmd = parts[:2] if head in groups and len(parts) > 1 and parts[1] in groups[head] else parts[:1]
        flags = {p for p in m.group(1).split() if p.startswith("--")}
        checked += 1
        # A wide terminal: Rich truncates a long flag name to `--verify-thresho…` at the default width, and a
        # truncated flag reads exactly like a missing one.
        help_text = re.sub(r"\s+", " ", subprocess.run(
            [sys.executable, "-m", "agentdistill.cli", *cmd, "--help"],
            capture_output=True, text=True, check=False, env={**os.environ, "COLUMNS": "200"},
        ).stdout)
        if not help_text:
            missing.append(f"`{' '.join(cmd)}` is not a command")
            continue
        missing += [f"`{' '.join(cmd)}` has no {f}" for f in sorted(flags) if f not in help_text]

    assert checked >= 8, f"expected several selectors, found {checked}"
    assert not missing, "selectors inside $( ) use flags that do not exist:\n  " + "\n  ".join(sorted(set(missing)))
