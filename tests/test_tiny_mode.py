"""Tiny mode: the CPU rehearsal.

The rehearsal exists to find the failures that only appear when stages run in sequence -- a registry query that
assumed a field only vLLM populates, a path that exists only after quantization, a report key that is None when
DPO was discarded. Those cost minutes on a laptop and an hour of rented GPU otherwise.

What is tested here is that the rehearsal is a rehearsal: the same stages, the same commands, a config that
actually loads, and a report that says its numbers are worthless.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gpu_day.sh"
TINY_CONFIG = ROOT / "examples" / "support_agent" / "project.tiny.yaml"
REAL_CONFIG = ROOT / "examples" / "support_agent" / "project.yaml"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def dry_run(tmp_path: Path, tiny: bool) -> str:
    env = {
        "AGENTDISTILL_DRY_RUN": "1",
        "AGENTDISTILL_MARKER_DIR": str(tmp_path / "markers"),
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
    }
    if tiny:
        env["AGENTDISTILL_TINY"] = "1"
    proc = subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


# No cleanup fixture: markers go under each test's own tmp_path, so the repo's artifacts/ is never touched.


def test_the_tiny_config_loads():
    from agentdistill.config import ProjectConfig

    cfg = ProjectConfig.load(str(TINY_CONFIG))
    assert cfg.eval.n_per_task == 1
    assert cfg.cascade.k_samples == 1
    assert cfg.serve.quantization == "fp8"


def test_tiny_uses_a_separate_registry_and_artifacts_directory():
    """A rehearsal that wrote into the real registry would leave garbage adapters next to real ones."""
    from agentdistill.config import ProjectConfig

    tiny = ProjectConfig.load(str(TINY_CONFIG))
    real = ProjectConfig.load(str(REAL_CONFIG))
    assert tiny.registry != real.registry
    assert tiny.artifacts != real.artifacts
    assert tiny.reports != real.reports


def test_tiny_keeps_the_same_curation_filters():
    """A rehearsal that skipped filters would not rehearse curation, which is where most surprises live."""
    from agentdistill.config import ProjectConfig

    assert (
        ProjectConfig.load(str(TINY_CONFIG)).curate.filters
        == ProjectConfig.load(str(REAL_CONFIG)).curate.filters
    )


def test_tiny_mode_announces_itself(tmp_path):
    out = dry_run(tmp_path, tiny=True)
    assert "tiny mode: CPU rehearsal, the numbers are not meaningful" in out


def test_tiny_runs_every_stage_the_real_day_runs(tmp_path):
    """Same stages. A rehearsal that skipped the hard ones would rehearse the easy ones."""
    def stages(out: str) -> list[str]:
        return [line.split()[1] for line in out.splitlines() if line.startswith("== ") and "skip" not in line]

    real = stages(dry_run(tmp_path / "real", tiny=False))
    tiny = [s for s in stages(dry_run(tmp_path / "tiny", tiny=True)) if s != "tiny"]
    assert len(real) >= 18, f"the comparison is vacuous; only found {real}"
    assert tiny == real


def test_tiny_uses_the_hf_backend_and_the_tiny_eval_sets(tmp_path):
    out = dry_run(tmp_path, tiny=True)
    assert "--backend hf" in out
    assert "--backend vllm" not in out
    assert "support-holdout-tiny" in out
    assert "support-holdout-v1" not in out


def test_the_real_day_still_uses_vllm(tmp_path):
    out = dry_run(tmp_path, tiny=False)
    assert "--backend vllm" in out
    assert "support-holdout-v1" in out


def test_tiny_points_at_the_tiny_config(tmp_path):
    out = dry_run(tmp_path, tiny=True)
    assert "project.tiny.yaml" in out
    assert "--config examples/support_agent/project.yaml" not in out


def test_the_report_says_its_tiny_numbers_are_worthless(project_config, registry):
    """The one thing a rehearsal must never produce is a number someone quotes."""
    from agentdistill.report.assemble import assemble

    project_config.source_path = Path("examples/support_agent/project.tiny.yaml")
    report = assemble(registry, project_config)
    assert report.tiny
    assert any("TINY MODE" in w for w in report.warnings)
    assert any("not a measurement" in w for w in report.warnings)


def test_a_real_config_is_not_flagged_as_tiny(project_config, registry):
    from agentdistill.report.assemble import assemble

    project_config.source_path = Path("examples/support_agent/project.yaml")
    report = assemble(registry, project_config)
    assert not report.tiny
    assert not any("TINY MODE" in w for w in report.warnings)


def test_the_fake_vllm_is_runnable_as_a_server():
    """Tiny mode's serve_smoke needs it on a socket, not in a TestClient."""
    import tests.fake_vllm as fv

    assert callable(fv.main)
    with pytest.raises(SystemExit):
        fv.main(["--help"])


def test_serve_smoke_uses_the_fake_only_when_asked():
    text = (ROOT / "scripts" / "serve_smoke.sh").read_text()
    assert "AGENTDISTILL_FAKE_VLLM" in text
    assert "rehearses everything except vLLM" in text
    # And the real path is still the default.
    assert "bash scripts/serve_vllm.sh" in text
