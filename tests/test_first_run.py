"""First-run correctness (plan 4.2): the things that only break on a machine that has never run anything.

A development checkout has a warm model cache, a registry with rows in it and a tiny model built weeks ago. A
clean rehearsal has none of those, and each item here is one way it would fail where the checkout does not.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU_DAY = ROOT / "scripts" / "gpu_day.sh"


# --------------------------------------------------------------------------------------------------------------
# model cache
# --------------------------------------------------------------------------------------------------------------


def test_gpu_day_pins_the_model_cache_before_anything_downloads():
    script = GPU_DAY.read_text()
    export = re.search(r'^export HF_HOME="\$\{HF_HOME:-\$PWD/[^"]+\}"$', script, re.MULTILINE)
    assert export, "gpu_day.sh must export HF_HOME, defaulting to a directory inside the repo"
    assert re.search(r'^mkdir -p "\$HF_HOME"$', script, re.MULTILINE)
    # Before the tiny setup (which builds a model) and before the first stage that loads one.
    assert export.start() < script.index("bash scripts/tiny_setup.sh")
    assert export.start() < script.index("ad() {")


# --------------------------------------------------------------------------------------------------------------
# base model revision
# --------------------------------------------------------------------------------------------------------------


def _load_make_tiny_model():
    spec = importlib.util.spec_from_file_location("make_tiny_model", ROOT / "scripts" / "make_tiny_model.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weights_digest(model_dir: Path) -> str:
    files = sorted(model_dir.glob("*.safetensors"))
    assert files, f"no weights written to {model_dir}"
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()


def test_tiny_model_is_pinned_by_its_seed(tmp_path):
    """The tiny model's `base_model_revision`: a local path has no Hub revision, so a fixed seed pins it.

    Two clean builds, with unrelated RNG use in between, must produce byte-identical weights. Otherwise two
    clean rehearsals train different models and nothing downstream is comparable.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    make = _load_make_tiny_model()
    assert isinstance(make.SEED, int)
    make.build(tmp_path / "a")
    torch.rand(1000)  # anything that ran earlier in the process must not move the weights
    make.build(tmp_path / "b")
    assert _weights_digest(tmp_path / "a") == _weights_digest(tmp_path / "b")


def test_tiny_config_points_at_the_locally_built_model():
    """The seed argument above only holds while tiny mode trains the locally built model, not a Hub id."""
    from agentdistill.config import ProjectConfig

    cfg = ProjectConfig.load(str(ROOT / "examples" / "support_agent" / "project.tiny.yaml"))
    assert cfg.train is not None
    assert Path(cfg.base_model).resolve() == (ROOT / "artifacts" / "tiny" / "model").resolve()


def test_real_config_pins_a_hub_base_model():
    """A Hub base model without a revision can be retagged under the GPU day. Shown, not failed, until then."""
    from agentdistill.config import ProjectConfig

    cfg = ProjectConfig.load(str(ROOT / "examples" / "support_agent" / "project.yaml"))
    assert cfg.train is not None
    if Path(cfg.base_model).is_dir():
        pytest.skip(f"base_model is a local path ({cfg.train.base_model}); there is no Hub revision to pin")
    if not cfg.train.base_model_revision:
        pytest.xfail(f"pin before the GPU day: set train.base_model_revision for {cfg.train.base_model}")


def test_revision_is_passed_only_for_the_configured_base(project_config):
    from agentdistill.config import TrainConfig
    from agentdistill.data.dataset import base_revision

    project_config.train = TrainConfig(base_model="org/model")
    assert base_revision(project_config, "org/model") == {}  # unpinned: nothing passed at all

    project_config.train = TrainConfig(base_model="org/model", base_model_revision="abc123")
    assert base_revision(project_config, "org/model") == {"revision": "abc123"}
    # A merged checkpoint is a different model; the base's revision means nothing to it.
    assert base_revision(project_config, "/tmp/merged") == {}


# --------------------------------------------------------------------------------------------------------------
# empty registry
# --------------------------------------------------------------------------------------------------------------


def test_gateway_boots_on_an_empty_registry(project_config, registry):
    """A clean run starts the gateway before anything is promoted. It must come up and say why it is degraded."""
    from agentdistill.gateway.backends import StubBackend
    from agentdistill.gateway.state import load_state

    state = load_state(project_config, registry, student=StubBackend(), teacher=StubBackend())
    assert state.prod_adapter is None
    assert state.canary_adapter is None
    assert state.notes, "an empty registry is a degraded state and /healthz should say so"
    assert any("no adapter is in prod" in n for n in state.notes)
