"""`train sft` refuses a dataset tokenized for a different base model or revision than the one it trains.

A forgotten rebuild after a base-model change produces no error anywhere else: token ids from one vocabulary train
another model, the loss still falls, and the run looks normal. The manifest records the tokenizer identity at
build time and training compares it with the config before any weights load.
"""

from __future__ import annotations

import json

import pytest

from agentdistill.config import TrainConfig
from agentdistill.data.artifact import verify
from agentdistill.data.dataset import build_dataset
from agentdistill.train.sft import TokenizerMismatch, check_dataset_tokenizer, train_sft
from tests.conftest import TOKENIZER_DIR, make_trace

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def _traces(n: int = 3) -> list[dict]:
    return [
        make_trace(f"t{i}", task=f"Where is order {i} for my account, I have been waiting a while now",
                   closing=" ".join(f"Point {j} of case {i} is confirmed as {i * 13 + j}" for j in range(6)))
        for i in range(n)
    ]


def _build(project_config, registry, tokenizer, revision: str | None = SHA):
    project_config.train = TrainConfig(base_model=str(TOKENIZER_DIR), base_model_revision=revision)
    registry.insert_traces(_traces())
    return build_dataset(_traces(), project_config, name="demo", version=1, registry=registry, tokenizer=tokenizer)


def test_the_manifest_records_tokenizer_id_and_revision(project_config, registry, tokenizer):
    built = _build(project_config, registry, tokenizer)
    m = json.loads(built.artifact.manifest_path.read_text())
    assert m["tokenizer"] == project_config.base_model == str(TOKENIZER_DIR)
    assert m["base_model_revision"] == SHA
    ok, detail = verify(built.artifact.path)
    assert ok, detail


def test_an_unpinned_build_records_the_absence_and_a_pinned_rebuild_is_new(project_config, registry, tokenizer):
    """Unpinned hashes are unchanged by this field, and a pinned rebuild of identical samples is a new dataset
    rather than a reuse of the unpinned one -- which would carry the old manifest and fail the check forever."""
    unpinned = _build(project_config, registry, tokenizer, revision=None)
    m = json.loads(unpinned.artifact.manifest_path.read_text())
    assert "base_model_revision" in m and m["base_model_revision"] is None
    assert verify(unpinned.artifact.path)[0]

    project_config.train = TrainConfig(base_model=str(TOKENIZER_DIR), base_model_revision=SHA)
    pinned = build_dataset(_traces(), project_config, name="demo", version=2, registry=registry,
                           tokenizer=tokenizer)
    assert not pinned.reused
    assert pinned.artifact.content_hash != unpinned.artifact.content_hash


def test_a_matching_config_passes(project_config, registry, tokenizer):
    built = _build(project_config, registry, tokenizer)
    assert check_dataset_tokenizer(project_config.train_config(), built.artifact.path) == []


def test_a_changed_base_model_fails_with_both_values(project_config, registry, tokenizer):
    built = _build(project_config, registry, tokenizer)
    cfg = {**project_config.train_config(), "base_model": "org/other-model"}
    with pytest.raises(TokenizerMismatch) as e:
        check_dataset_tokenizer(cfg, built.artifact.path)
    msg = str(e.value)
    assert str(TOKENIZER_DIR) in msg and "org/other-model" in msg
    assert "agentdistill curate" in msg


def test_a_changed_revision_fails_with_both_values(project_config, registry, tokenizer):
    built = _build(project_config, registry, tokenizer)
    cfg = {**project_config.train_config(), "base_model_revision": OTHER_SHA}
    with pytest.raises(TokenizerMismatch) as e:
        check_dataset_tokenizer(cfg, built.artifact.path)
    assert SHA in str(e.value) and OTHER_SHA in str(e.value)


def test_train_sft_refuses_before_loading_anything(project_config, registry, tokenizer, tmp_path):
    """The check runs first, so a mismatch costs nothing -- not even the training extra."""
    built = _build(project_config, registry, tokenizer)
    cfg = {**project_config.train_config(), "base_model_revision": OTHER_SHA}
    with pytest.raises(TokenizerMismatch):
        train_sft(cfg, built.artifact.path, tmp_path / "adapter")
    assert not (tmp_path / "adapter").exists()


def _legacy(built) -> None:
    """Rewrite the manifest as it was before tokenizer identity was recorded."""
    path = built.artifact.manifest_path
    m = json.loads(path.read_text())
    m.pop("base_model_revision")
    path.write_text(json.dumps(m))


def test_a_legacy_manifest_fails_when_the_config_pins_a_revision(project_config, registry, tokenizer):
    built = _build(project_config, registry, tokenizer, revision=None)
    _legacy(built)
    project_config.train = TrainConfig(base_model=str(TOKENIZER_DIR), base_model_revision=SHA)
    with pytest.raises(TokenizerMismatch, match="manifest predates tokenizer recording") as e:
        check_dataset_tokenizer(project_config.train_config(), built.artifact.path)
    assert SHA in str(e.value)


def test_a_legacy_manifest_only_warns_when_unpinned(project_config, registry, tokenizer):
    built = _build(project_config, registry, tokenizer, revision=None)
    _legacy(built)
    warnings = check_dataset_tokenizer(project_config.train_config(), built.artifact.path)
    assert len(warnings) == 1 and "predates tokenizer recording" in warnings[0]


def test_a_legacy_manifest_for_another_model_still_fails(project_config, registry, tokenizer):
    """The model id was recorded from the first schema on, so it is checked even on a legacy manifest: this is
    exactly the registry dataset tokenized by the fixture tokenizer meeting a real base model."""
    built = _build(project_config, registry, tokenizer, revision=None)
    _legacy(built)
    cfg = {**project_config.train_config(), "base_model": "org/real-model"}
    with pytest.raises(TokenizerMismatch, match="org/real-model"):
        check_dataset_tokenizer(cfg, built.artifact.path)

