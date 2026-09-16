"""Dataset artifacts: hashing, determinism, immutability."""

from __future__ import annotations

import json

import pytest

from agentdistill.data.artifact import (
    compute_content_hash,
    manifest_core,
    read_manifest,
    read_samples,
    verify,
    write_dataset,
)
from agentdistill.data.build import Sample


def _samples(n: int = 3) -> list[Sample]:
    return [
        Sample(input_ids=[1, 2, 3, i], labels=[-100, -100, 3, i], n_target_tokens=2, trace_id=f"t{i}")
        for i in range(n)
    ]


def _write(path, samples, **kw):
    defaults = {"name": "demo", "version": 1, "tokenizer": "fixture", "max_seq_len": 128,
                "filter_config": {"filters": ["outcome"]}}
    return write_dataset(samples, path, **{**defaults, **kw})


def test_roundtrip(tmp_path):
    a = _write(tmp_path / "ds", _samples())
    assert a.n_samples == 3
    table = read_samples(a.path)
    assert table.num_rows == 3
    assert table.column("trace_id").to_pylist() == ["t0", "t1", "t2"]
    manifest = read_manifest(a.path)
    assert manifest["content_hash"] == a.content_hash
    assert manifest["kinds"] == {"trajectory": 3}


def test_hash_is_order_independent(tmp_path):
    samples = _samples()
    a = _write(tmp_path / "a", samples)
    b = _write(tmp_path / "b", list(reversed(samples)))
    assert a.content_hash == b.content_hash


def test_hash_changes_with_content(tmp_path):
    a = _write(tmp_path / "a", _samples(3))
    b = _write(tmp_path / "b", _samples(4))
    assert a.content_hash != b.content_hash


@pytest.mark.parametrize("field", ["tokenizer", "max_seq_len", "kind", "target"])
def test_hash_changes_with_content_determining_manifest_fields(tmp_path, field):
    overrides = {"tokenizer": "other", "max_seq_len": 999, "kind": "dpo", "target": "last_turn"}
    a = _write(tmp_path / "a", _samples())
    b = _write(tmp_path / "b", _samples(), **{field: overrides[field]})
    assert a.content_hash != b.content_hash, f"{field} determines content and must be in the hash"


def test_hash_ignores_non_content_fields(tmp_path):
    """Editing the GPU price must not invalidate a dataset."""
    a = _write(tmp_path / "a", _samples())
    b = _write(tmp_path / "b", _samples(), extra={"unrelated": "value"})
    assert a.content_hash == b.content_hash


def test_filter_config_is_in_the_hash(tmp_path):
    a = _write(tmp_path / "a", _samples(), filter_config={"filters": ["outcome"]})
    b = _write(tmp_path / "b", _samples(), filter_config={"filters": ["outcome", "pii"]})
    assert a.content_hash != b.content_hash


def test_rewriting_identical_content_is_allowed(tmp_path):
    _write(tmp_path / "ds", _samples())
    _write(tmp_path / "ds", _samples())  # must not raise


def test_overwriting_with_different_content_is_refused(tmp_path):
    _write(tmp_path / "ds", _samples(3))
    with pytest.raises(FileExistsError, match="immutable"):
        _write(tmp_path / "ds", _samples(4))


def test_verify_detects_tampering(tmp_path):
    a = _write(tmp_path / "ds", _samples())
    ok, detail = verify(a.path)
    assert ok, detail

    manifest = json.loads(a.manifest_path.read_text())
    manifest["n_samples"] = 99
    a.manifest_path.write_text(json.dumps(manifest))
    ok, detail = verify(a.path)
    assert not ok
    assert "99" in detail


def test_verify_detects_hash_mismatch(tmp_path):
    a = _write(tmp_path / "ds", _samples())
    manifest = json.loads(a.manifest_path.read_text())
    manifest["content_hash"] = "0" * 64
    a.manifest_path.write_text(json.dumps(manifest))
    ok, detail = verify(a.path)
    assert not ok
    assert "mismatch" in detail


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "nothing")
    with pytest.raises(FileNotFoundError):
        read_samples(tmp_path / "nothing")


def test_compute_content_hash_is_pure():
    core = manifest_core("tok", None, 128, {"filters": []}, "sft", "all_assistant")
    assert compute_content_hash(["b", "a"], core) == compute_content_hash(["a", "b"], core)
    assert compute_content_hash(["a"], core) != compute_content_hash(["a", "b"], core)


def test_token_totals_are_recorded(tmp_path):
    samples = _samples(3)
    a = _write(tmp_path / "ds", samples)
    assert a.n_tokens == sum(s.n_tokens for s in samples)
    assert a.n_target_tokens == sum(s.n_target_tokens for s in samples)
