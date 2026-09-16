"""Dataset artifacts: parquet, manifest, content hash.

Datasets are immutable. A change in filters, in the tokenizer, or in `max_seq_len` produces a new version rather
than mutating an existing one, because every training run and every eval result is reported against a dataset
hash. If a dataset could change under a hash, none of those numbers would mean anything.

The content hash is computed over the *sorted* sample hashes, so it does not depend on the order samples happened
to be written in, and over the manifest fields that determine content (tokenizer, max_seq_len, filter config).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from agentdistill.data.build import Sample

SCHEMA = pa.schema(
    [
        pa.field("input_ids", pa.list_(pa.int32())),
        pa.field("labels", pa.list_(pa.int32())),
        pa.field("n_tokens", pa.int32()),
        pa.field("n_target_tokens", pa.int32()),
        pa.field("trace_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("sample_hash", pa.string()),
        pa.field("meta", pa.string()),
    ]
)

MANIFEST_NAME = "manifest.json"
DATA_NAME = "data.parquet"


@dataclass
class DatasetArtifact:
    path: Path
    content_hash: str
    n_samples: int
    n_tokens: int
    n_target_tokens: int
    manifest: dict[str, Any]

    @property
    def parquet_path(self) -> Path:
        return self.path / DATA_NAME

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_NAME


def compute_content_hash(sample_hashes: list[str], manifest_core: dict[str, Any]) -> str:
    """sha256 over the sorted sample hashes plus the manifest fields that determine content."""
    h = hashlib.sha256()
    h.update(json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode())
    h.update(b"\x00")
    for sh in sorted(sample_hashes):
        h.update(sh.encode())
        h.update(b"\x00")
    return h.hexdigest()


def manifest_core(
    tokenizer: str,
    tokenizer_revision: str | None,
    max_seq_len: int,
    filter_config: dict,
    kind: str,
    target: str,
) -> dict[str, Any]:
    """Only fields that change the samples belong here; timestamps and paths must not, or the hash would differ
    on every run."""
    return {
        "kind": kind,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "max_seq_len": max_seq_len,
        "target": target,
        "filter_config": filter_config,
        "schema_version": 1,
    }


def write_dataset(
    samples: list[Sample],
    path: str | Path,
    *,
    name: str,
    version: int,
    kind: str = "sft",
    tokenizer: str = "",
    tokenizer_revision: str | None = None,
    max_seq_len: int = 8192,
    filter_config: dict | None = None,
    target: str = "all_assistant",
    extra: dict | None = None,
) -> DatasetArtifact:
    """Write parquet + manifest and return the artifact. Refuses to overwrite a different dataset at the same path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    sample_hashes = [s.hash() for s in samples]
    core = manifest_core(tokenizer, tokenizer_revision, max_seq_len, filter_config or {}, kind, target)
    content_hash = compute_content_hash(sample_hashes, core)

    existing = out / MANIFEST_NAME
    if existing.exists():
        prior = json.loads(existing.read_text())
        if prior.get("content_hash") != content_hash:
            raise FileExistsError(
                f"{out} already holds dataset {prior.get('name')} v{prior.get('version')} with content hash "
                f"{prior.get('content_hash', '')[:12]}, which differs from the {content_hash[:12]} being written. "
                f"Datasets are immutable; write a new version instead."
            )

    table = pa.Table.from_pydict(
        {
            "input_ids": [s.input_ids for s in samples],
            "labels": [s.labels for s in samples],
            "n_tokens": [s.n_tokens for s in samples],
            "n_target_tokens": [s.n_target_tokens for s in samples],
            "trace_id": [s.trace_id for s in samples],
            "kind": [s.kind for s in samples],
            "sample_hash": sample_hashes,
            "meta": [json.dumps(s.meta, sort_keys=True, separators=(",", ":")) for s in samples],
        },
        schema=SCHEMA,
    )
    pq.write_table(table, out / DATA_NAME, compression="zstd")

    n_tokens = sum(s.n_tokens for s in samples)
    n_target = sum(s.n_target_tokens for s in samples)
    manifest = {
        **core,
        "name": name,
        "version": version,
        "content_hash": content_hash,
        "n_samples": len(samples),
        "n_tokens": n_tokens,
        "n_target_tokens": n_target,
        "kinds": _count_kinds(samples),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "agentdistill_version": _version(),
        **(extra or {}),
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return DatasetArtifact(
        path=out,
        content_hash=content_hash,
        n_samples=len(samples),
        n_tokens=n_tokens,
        n_target_tokens=n_target,
        manifest=manifest,
    )


def _count_kinds(samples: list[Sample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        out[s.kind] = out.get(s.kind, 0) + 1
    return out


def _version() -> str:
    from agentdistill import __version__

    return __version__


def read_manifest(path: str | Path) -> dict:
    p = Path(path)
    mp = p if p.name == MANIFEST_NAME else p / MANIFEST_NAME
    if not mp.exists():
        raise FileNotFoundError(f"no dataset manifest at {mp}")
    return json.loads(mp.read_text())


def read_samples(path: str | Path) -> pa.Table:
    p = Path(path)
    dp = p if p.suffix == ".parquet" else p / DATA_NAME
    if not dp.exists():
        raise FileNotFoundError(f"no dataset parquet at {dp}")
    return pq.read_table(dp)


def verify(path: str | Path) -> tuple[bool, str]:
    """Recompute the content hash from the parquet and compare it to the manifest.

    This is what catches a dataset directory that was edited by hand, or a partial write.
    """
    manifest = read_manifest(path)
    table = read_samples(path)
    hashes = table.column("sample_hash").to_pylist()
    core_fields = ("kind", "tokenizer", "tokenizer_revision", "max_seq_len", "target", "filter_config",
                   "schema_version")
    core = {k: manifest[k] for k in core_fields}
    recomputed = compute_content_hash(hashes, core)
    if recomputed != manifest["content_hash"]:
        return False, f"content hash mismatch: manifest {manifest['content_hash'][:12]}, recomputed {recomputed[:12]}"
    if len(hashes) != manifest["n_samples"]:
        return False, f"manifest says {manifest['n_samples']} samples, parquet holds {len(hashes)}"
    return True, f"{len(hashes)} samples, hash {recomputed[:12]}"
