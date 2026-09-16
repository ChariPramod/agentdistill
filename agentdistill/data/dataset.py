"""Dataset assembly: curated traces -> samples -> parquet artifact -> registry rows.

This is the seam between curation (which decides *which* traces) and training (which consumes *tokens*). It is
also where the template check runs, because a template that cannot be masked must stop the pipeline here rather
than produce a dataset that trains on the wrong tokens.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentdistill.config import ProjectConfig
from agentdistill.data.artifact import DatasetArtifact, compute_content_hash, manifest_core, write_dataset
from agentdistill.data.build import Sample, build_samples_for_trace
from agentdistill.data.template_check import check_template
from agentdistill.registry import Registry, utcnow


@dataclass
class BuildResult:
    artifact: DatasetArtifact
    dataset_id: str
    name: str
    version: int
    n_traces_in: int
    n_traces_used: int
    notes: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    #: True when an identical dataset already existed and was reused rather than written again.
    reused: bool = False

    @property
    def n_samples(self) -> int:
        return self.artifact.n_samples


def load_tokenizer(base_model: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "building a dataset needs a tokenizer; install `agentdistill[tokenizers]` (or `[train]`)"
        ) from e
    return AutoTokenizer.from_pretrained(base_model)


def build_dataset(
    traces: list[dict],
    cfg: ProjectConfig,
    *,
    name: str,
    version: int,
    registry: Registry | None = None,
    tokenizer: Any = None,
    base_model: str | None = None,
    filter_config: dict | None = None,
    kind: str = "sft",
    target: str = "all_assistant",
    report_path: str | None = None,
) -> BuildResult:
    """Tokenize curated traces into a hashed, immutable dataset artifact.

    Raises TemplateError before doing any work if the base model's chat template cannot render tools or cannot be
    masked from offsets.
    """
    model = base_model or (cfg.train.base_model if cfg.train else None)
    if tokenizer is None:
        if not model:
            raise ValueError("no base model: set train.base_model in project.yaml or pass base_model")
        tokenizer = load_tokenizer(model)
    model = str(model or getattr(tokenizer, "name_or_path", "<tokenizer>"))

    check_template(tokenizer, model).raise_if_failed()

    max_seq_len = cfg.dataset.max_seq_len
    samples: list[Sample] = []
    notes: list[str] = []
    skipped: dict[str, str] = {}
    used = 0

    for trace in traces:
        built, note = build_samples_for_trace(
            tokenizer,
            trace,
            max_seq_len=max_seq_len,
            window_turns=cfg.dataset.window_turns,
            windows_for_long=cfg.dataset.windows_for_long_trajectories,
            target=target,  # type: ignore[arg-type]
        )
        if not built:
            skipped[trace["id"]] = note or "produced no samples"
            continue
        used += 1
        if note:
            notes.append(f"{trace['id']}: {note}")
        samples.extend(built)

    if not samples:
        raise ValueError(
            f"no samples were produced from {len(traces)} traces. "
            f"Most likely every trajectory exceeds dataset.max_seq_len={max_seq_len}, "
            f"or curation removed everything -- read the curation report."
        )

    filters = filter_config if filter_config is not None else cfg.curation_fingerprint()
    core = manifest_core(model, _revision(tokenizer), max_seq_len, filters, kind, target)
    content_hash = compute_content_hash([s.hash() for s in samples], core)

    if registry is not None:
        prior = registry.get_dataset_by_hash(content_hash)
        if prior is not None:
            from agentdistill.data.artifact import read_manifest

            return BuildResult(
                artifact=DatasetArtifact(
                    path=Path(prior["path"]),
                    content_hash=content_hash,
                    n_samples=prior["n_samples"],
                    n_tokens=prior["n_tokens"] or 0,
                    n_target_tokens=sum(s.n_target_tokens for s in samples),
                    manifest=read_manifest(prior["path"]) if Path(prior["path"]).exists() else {},
                ),
                dataset_id=prior["id"],
                name=prior["name"],
                version=prior["version"],
                n_traces_in=len(traces),
                n_traces_used=used,
                notes=notes,
                skipped=skipped,
                reused=True,
            )

    out_dir = cfg.artifacts_dir / "datasets" / f"{name}-v{version}"
    artifact = write_dataset(
        samples,
        out_dir,
        name=name,
        version=version,
        kind=kind,
        tokenizer=model,
        tokenizer_revision=_revision(tokenizer),
        max_seq_len=max_seq_len,
        filter_config=filters,
        target=target,
        extra={"n_traces_in": len(traces), "n_traces_used": used, "window_turns": cfg.dataset.window_turns},
    )

    dataset_id = f"ds_{artifact.content_hash[:16]}"
    if registry is not None:
        registry.insert_dataset(
            {
                "id": dataset_id,
                "name": name,
                "version": version,
                "kind": kind,
                "filter_config": artifact.manifest["filter_config"],
                "n_samples": artifact.n_samples,
                "n_tokens": artifact.n_tokens,
                "content_hash": artifact.content_hash,
                "path": str(artifact.path),
                "report_path": report_path,
                "created_at": utcnow(),
            }
        )
        registry.insert_samples(
            [
                {
                    "id": f"sm_{uuid.uuid4().hex[:16]}",
                    "dataset_id": dataset_id,
                    "trace_id": s.trace_id,
                    "kind": s.kind,
                    "n_tokens": s.n_tokens,
                    "n_target_tokens": s.n_target_tokens,
                    "row_idx": i,
                    "sample_hash": s.hash(),
                }
                for i, s in enumerate(samples)
            ]
        )

    if skipped:
        notes.append(f"{len(skipped)} traces produced no samples and were dropped at tokenization")

    return BuildResult(
        artifact=artifact,
        dataset_id=dataset_id,
        name=name,
        version=version,
        n_traces_in=len(traces),
        n_traces_used=used,
        notes=notes,
        skipped=skipped,
    )


def _revision(tokenizer: Any) -> str | None:
    """Pin what we can: the tokenizer's resolved revision if transformers recorded one."""
    for attr in ("_commit_hash", "commit_hash"):
        rev = getattr(tokenizer, attr, None)
        if rev:
            return str(rev)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        rev = init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
        if rev:
            return str(rev)
    return None


def dataset_dir(cfg: ProjectConfig, name: str, version: int) -> Path:
    return cfg.artifacts_dir / "datasets" / f"{name}-v{version}"
