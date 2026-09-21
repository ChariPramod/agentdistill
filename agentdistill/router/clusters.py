"""Placing a live request in a curation cluster.

The router's context is the cluster, so the gateway has to reproduce curation's clustering on a request it has
never seen: embed the task as posed (the same text curation embedded) with the same embedder, and take the
nearest centroid. The centroids are saved by `curate` next to the dataset they were fitted on.

A missing cluster model is a health state, not a routing decision. `NoClusterModel` assigns `UNASSIGNED` to
every request, and the gateway routes those on the pooled posterior and counts them on /healthz. The floor is
for clusters the student is measurably bad at; applying it to "we do not know which cluster this is" would turn
a missing file into a teacher bill.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

UNASSIGNED = -1


class ClusterAssigner:
    """Nearest-centroid assignment in the embedding space curation clustered in."""

    def __init__(self, centroids: np.ndarray | None, embedder: Any, model_id: str | None = None) -> None:
        self.c: np.ndarray | None = centroids
        self.embed = embedder
        self.model_id = model_id

    @staticmethod
    def task_text(messages: list[dict]) -> str:
        from agentdistill.curate.decontaminate import task_text
        from agentdistill.ingest.normalize import extract_task_input

        return task_text(extract_task_input(messages))

    def assign(self, messages: list[dict]) -> int:
        if self.c is None or len(self.c) == 0:
            return UNASSIGNED
        return self._assign_text(self.task_text(messages))

    def _assign_text(self, text: str) -> int:
        x = self.embed.embed([text])[0]
        return int(np.argmin(((self.c - x) ** 2).sum(axis=1)))

    def describe(self) -> dict:
        if self.c is None or len(self.c) == 0:
            return {"state": "missing", "reason": "cluster model has no centroids"}
        return {"state": "loaded", "id": self.model_id, "k": len(self.c),
                "embedder": getattr(self.embed, "name", None)}


class NoClusterModel(ClusterAssigner):
    """Explicit null object: the gateway loaded no cluster model, and says why."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.c, self.embed, self.model_id = None, None, None

    def assign(self, messages: list[dict]) -> int:
        return UNASSIGNED

    def describe(self) -> dict:
        return {"state": "missing", "reason": self.reason}


def embedder_spec(embeddings_cfg: Any, name: str) -> dict:
    """What it takes to rebuild the embedder the centroids live in. The stored spec, not today's config, is
    authoritative: centroids from a hash embedder are meaningless to a sentence-transformer."""
    return {"provider": embeddings_cfg.provider, "model": embeddings_cfg.model, "dim": embeddings_cfg.dim,
            "name": name}


def save_cluster_model(registry: Any, artifacts_dir: Path, dataset_id: str, centroids: np.ndarray,
                       spec: dict) -> str:
    """Persist centroids beside the dataset they were fitted on, and register them."""
    import uuid

    from sqlalchemy import text

    from agentdistill.registry.base import utcnow

    model_id = f"cm_{uuid.uuid4().hex[:16]}"
    out = Path(artifacts_dir) / "clusters"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model_id}.npy"
    np.save(path, centroids)
    with registry.engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO cluster_models (id, dataset_id, embedder, k, centroids_ref, labels, created_at)
                    VALUES (:id, :ds, :embedder, :k, :ref, NULL, :at)"""),
            {"id": model_id, "ds": dataset_id, "embedder": json.dumps(spec, sort_keys=True),
             "k": len(centroids), "ref": str(path), "at": utcnow()},
        )
    return model_id


def load_cluster_assigner(registry: Any) -> ClusterAssigner:
    """The latest cluster model, or a `NoClusterModel` saying why there is none. Never raises: a gateway that
    cannot place requests still serves them."""
    from sqlalchemy import text

    from agentdistill.curate.stratify import make_embedder

    try:
        with registry.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM cluster_models ORDER BY created_at DESC LIMIT 1")
            ).mappings().first()
    except Exception as e:
        return NoClusterModel(f"could not read cluster_models: {e}")
    if row is None:
        return NoClusterModel("no cluster model in the registry; run `agentdistill curate` with the stratify filter")
    try:
        centroids = np.load(row["centroids_ref"])
    except (OSError, ValueError) as e:
        return NoClusterModel(f"cluster model {row['id']} centroids unreadable at {row['centroids_ref']}: {e}")
    try:
        spec = json.loads(row["embedder"])
        embedder = make_embedder(SimpleNamespace(**{k: spec.get(k) for k in ("provider", "model", "dim")}))
    except Exception as e:  # an embedder that cannot be rebuilt cannot place a request
        return NoClusterModel(f"cluster model {row['id']} embedder could not be rebuilt: {e}")
    return ClusterAssigner(centroids, embedder, model_id=row["id"])
