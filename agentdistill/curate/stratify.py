"""Task clustering, per-cluster caps, and the coverage report.

Sparse clusters are where the student will fail first. Curation caps the dominant clusters so the student does not
learn only the easy majority, and the report names the thin ones so the eval and the router floor can protect them.

Embeddings are pluggable. The `hash` provider needs no network and no API key, which keeps the pipeline testable
and deterministic; it clusters by token overlap rather than meaning, and the curation report says so.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Protocol

import numpy as np

from agentdistill.curate.decontaminate import normalize_text, task_text


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def name(self) -> str: ...


class HashEmbedder:
    """Deterministic hashed bag-of-words with L2 normalization. No network, no API key, reproducible across runs."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hash-{self.dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in normalize_text(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
                # A signed hash keeps unrelated tokens from piling up in the same bucket with the same sign.
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy optional dependency

        self.model_name = model
        self._model = SentenceTransformer(model)

    @property
    def name(self) -> str:
        return self.model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float64)


def make_embedder(cfg: Any) -> Embedder:
    if cfg.provider == "hash":
        return HashEmbedder(dim=cfg.dim)
    if cfg.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.model or "sentence-transformers/all-MiniLM-L6-v2")
    if cfg.provider == "openai":
        raise NotImplementedError(
            "the openai embedding provider is not wired yet; use `hash` for local runs or "
            "`sentence-transformers` with the local-embeddings extra"
        )
    raise ValueError(f"unknown embeddings provider {cfg.provider!r}")


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """k-means++ init, Lloyd iterations. Returns (labels, centroids).

    Implemented here rather than pulled from sklearn so that cluster assignment is byte-identical across sklearn
    versions; a dataset's content hash depends on it.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(1, min(k, n))
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
    centroids[0] = X[rng.integers(n)]
    closest = ((X - centroids[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = closest.sum()
        probs = np.full(n, 1.0 / n) if total <= 0 else closest / total
        centroids[j] = X[rng.choice(n, p=probs)]
        closest = np.minimum(closest, ((X - centroids[j]) ** 2).sum(axis=1))

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return labels, centroids


def assign_clusters(traces: list[dict], embedder: Embedder, k: int, seed: int = 0) -> tuple[dict[str, int], np.ndarray]:
    """Cluster traces by task input. Returns (trace_id -> cluster, centroids)."""
    if not traces:
        return {}, np.zeros((0, 0))
    texts = [task_text(t.get("task_input")) for t in traces]
    X = embedder.embed(texts)
    labels, centroids = kmeans(X, k, seed=seed)
    return {t["id"]: int(lbl) for t, lbl in zip(traces, labels, strict=True)}, centroids


def cap_per_cluster(traces: list[dict], assignments: dict[str, int], cap: int) -> set[str]:
    """Return ids to drop so that no cluster exceeds `cap`.

    Keeps the first `cap` in input order, which is stable (registry order), so the hash is reproducible.
    """
    seen: Counter[int] = Counter()
    drop: set[str] = set()
    for t in traces:
        c = assignments.get(t["id"], -1)
        seen[c] += 1
        if seen[c] > cap:
            drop.add(t["id"])
    return drop


def tool_sequence(trace: dict) -> tuple[str, ...]:
    return tuple(
        c["function"]["name"] for m in trace["messages"] for c in (m.get("tool_calls") or [])
    )


def coverage(
    traces_before: list[dict], traces_after: list[dict], assignments: dict[str, int], sparse_threshold: int = 10
) -> dict:
    """Cluster sizes before and after capping, the share of thin clusters, and the top tool sequences per cluster."""
    before: Counter[int] = Counter(assignments.get(t["id"], -1) for t in traces_before)
    after: Counter[int] = Counter(assignments.get(t["id"], -1) for t in traces_after)
    seqs: dict[int, Counter[tuple[str, ...]]] = {}
    for t in traces_after:
        c = assignments.get(t["id"], -1)
        seqs.setdefault(c, Counter())[tool_sequence(t)] += 1

    clusters = []
    for c in sorted(set(before) | set(after)):
        top = seqs.get(c, Counter()).most_common(3)
        clusters.append(
            {
                "cluster": c,
                "before": before.get(c, 0),
                "after": after.get(c, 0),
                "sparse": after.get(c, 0) < sparse_threshold,
                "top_tool_sequences": [
                    {"sequence": list(seq) or ["(no tool calls)"], "n": n} for seq, n in top
                ],
            }
        )
    n_sparse = sum(1 for c in clusters if c["sparse"])
    return {
        "n_clusters": len(clusters),
        "n_sparse": n_sparse,
        "sparse_share": n_sparse / len(clusters) if clusters else 0.0,
        "sparse_threshold": sparse_threshold,
        "clusters": clusters,
    }
