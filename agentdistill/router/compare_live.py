"""Comparing two adapters on live traffic.

Offline eval compares adapters on a frozen set. This compares them on what production actually sent, which is
the only way to catch a distribution shift between the eval set and reality.

Live traffic is not a randomized trial: the router decides which requests reach the student at all, and the
canary split then decides which adapter serves them. The split is deterministic on the request id and
independent of content, so within the student-routed population the two adapters see comparable work. Pairing by
cluster removes the remaining confound -- if the canary happened to draw more of an easy cluster, an unpaired
comparison would credit it for the mix rather than the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compare_live(
    rows: list[dict],
    prod: str,
    canary: str,
    iters: int = 5000,
    seed: int = 0,
    min_per_arm: int = 5,
    min_clusters: int = 3,
) -> dict | None:
    """Paired per-cluster comparison of two adapters. `None` when there is not enough traffic to say anything.

    Returning `None` rather than a wide interval is deliberate: a caller that sees a number tends to act on it,
    and the lifecycle check that reads this should fail with "insufficient live traffic" rather than pass on
    three observations.
    """
    by: dict[Any, dict[str, list[bool]]] = {}
    for r in rows:
        if r.get("arm") != "student" or r.get("fallback"):
            continue
        if r.get("outcome") is None or r.get("cluster_id") is None:
            continue
        adapter = r.get("adapter_id")
        if adapter not in (prod, canary):
            continue
        by.setdefault(r["cluster_id"], {}).setdefault(adapter, []).append(bool(r["outcome"]))

    clusters = sorted(
        c
        for c, d in by.items()
        if len(d.get(prod, [])) >= min_per_arm and len(d.get(canary, [])) >= min_per_arm
    )
    if len(clusters) < min_clusters:
        return None

    diff = np.array([np.mean(by[c][canary]) - np.mean(by[c][prod]) for c in clusters], dtype=float)
    # Weight by the smaller arm: a cluster's difference is only as well measured as its thinner side.
    w = np.array([min(len(by[c][prod]), len(by[c][canary])) for c in clusters], dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(iters, len(clusters)))
    boots = (diff[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return {
        "success": {
            "delta": float((diff * w).sum() / w.sum()),
            "ci95": (float(lo), float(hi)),
        },
        "n_clusters": len(clusters),
        "clusters": clusters,
        "n_prod": int(sum(len(by[c][prod]) for c in clusters)),
        "n_canary": int(sum(len(by[c][canary]) for c in clusters)),
        "per_cluster": {
            str(c): {
                "delta": float(np.mean(by[c][canary]) - np.mean(by[c][prod])),
                "n_prod": len(by[c][prod]),
                "n_canary": len(by[c][canary]),
            }
            for c in clusters
        },
    }


def _iso_since(spec: str) -> str:
    """Parse `7d`, `48h`, or an ISO timestamp into an ISO timestamp."""
    from datetime import UTC, datetime, timedelta

    spec = spec.strip()
    if spec.endswith(("d", "h")):
        try:
            n = int(spec[:-1])
        except ValueError:
            return spec
        delta = timedelta(days=n) if spec.endswith("d") else timedelta(hours=n)
        return (datetime.now(UTC) - delta).isoformat()
    return spec


def live_rows(registry: Any, since: str | None = None, limit: int = 100_000) -> list[dict]:
    """Gateway requests with an outcome, newest first."""
    from sqlalchemy import text

    sql = (
        "SELECT adapter_id, cluster_id, arm, outcome, fallback, received_at FROM requests "
        "WHERE outcome IS NOT NULL"
    )
    params: dict[str, Any] = {"lim": limit}
    if since:
        sql += " AND received_at >= :since"
        params["since"] = since
    sql += " ORDER BY received_at DESC LIMIT :lim"
    with registry.engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings()]
