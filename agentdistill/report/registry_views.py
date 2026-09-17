"""Registry queries the report needs.

Kept out of `Registry` because they are report-shaped rather than storage-shaped: a per-cluster table joining
three eval runs, an adapter's provenance chain, and the exact commands that produced a tag's rows.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from agentdistill.registry.base import loads


def per_cluster_table(
    registry: Any,
    base_run: str | None,
    student_run: str | None,
    teacher_run: str | None,
    floor: float = 0.55,
) -> list[dict]:
    """Success per cluster for each subject, with the routing decision the floor implies.

    This is the row that tells you where the student cannot be trusted, and it is the input to both the next
    curation round and the router's floor.
    """
    runs = {"base": base_run, "student": student_run, "teacher": teacher_run}
    per_subject: dict[str, dict[str, dict]] = {}
    for name, run_id in runs.items():
        if not run_id:
            continue
        run = registry.get_eval_run(run_id)
        per_subject[name] = (run or {}).get("per_cluster") or {}

    clusters = sorted({c for table in per_subject.values() for c in table},
                      key=lambda c: (c == "none", c))
    rows = []
    for cluster in clusters:
        row: dict[str, Any] = {"cluster": cluster}
        for name in ("base", "student", "teacher"):
            entry = per_subject.get(name, {}).get(cluster) or {}
            row[name] = entry.get("success")
            row.setdefault("n_tasks", entry.get("n_tasks"))
        student = row.get("student")
        row["routing"] = (
            "teacher (below floor)" if student is not None and student < floor
            else ("student" if student is not None else "unmeasured")
        )
        rows.append(row)
    return rows


def lineage(registry: Any, adapter_id: str) -> dict:
    """Everything that produced an adapter: dataset and its hash, training run, parents, quantization."""
    from agentdistill.registry.lifecycle import adapter as get_adapter
    from agentdistill.registry.lifecycle import events

    try:
        row = get_adapter(registry, adapter_id)
    except LookupError:
        return {}

    out: dict[str, Any] = {
        "adapter": {"id": row["id"], "name": row["name"], "version": row["version"], "status": row["status"],
                    "base_model": row["base_model"], "tag": row.get("tag"),
                    "quantization": row.get("quantization"), "merged": bool(row.get("merged"))},
        "parents": [],
        "events": [
            {"at": e["created_at"], "from": e["from_status"], "to": e["to_status"], "actor": e["actor"]}
            for e in events(registry, row["id"])
        ],
    }

    run = registry.get_training_run(row["training_run_id"])
    if run:
        out["training_run"] = {"id": run["id"], "method": run["method"], "status": run["status"],
                               "command": run.get("command")}
        dataset = next((d for d in registry.list_datasets() if d["id"] == run["dataset_id"]), None)
        if dataset:
            out["dataset"] = {
                "id": dataset["id"], "name": dataset["name"], "version": dataset["version"],
                "n_samples": dataset["n_samples"], "content_hash": dataset["content_hash"],
                "filter_config": dataset["filter_config"], "report_path": dataset.get("report_path"),
            }

    parent = row.get("parent_adapter_id")
    seen = {row["id"]}
    while parent and parent not in seen:
        seen.add(parent)
        try:
            prow = get_adapter(registry, parent)
        except LookupError:
            break
        out["parents"].append({"id": prow["id"], "name": prow["name"], "version": prow["version"],
                               "merged": bool(prow.get("merged"))})
        parent = prow.get("parent_adapter_id")
    return out


def commands_for(registry: Any, tag_glob: str | None = None) -> list[dict]:
    """The exact invocations behind a tag's rows, so a report can say how to reproduce itself."""
    import fnmatch

    out: list[dict] = []
    with registry.engine.connect() as conn:
        for kind, table, tag_column in (
            ("training_run", "training_runs", None),
            ("eval_run", "eval_runs", "tag"),
            ("calibration", "calibrations", None),
        ):
            try:
                rows = conn.execute(text(f"SELECT * FROM {table}")).mappings().fetchall()
            except Exception:
                continue
            for r in rows:
                row = dict(r)
                if not row.get("command"):
                    continue
                if tag_glob and tag_column:
                    tag = row.get(tag_column)
                    if not tag or not fnmatch.fnmatch(tag, tag_glob):
                        continue
                out.append({"kind": kind, "id": row["id"], "command": row["command"],
                            "at": row.get("started_at") or row.get("created_at")})
    out.sort(key=lambda r: r.get("at") or "")
    return out


def latest_quantized(registry: Any, parent_adapter_id: str) -> dict | None:
    """The quantized artifact derived from an adapter, if one was registered."""
    with registry.engine.connect() as conn:
        row = conn.execute(
            text(
                """SELECT * FROM adapters
                   WHERE parent_adapter_id = :p AND quantization IS NOT NULL
                   ORDER BY created_at DESC LIMIT 1"""
            ),
            {"p": parent_adapter_id},
        ).mappings().first()
    return dict(row) if row else None


def pricing(registry: Any, provider: str, model: str) -> dict | None:
    """The most recent price on file for a model."""
    with registry.engine.connect() as conn:
        row = conn.execute(
            text(
                """SELECT * FROM model_pricing WHERE provider = :p AND model = :m
                   ORDER BY effective_from DESC LIMIT 1"""
            ),
            {"p": provider, "m": model},
        ).mappings().first()
    return dict(row) if row else None


def per_cluster_counts(registry: Any, run_id: str | None) -> dict[tuple[int, str], tuple[int, int]]:
    """Per-cluster (successes, failures) from an eval run, for the router's warm start."""
    if not run_id:
        return {}
    out: dict[tuple[int, str], tuple[int, int]] = {}
    run = registry.get_eval_run(run_id)
    for cluster, entry in ((run or {}).get("per_cluster") or {}).items():
        if cluster == "none":
            continue
        n = int(entry.get("n_tasks") or 0)
        successes = round(float(entry.get("success") or 0.0) * n)
        out[(int(cluster), "student")] = (successes, n - successes)
    return out


def router_state(registry: Any) -> list[dict]:
    with registry.engine.connect() as conn:
        return [dict(r) for r in conn.execute(text("SELECT * FROM router_state")).mappings()]


def upsert_router_state(registry: Any, rows: list[tuple[int, str, float, float]]) -> None:
    from agentdistill.registry.base import utcnow

    with registry.engine.begin() as conn:
        for cluster_id, arm, alpha, beta in rows:
            conn.execute(
                text("DELETE FROM router_state WHERE cluster_id = :c AND arm = :a"),
                {"c": cluster_id, "a": arm},
            )
            conn.execute(
                text(
                    """INSERT INTO router_state (cluster_id, arm, alpha, beta, updated_at)
                       VALUES (:c, :a, :alpha, :beta, :t)"""
                ),
                {"c": cluster_id, "a": arm, "alpha": alpha, "beta": beta, "t": utcnow()},
            )


def reset_router_state(registry: Any) -> None:
    """Wipe the posteriors.

    Called when a new adapter reaches prod: the old adapter's history is not evidence about the new one, and
    inheriting it would take many requests to unlearn.
    """
    with registry.engine.begin() as conn:
        conn.execute(text("DELETE FROM router_state"))


def eval_run_command(registry: Any, run_id: str) -> str | None:
    with registry.engine.connect() as conn:
        row = conn.execute(text("SELECT command FROM eval_runs WHERE id = :i"), {"i": run_id}).first()
    return row[0] if row else None


def calibration_for(registry: Any, adapter_id: str) -> dict | None:
    with registry.engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM calibrations WHERE adapter_id = :a ORDER BY created_at DESC LIMIT 1"),
            {"a": adapter_id},
        ).mappings().first()
    if not row:
        return None
    out = dict(row)
    for key in ("target", "holdout_metrics", "reliability_bins", "verified", "features", "feature_order"):
        if key in out:
            out[key] = loads(out[key])
    return out
