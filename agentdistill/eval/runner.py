"""Eval runs and paired comparisons.

`run_eval` executes an eval set under one subject and stores every repeat. `compare` produces the paired report
that every claim in this project has to come from.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agentdistill.eval.harness import TaskOutcome, run_task
from agentdistill.eval.replay import ReplayStats, ReplayToolProvider
from agentdistill.eval.stats import (
    cluster_bootstrap_diff,
    holm,
    mcnemar_paired,
    metric_by_task,
    minimum_n_guard,
    success_by_task,
    wilcoxon_metric,
)

#: (trace, outcome) -> (success, detail dict)
Grader = Callable[[dict, TaskOutcome], tuple[bool, dict]]


@dataclass
class RunSpec:
    subject: str
    eval_set: str
    n_per_task: int = 5
    policy: str = "strict"
    max_turns: int = 12
    fuzzy_threshold: float = 0.92
    #: Groups the runs of one session so `eval latest --tag` can find them.
    tag: str | None = None


def label_grader(trace: dict, outcome: TaskOutcome) -> tuple[bool, dict]:
    """Fallback grader: compare the final text to the recorded one.

    Weak on purpose, and it says so in the report. A project without state predicates should use a judge or write
    predicates rather than trust this.
    """
    recorded = next((m.get("content") or "" for m in reversed(trace["messages"]) if m["role"] == "assistant"), "")
    return outcome.final_text.strip() == recorded.strip(), {"detail": "exact final-text match"}


def run_eval(
    registry: Any,
    eval_set: dict,
    traces_by_task: dict[str, dict],
    client: Any,
    grader: Grader,
    spec: RunSpec,
    store_messages: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Run every task `n_per_task` times, grade each, and store the rows. Returns the run id."""
    run_id = f"ev_{uuid.uuid4().hex[:16]}"
    registry.start_eval_run(run_id, eval_set["id"], spec.subject, spec.n_per_task, tag=spec.tag)

    task_ids = [t for t in eval_set["trace_ids"] if t in traces_by_task]
    total = len(task_ids) * spec.n_per_task
    done = 0
    for trace_id in task_ids:
        trace = traces_by_task[trace_id]
        for k in range(spec.n_per_task):
            # A fresh provider per repeat: replay stats are per-run, and a shared provider would accumulate
            # counts across repeats and report a divergence rate several times too high.
            provider = ReplayToolProvider(trace, policy=spec.policy, fuzzy_threshold=spec.fuzzy_threshold)
            if hasattr(client, "reset"):
                client.reset()
            outcome = run_task(trace, client, provider, repeat_idx=k, max_turns=spec.max_turns)
            success, detail = grader(trace, outcome)
            outcome.success = success
            outcome.grader_detail = str(detail.get("detail", ""))[:500]
            outcome.grader_out = detail
            registry.write_eval_result(run_id, outcome, cluster=trace.get("cluster"),
                                       store_messages=store_messages)
            done += 1
            if progress:
                progress(done, total)

    metrics, per_cluster = aggregate(registry.eval_results(run_id), traces_by_task)
    registry.finish_eval_run(run_id, metrics, per_cluster)
    return run_id


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 else float((s[mid - 1] + s[mid]) / 2)


def aggregate(rows: list[dict], traces_by_task: dict[str, dict]) -> tuple[dict, dict]:
    """Run-level metrics and a per-cluster breakdown."""
    if not rows:
        return {"n_rows": 0}, {}

    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)

    replay = ReplayStats()
    for r in rows:
        if r.get("replay_stats"):
            replay.add(r["replay_stats"])

    per_cluster: dict[str, dict] = {}
    for task, task_rows in by_task.items():
        cluster = str(traces_by_task.get(task, {}).get("cluster", "none"))
        entry = per_cluster.setdefault(cluster, {"n_tasks": 0, "success": 0.0, "divergence": 0.0})
        entry["n_tasks"] += 1
        entry["success"] += sum(bool(r["success"]) for r in task_rows) / len(task_rows)
        entry["divergence"] += sum(bool(r["diverged"]) for r in task_rows) / len(task_rows)
    for entry in per_cluster.values():
        entry["success"] /= entry["n_tasks"]
        entry["divergence"] /= entry["n_tasks"]

    n = len(rows)
    metrics = {
        "n_rows": n,
        "n_tasks": len(by_task),
        "n_per_task": max(len(v) for v in by_task.values()),
        # Task-level mean, not row-level: every task weighs the same regardless of how many repeats it got.
        "success": sum(
            sum(bool(r["success"]) for r in rs) / len(rs) for rs in by_task.values()
        ) / len(by_task),
        "schema_valid": sum(bool(r["schema_valid"]) for r in rows) / n,
        "divergence_rate": sum(bool(r["diverged"]) for r in rows) / n,
        "turns_median": _median([r["n_turns"] for r in rows]),
        "tool_calls_median": _median([r["n_tool_calls"] for r in rows]),
        "tokens_est_median": _median([r["completion_tokens_est"] for r in rows]),
        "latency_ms_median": _median([r["latency_ms"] for r in rows]),
        "stop_reasons": _counts(r["stop_reason"] for r in rows),
        "replay": replay.to_dict(),
    }
    return metrics, per_cluster


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def compare(registry: Any, run_a: str, run_b: str, alpha: float = 0.05) -> dict:
    """Paired comparison of two runs over the same eval set. `a - b`, so positive favours `a`."""
    ra, rb = registry.get_eval_run(run_a), registry.get_eval_run(run_b)
    if ra is None or rb is None:
        raise ValueError(f"unknown eval run: {run_a if ra is None else run_b}")
    if ra["eval_set_id"] != rb["eval_set_id"]:
        raise ValueError(
            f"the two runs used different eval sets ({ra['eval_set_id']} and {rb['eval_set_id']}); "
            f"they cannot be paired"
        )

    rows_a, rows_b = registry.eval_results(run_a), registry.eval_results(run_b)
    oa, ob = success_by_task(rows_a, "success"), success_by_task(rows_b, "success")
    shared = set(oa) & set(ob)
    minimum_n_guard(len(shared), min(min(len(v) for v in oa.values()), min(len(v) for v in ob.values())))

    success = cluster_bootstrap_diff(oa, ob)
    mcnemar = mcnemar_paired(oa, ob)
    tokens = wilcoxon_metric(
        metric_by_task(rows_a, "completion_tokens_est"), metric_by_task(rows_b, "completion_tokens_est")
    )
    turns = wilcoxon_metric(metric_by_task(rows_a, "n_turns"), metric_by_task(rows_b, "n_turns"))
    significance = holm({"success": mcnemar["p"], "tokens": tokens["p"], "turns": turns["p"]}, alpha)

    return {
        "subject_a": ra["subject"],
        "subject_b": rb["subject"],
        "run_a": run_a,
        "run_b": run_b,
        "eval_set_id": ra["eval_set_id"],
        "n_shared_tasks": len(shared),
        "success": success.to_dict(),
        "mcnemar": mcnemar,
        "tokens": tokens,
        "turns": turns,
        "holm": significance,
        "metrics_a": ra["metrics"],
        "metrics_b": rb["metrics"],
        "weakest_clusters": weakest_clusters(ra.get("per_cluster") or {}, rb.get("per_cluster") or {}),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "alpha": alpha,
    }


def weakest_clusters(a: dict, b: dict, limit: int = 5) -> list[dict]:
    """Clusters where `a` trails `b` most. The input to the next curation round and to the router floor."""
    out = []
    for cluster, entry in a.items():
        other = b.get(cluster)
        if not other:
            continue
        out.append({
            "cluster": cluster,
            "n_tasks": entry["n_tasks"],
            "success_a": entry["success"],
            "success_b": other["success"],
            "delta": entry["success"] - other["success"],
        })
    out.sort(key=lambda c: c["delta"])
    return out[:limit]
