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
from agentdistill.eval.lockstep import LockstepItem, LockstepStats, run_lockstep, supports_batching
from agentdistill.eval.replay import ReplayStats, ReplayToolProvider
from agentdistill.eval.stats import (
    InsufficientPower,
    cluster_bootstrap_diff,
    holm,
    mcnemar_paired,
    metric_by_task,
    power_check,
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
    batch_size: int | None = None,
) -> str:
    """Run every task `n_per_task` times, grade each, and store the rows. Returns the run id.

    `batch_size=None` runs one task at a time. With a batch size, the (task, repeat) items run through the
    lockstep runner with at most that many in flight -- but only when the client can batch and carries no
    per-task state; otherwise the run falls back to the sequential path, and its throughput is labelled
    unbatched either way.
    """
    run_id = f"ev_{uuid.uuid4().hex[:16]}"
    registry.start_eval_run(run_id, eval_set["id"], spec.subject, spec.n_per_task, tag=spec.tag)

    task_ids = [t for t in eval_set["trace_ids"] if t in traces_by_task]
    total = len(task_ids) * spec.n_per_task
    done = 0
    usage_rows: list[dict[str, int]] = []

    def record(trace: dict, outcome: TaskOutcome) -> None:
        nonlocal done
        success, detail = grader(trace, outcome)
        outcome.success = success
        outcome.grader_detail = str(detail.get("detail", ""))[:500]
        outcome.grader_out = detail
        registry.write_eval_result(run_id, outcome, cluster=trace.get("cluster"), store_messages=store_messages)
        done += 1
        if progress:
            progress(done, total)

    lockstep = None
    if batch_size is not None and lockstep_eligible(client):
        items = [
            # A fresh provider per repeat, exactly as on the sequential path.
            LockstepItem(traces_by_task[trace_id],
                         ReplayToolProvider(traces_by_task[trace_id], policy=spec.policy,
                                            fuzzy_threshold=spec.fuzzy_threshold),
                         repeat_idx=k)
            for trace_id in task_ids for k in range(spec.n_per_task)
        ]
        _, lockstep = run_lockstep(items, client, batch_size, max_turns=spec.max_turns,
                                   on_done=lambda item, outcome: record(item.trace, outcome))
    else:
        for trace_id in task_ids:
            trace = traces_by_task[trace_id]
            for k in range(spec.n_per_task):
                # A fresh provider per repeat: replay stats are per-run, and a shared provider would accumulate
                # counts across repeats and report a divergence rate several times too high.
                provider = ReplayToolProvider(trace, policy=spec.policy, fuzzy_threshold=spec.fuzzy_threshold)
                if hasattr(client, "reset"):
                    client.reset()
                before = dict(getattr(client, "usage", None) or {})
                outcome = run_task(trace, client, provider, repeat_idx=k, max_turns=spec.max_turns)
                after = dict(getattr(client, "usage", None) or {})
                if after:
                    usage_rows.append({k2: after.get(k2, 0) - before.get(k2, 0) for k2 in after})
                record(trace, outcome)

    rows = registry.eval_results(run_id)
    metrics, per_cluster = aggregate(rows, traces_by_task)
    metrics.update(usage_metrics(usage_rows))
    batched = lockstep is not None and lockstep.batched
    metrics.update(batched_throughput_metrics(rows, lockstep) if batched else throughput_metrics(rows))
    # Recorded even when throughput itself could not be measured, so the report never has to guess the mode.
    metrics["throughput_mode"] = "batched" if batched else "unbatched"
    if getattr(client, "backend_name", None):
        # A replay stub's numbers are structural. The report keys its disclosure on this field.
        metrics["teacher_backend"] = client.backend_name
    if hasattr(client, "summary") and rows:
        # Only a cascade has a gate summary. Without these the run records per-row escalations and nothing
        # run-level, and `--verify-threshold` has nothing to verify.
        metrics.update(cascade_metrics(rows, getattr(client, "threshold", None)))
    registry.finish_eval_run(run_id, metrics, per_cluster)
    return run_id


def lockstep_eligible(client: Any) -> bool:
    """Whether a client can run in lockstep without changing what the run records.

    It has to batch, and it must not keep per-task state: `reset`, a gate `summary`, or running `usage` totals are
    all read around one task at a time, and interleaving tasks would mix them up. Such clients run sequentially.
    """
    if not supports_batching(client):
        return False
    return not any(hasattr(client, attr) for attr in ("reset", "summary", "usage"))


def usage_metrics(usage_rows: list[dict[str, int]]) -> dict:
    """Per-task token usage as the provider reported it, for the teacher's cost. Empty when the client does not
    report usage: an estimate from completion text would understate the prompt, which is the larger half."""
    if not usage_rows:
        return {}
    out = {
        "prompt_tokens_median": _median([float(u.get("prompt_tokens", 0)) for u in usage_rows]),
        "completion_tokens_median": _median([float(u.get("completion_tokens", 0)) for u in usage_rows]),
    }
    cached = [float(u.get("cached_prompt_tokens", 0)) for u in usage_rows]
    prompts = sum(float(u.get("prompt_tokens", 0)) for u in usage_rows)
    if any(cached) and prompts:
        out["cache_hit_frac"] = sum(cached) / prompts
    return out


def throughput_metrics(rows: list[dict]) -> dict:
    """Completion tokens per second of wall clock, and the conditions it was measured under.

    The harness sends one request at a time, so this is a floor on what batched serving achieves and a cost
    computed from it is an upper bound. The conditions string says so, and the report prints it beside the cost.
    """
    tokens = sum(float(r.get("completion_tokens_est") or 0) for r in rows)
    seconds = sum(float(r.get("latency_ms") or 0) for r in rows) / 1000
    if not tokens or seconds <= 0:
        return {}
    return {"throughput_tok_per_s": tokens / seconds,
            "throughput_mode": "unbatched",
            "throughput_conditions": "sequential eval harness, one request at a time (unbatched; overstates cost)"}


def batched_throughput_metrics(rows: list[dict], stats: LockstepStats) -> dict:
    """Completion tokens per second of generation time, measured with real batched calls.

    The denominator is the time spent inside `next_turns_batch`, not the run's wall clock, so replay and grading
    overhead does not deflate a serving figure. The conditions string carries the batch size, because a
    throughput without its concurrency is not comparable to anything.
    """
    tokens = sum(float(r.get("completion_tokens_est") or 0) for r in rows)
    if not tokens or stats.generate_seconds <= 0:
        return {}
    return {"throughput_tok_per_s": tokens / stats.generate_seconds,
            "throughput_mode": "batched",
            "throughput_conditions": f"batched lockstep eval, batch={stats.batch_size}",
            "lockstep": {"calls": stats.calls, "item_turns": stats.item_turns,
                         "max_inflight": stats.max_inflight}}


def cascade_metrics(rows: list[dict], threshold: float | None) -> dict:
    """Run-level gate figures for a cascade subject: escalated turns over all turns, and the student tokens the
    escalations threw away (generated, paid for, discarded)."""
    turns = sum(int(r.get("n_turns") or 0) for r in rows)
    escalations = sum(int(r.get("escalations") or 0) for r in rows)
    return {
        "escalation_rate": escalations / turns if turns else 0.0,
        "escalations": escalations,
        "wasted_student_tokens_median": _median([float(r.get("wasted_student_tokens") or 0) for r in rows]),
        "cascade_threshold": threshold,
    }


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


def _observed_rate(outcomes: dict[str, list[float]], tasks: list[str]) -> float | None:
    """Task-level mean success over `tasks`: what happened, with no claim about what it means."""
    if not tasks:
        return None
    return sum(sum(outcomes[t]) / len(outcomes[t]) for t in tasks) / len(tasks)


def compare(registry: Any, run_a: str, run_b: str, alpha: float = 0.05) -> dict:
    """Paired comparison of two runs over the same eval set. `a - b`, so positive favours `a`.

    Below the power floor, or on degenerate data, the result carries `insufficient_power` and the raw observed
    rates instead of statistics. It never raises for lack of data; it does raise for runs that cannot be paired at
    all (unknown ids, different eval sets), because those are mistakes, not small samples.
    """
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
    common = sorted(set(oa) & set(ob))
    n_repeats = min([len(oa[t]) for t in common] + [len(ob[t]) for t in common], default=0)
    head = {
        "subject_a": ra["subject"],
        "subject_b": rb["subject"],
        "run_a": run_a,
        "run_b": run_b,
        "eval_set_id": ra["eval_set_id"],
        "n_shared_tasks": len(common),
        "n_repeats": n_repeats,
        "metrics_a": ra["metrics"],
        "metrics_b": rb["metrics"],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "alpha": alpha,
    }

    def refused(weak: InsufficientPower) -> dict:
        # `observed` is labelled as such and deliberately carries no delta: a difference printed without an
        # interval reads as a finding.
        return {**head, "insufficient_power": weak.as_dict(),
                "observed": {"rate_a": _observed_rate(oa, common), "rate_b": _observed_rate(ob, common)}}

    weak = power_check(len(common), n_repeats)
    if weak:
        return refused(weak)

    success = cluster_bootstrap_diff(oa, ob)
    lo, hi = success.ci95
    if lo == hi and success.delta == 0.0:
        # Every task came out the same under both subjects, so the interval has zero width. That is not a
        # precise measurement of no difference; it is data with no variance to measure anything from.
        return refused(InsufficientPower(
            len(common), n_repeats,
            degenerate="every task had the same outcome under both subjects, so the interval has zero width",
        ))

    mcnemar = mcnemar_paired(oa, ob)
    tokens = wilcoxon_metric(
        metric_by_task(rows_a, "completion_tokens_est"), metric_by_task(rows_b, "completion_tokens_est")
    )
    turns = wilcoxon_metric(metric_by_task(rows_a, "n_turns"), metric_by_task(rows_b, "n_turns"))
    significance = holm({"success": mcnemar["p"], "tokens": tokens["p"], "turns": turns["p"]}, alpha)

    return {
        **head,
        "success": success.to_dict(),
        "mcnemar": mcnemar,
        "tokens": tokens,
        "turns": turns,
        "holm": significance,
        "weakest_clusters": weakest_clusters(ra.get("per_cluster") or {}, rb.get("per_cluster") or {}),
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
