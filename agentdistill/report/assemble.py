"""Assembling the report from the registry.

Everything the report says comes from a stored row, and every number carries the run id that produced it. A
number without a run id is a number nobody can check, which is the failure mode this whole project exists to
avoid.

A report with warnings is still a report. Missing a calibration, a teacher run, or a throughput measurement does
not suppress the rest -- it produces a yellow block saying which claim could not be made and why.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from agentdistill.report.cost import (
    breakeven_tasks_per_day,
    cascade_cost_per_task,
    saving_fraction,
    student_cost_per_mtok,
    teacher_cost_per_task,
)
from agentdistill.report.registry_views import (
    calibration_for,
    commands_for,
    latest_quantized,
    latest_round,
    lineage,
    per_cluster_table,
    pricing,
)

#: Subjects the report shows, in the order it shows them.
SUBJECT_ORDER = ("base", "student", "teacher", "student_unseen", "teacher_unseen")


#: Every warning code the report can carry. `scripts/clean_rehearsal.sh` must place each one as allowed or
#: forbidden, and a test holds it to that.
WARNING_CODES = (
    "tiny_mode", "no_eval_set", "no_run_found", "teacher_skipped", "no_student", "paired_failed",
    "no_calibration", "gate_not_usable", "gate_degenerate", "cascade_unverified", "quantized_unevaluated",
    "quantization_missing", "no_teacher_run", "no_teacher_config", "no_pricing", "no_prompt_tokens",
    "no_throughput", "cost_unbatched", "replay_teacher", "dirty_tree",
)


@dataclass
class ReportData:
    """Everything a renderer or `assert_report` needs, as plain data.

    Warnings are two parallel lists: `warnings` holds the prose the renderers print, and `warning_codes[i]` is
    the stable code for `warnings[i]`. Tooling asserts on codes because prose drifts; keeping the prose list as
    plain strings means no renderer had to change shape. Always add a warning through `warn` so the two stay
    aligned. The codes in use are `WARNING_CODES`; `warn` refuses any other.

    `onpolicy` is the latest on-policy round in the report's tag scope, whatever it decided: a discarded round is
    a finding (one round of RFT plus DPO did not beat SFT on this data), not a missing stage.
    """

    generated_at: str
    project: str
    eval_set: str
    unseen_set: str | None = None
    subjects: dict = field(default_factory=dict)
    paired: dict = field(default_factory=dict)
    cascade: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    per_cluster: list = field(default_factory=list)
    quantization: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    commands: list = field(default_factory=list)
    onpolicy: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    warning_codes: list = field(default_factory=list)
    tiny: bool = False

    def warn(self, code: str, message: str) -> None:
        if code not in WARNING_CODES:
            # A code outside the list is one the rehearsal gate has not placed as allowed or forbidden, so it
            # would pass that gate silently. Adding it here is the prompt to place it there too.
            raise ValueError(f"unknown warning code {code!r}; add it to WARNING_CODES and to clean_rehearsal.sh")
        self.warnings.append(message)
        self.warning_codes.append(code)

    def to_dict(self) -> dict:
        return asdict(self)


def assemble(registry: Any, cfg: Any, tag_glob: str | None = None, now_iso: str | None = None) -> ReportData:
    """Build the report's data from the registry. Never raises on missing pieces; warns instead."""
    from agentdistill.registry.select import NoMatch, best_adapter, latest_eval

    holdout = cfg.eval.eval_set
    report = ReportData(
        generated_at=now_iso or datetime.now(UTC).isoformat(timespec="seconds"),
        project=cfg.name,
        eval_set=holdout or "",
        unseen_set=getattr(cfg.eval, "unseen_set", None),
        tiny=str(getattr(cfg, "source_path", "") or "").endswith(".tiny.yaml"),
    )
    if report.tiny:
        report.warn(
            "tiny_mode",
            "TINY MODE: this run used the CPU rehearsal config. Every number here is a smoke test of the "
            "pipeline, not a measurement of anything."
        )
    if not holdout:
        report.warn("no_eval_set", "no eval set configured; there is nothing to report")
        return report

    def find(subject: str, eval_set: str) -> dict | None:
        try:
            return latest_eval(registry, subject=subject, eval_set=eval_set)
        except NoMatch:
            return None

    def record(name: str, run: dict | None) -> None:
        if run is None:
            if name.startswith("teacher") and getattr(cfg.eval, "skip_teacher", False):
                # Declared, not missing. The two must read differently: one is a decision, the other a hole.
                report.warn("teacher_skipped", f"`{name}` was skipped by configuration (eval.skip_teacher set)")
                return
            report.warn("no_run_found", f"no eval run for `{name}` on `{holdout}`")
            return
        m = run.get("metrics") or {}
        report.subjects[name] = {
            "run_id": run["id"], "subject": run["subject"],
            "success": m.get("success"), "schema_valid": m.get("schema_valid"),
            "divergence_rate": m.get("divergence_rate"), "tokens_median": m.get("tokens_est_median"),
            "turns_median": m.get("turns_median"),
            "n_tasks": m.get("n_tasks"), "n_per_task": m.get("n_per_task"),
            "command": run.get("command"),
        }

    base = find("base", holdout)
    teacher = find("teacher", holdout)
    try:
        best = best_adapter(registry, tag=tag_glob, eval_set=holdout)
    except NoMatch:
        best = None
        report.warn(
            "no_student",
            "no adapter has an eval run on the frozen set, so there is no student to report on"
        )
    student = find(best["id"], holdout) or find(best["name"], holdout) if best else None

    record("base", base)
    record("student", student)
    record("teacher", teacher)

    if best and report.unseen_set:
        record("student_unseen", find(best["id"], report.unseen_set) or find(best["name"], report.unseen_set))
        record("teacher_unseen", find("teacher", report.unseen_set))

    report.paired = _paired(registry, student, teacher, base, report)
    report.calibration, report.cascade = _calibration(registry, best, report)

    if best:
        report.per_cluster = per_cluster_table(
            registry,
            base_run=base["id"] if base else None,
            student_run=student["id"] if student else None,
            teacher_run=teacher["id"] if teacher else None,
            floor=cfg.router.floor,
        )
        report.quantization = _quantization(
            registry, best, student, report,
            quantization_configured=bool(getattr(getattr(cfg, "serve", None), "quantization", None)),
        )
        report.lineage = lineage(registry, best["id"])

    if teacher and (teacher.get("metrics") or {}).get("teacher_backend") == "replay":
        # Raised here, not in cost_block, so the disclosure survives any path that skips the cost section: a
        # replay teacher's success rate is as structural as its cost.
        report.warn(
            "replay_teacher",
            "teacher metrics come from a replay stub (tiny mode); cost figures are structural, not measured",
        )

    report.cost = cost_block(cfg, registry, teacher, student, report.cascade, report)
    report.onpolicy = onpolicy_block(latest_round(registry, tag_glob))
    report.commands = commands_for(registry, tag_glob)
    dirty = sum(1 for c in report.commands if (c.get("provenance") or {}).get("dirty") is True)
    if dirty:
        report.warn(
            "dirty_tree",
            f"{dirty} run(s) were recorded from a dirty working tree; their numbers cannot be reproduced from a "
            "commit",
        )
    return report


def _paired(registry: Any, student: dict | None, teacher: dict | None, base: dict | None,
            report: ReportData) -> dict:
    from agentdistill.eval.runner import compare

    out: dict[str, Any] = {}
    for name, other in (("student_vs_teacher", teacher), ("student_vs_base", base)):
        if not (student and other):
            continue
        try:
            # Student first, so a positive delta favours the student throughout the report.
            out[name] = compare(registry, student["id"], other["id"])
        except ValueError as e:
            report.warn("paired_failed", f"{name} could not be computed: {e}")
    return out


def _calibration(registry: Any, best: dict | None, report: ReportData) -> tuple[dict, dict]:
    if not best:
        return {}, {}
    row = calibration_for(registry, best["id"])
    if not row:
        report.warn(
            "no_calibration",
            "no calibration for the best adapter, so the gateway would escalate every turn and no cost saving "
            "can be claimed"
        )
        return {}, {}
    holdout_metrics = row.get("holdout_metrics") or {}
    verdict = row.get("verdict")
    calibration = {
        "id": row["id"], "holdout": holdout_metrics, "bins": row.get("reliability_bins") or [],
        "threshold": row.get("threshold"), "features": row.get("feature_order") or row.get("features"),
        "ece": holdout_metrics.get("ece"), "auroc": holdout_metrics.get("auroc"), "verdict": verdict,
        "note": (row.get("report") or {}).get("threshold_note") or "",
        "verdict_reason": (row.get("report") or {}).get("verdict_reason") or "",
    }
    if verdict == "degenerate_labels":
        # A data problem, not a feature problem: the tasks did not separate, so there is nothing to rank. Its own
        # code, so tiny mode (where a random model gets every turn wrong) can allow it and the GPU day forbid it.
        reason = calibration["verdict_reason"] or "every labelled turn has the same label"
        report.warn(
            "gate_degenerate",
            f"the confidence gate was fitted on degenerate labels ({reason}), so AUROC is undefined; the gateway "
            "refuses it and escalates every turn"
        )
    elif verdict is not None and verdict != "usable":
        report.warn(
            "gate_not_usable",
            f"the confidence gate's verdict is `{verdict}` (holdout AUROC "
            f"{_fmt(holdout_metrics.get('auroc'))}), so the gateway refuses it and escalates every turn"
        )
    cascade = {"threshold": row.get("threshold"), "verified": row.get("verified") or [],
               "escalate_everything": verdict is not None and verdict != "usable"}
    if not cascade["verified"]:
        report.warn(
            "cascade_unverified",
            "the cascade threshold was never verified by the harness, so the cascade numbers below are an "
            "analytic estimate that assumes an escalated turn is as good as the teacher's"
        )
    return calibration, cascade


def _fmt(x: float | None) -> str:
    return "undefined" if x is None or x != x else f"{x:.3f}"


def _quantization(registry: Any, best: dict, student: dict | None, report: ReportData,
                  quantization_configured: bool = False) -> dict:
    from agentdistill.registry.select import NoMatch, latest_eval

    row = latest_quantized(registry, best["id"])
    if not row:
        if quantization_configured:
            # Serving is configured quantized, so a missing artifact means the quantize stage did not run or
            # wrote nothing; an empty section would read as "not applicable".
            report.warn(
                "quantization_missing",
                "no quantized artifact for the best adapter, so the quantization delta is unknown",
            )
        return {}
    try:
        run = latest_eval(registry, subject=row["id"])
    except NoMatch:
        report.warn(
            "quantized_unevaluated",
            f"the quantized artifact {row['id']} was never evaluated, so its quality delta is unknown"
        )
        return {"method": row.get("quantization"), "adapter_id": row["id"]}
    bf16 = (student or {}).get("metrics", {}).get("success") if student else None
    quantized = (run.get("metrics") or {}).get("success")
    return {
        "method": row.get("quantization"),
        "adapter_id": row["id"],
        "run_id": run["id"],
        "bf16_success": bf16,
        "quantized_success": quantized,
        "delta_pp": (quantized - bf16) * 100 if (bf16 is not None and quantized is not None) else None,
    }


def cost_block(cfg: Any, registry: Any, teacher: dict | None, student: dict | None,
               cascade: dict, report: ReportData) -> dict:
    """Cost per task before and after, or an explicit reason it could not be computed."""
    if not teacher:
        report.warn("no_teacher_run", "no teacher eval run, so there is no baseline cost to compare against")
        return {}
    if not cfg.teacher:
        report.warn("no_teacher_config", "no teacher configured, so the teacher's price is unknown")
        return {}

    metrics = teacher.get("metrics") or {}
    price = pricing(registry, cfg.teacher.provider, cfg.teacher.model) or _configured_price(cfg)
    if not price:
        report.warn(
            "no_pricing",
            f"no pricing on file for {cfg.teacher.provider}/{cfg.teacher.model}; set teacher.input_per_mtok and "
            f"teacher.output_per_mtok, or load model_pricing"
        )
        return {}

    prompt_median = metrics.get("prompt_tokens_median")
    completion_median = metrics.get("completion_tokens_median") or metrics.get("tokens_est_median")
    if prompt_median is None:
        report.warn(
            "no_prompt_tokens",
            "the teacher eval did not record prompt-token usage, so its per-task cost is estimated from "
            "completion tokens only and understates the real figure"
        )
        prompt_median = 0.0

    teacher_task = teacher_cost_per_task(
        prompt_median, completion_median or 0.0,
        float(price["input_per_mtok"]), float(price["output_per_mtok"]),
        cache_hit_frac=metrics.get("cache_hit_frac", 0.0),
        cache_read_per_mtok=(float(price["cache_read_per_mtok"]) if price.get("cache_read_per_mtok") else None),
    )
    out: dict[str, Any] = {"teacher_cost_per_task": teacher_task, "pricing": dict(price)}

    if not student:
        return out

    student_metrics = student.get("metrics") or {}
    throughput = student_metrics.get("throughput_tok_per_s")
    if not throughput:
        report.warn(
            "no_throughput",
            "student throughput was not measured, so cost per task cannot be computed. Throughput comes from a "
            "batched eval run; an unbatched one would overstate the cost several times over."
        )
        return out

    per_mtok = student_cost_per_mtok(cfg.serve.gpu_usd_per_hour, throughput, getattr(cfg.serve, "utilization", 0.6))
    out["student_cost_per_mtok"] = per_mtok
    out["student_only_cost_per_task"] = (student_metrics.get("tokens_est_median") or 0.0) * per_mtok / 1e6
    out["throughput_tok_per_s"] = throughput
    out["throughput_conditions"] = student_metrics.get("throughput_conditions", "unstated")
    # A run that predates the field was measured by the sequential harness, so missing reads as unbatched.
    mode = student_metrics.get("throughput_mode") or "unbatched"
    out["throughput_mode"] = mode
    unbatched = mode != "batched"
    if unbatched:
        report.warn(
            "cost_unbatched",
            "student throughput was measured unbatched (one request at a time), so its cost per token is an "
            "upper bound from sequential measurement and the cascade is not priced against the teacher. Rerun the "
            "student eval with a batch size to measure a serving figure.",
        )

    verified = cascade.get("verified") or []
    chosen = _chosen_point(verified, cascade.get("threshold"))
    if chosen:
        cost = cascade_cost_per_task(
            student_metrics.get("tokens_est_median") or 0.0, per_mtok,
            chosen.get("escalation_rate", 0.0), teacher_task,
            chosen.get("wasted_student_tokens", 0.0),
        )
        out["cascade"] = {
            "threshold": chosen.get("threshold"),
            "cost_per_task": cost,
            "success": chosen.get("success"),
            "escalation_rate": chosen.get("escalation_rate"),
            # Not computed from an upper-bound cost: a saving derived from it would understate the real one by an
            # unknown factor, and a number that is wrong in a known direction still reads as a finding.
            "saving_frac": None if unbatched else saving_fraction(teacher_task, cost),
            "breakeven_tasks_per_day": (None if unbatched
                                        else breakeven_tasks_per_day(cfg.serve.gpu_usd_per_hour, teacher_task, cost)),
            "cost_is_upper_bound": unbatched,
            "measured": True,
        }
    return out


def onpolicy_block(row: dict | None) -> dict:
    """The report's view of one on-policy round row, or {} when no round was ever run."""
    if not row:
        return {}
    stats = row.get("pair_stats") or {}
    compare = row.get("compare") or {}
    weak = compare.get("insufficient_power") if isinstance(compare, dict) else None
    out = {
        "round_id": row.get("id"),
        "tag": row.get("tag"),
        "round_idx": row.get("round_idx"),
        "start_adapter": row.get("start_adapter_id"),
        "candidate_adapter": row.get("candidate_adapter"),
        "eval_run_id": row.get("eval_run_id"),
        "decision": row.get("decision"),
        "reason": row.get("reason") or "",
        "n_rollouts": row.get("n_rollouts"),
        "fuzzy_share": row.get("fuzzy_share"),
        "pair_stats": {key: stats.get(key) for key in (
            "n_pairs", "n_rollout", "n_teacher", "max_per_task", "cap_per_task", "per_task_histogram",
            "diff_kind", "warnings")} if stats else {},
        "insufficient_power": (weak or {}).get("reason") if weak else None,
    }
    return out


def _chosen_point(verified: list[dict], threshold: float | None) -> dict | None:
    """The verified point at the chosen threshold, or the middle one if the exact threshold is not among them."""
    if not verified:
        return None
    if threshold is not None:
        exact = next((p for p in verified if abs((p.get("threshold") or -1) - threshold) < 1e-6), None)
        if exact:
            return exact
    return verified[len(verified) // 2]


def _configured_price(cfg: Any) -> dict | None:
    """Fall back to prices written into project.yaml when the registry has none."""
    teacher = cfg.teacher
    if teacher and teacher.input_per_mtok is not None and teacher.output_per_mtok is not None:
        return {
            "provider": teacher.provider, "model": teacher.model,
            "input_per_mtok": teacher.input_per_mtok, "output_per_mtok": teacher.output_per_mtok,
            "cache_read_per_mtok": teacher.cache_read_per_mtok, "source": "project.yaml",
        }
    return None
