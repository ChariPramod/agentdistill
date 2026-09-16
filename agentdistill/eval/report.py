"""Rendering for eval runs and comparisons."""

from __future__ import annotations

from typing import Any


def _pp(x: float) -> str:
    return f"{x * 100:+.1f} pp"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_comparison(c: dict) -> str:
    """The section 7.3 report, plus divergence and schema validity for both sides."""
    a, b = c["subject_a"], c["subject_b"]
    ma, mb = c["metrics_a"], c["metrics_b"]
    s = c["success"]
    lo, hi = s["ci95"]
    holm_success = c["holm"]["success"]

    lines: list[str] = []
    lines.append(f"{a}  vs  {b}")
    lines.append(f"eval set {c['eval_set_id']}, {c['n_shared_tasks']} shared tasks, "
                 f"n={ma.get('n_per_task', '?')} per task")
    lines.append("")
    lines.append(
        f"Task success        {b} {_pct(s['mean_b'])}   {a} {_pct(s['mean_a'])}   "
        f"delta {_pp(s['delta'])}  [95% CI {_pp(lo)}, {_pp(hi)}]  "
        f"McNemar p={c['mcnemar']['p']:.3g}"
    )
    lines.append(
        f"Schema validity     {b} {_pct(mb.get('schema_valid', float('nan')))}   "
        f"{a} {_pct(ma.get('schema_valid', float('nan')))}"
    )
    lines.append(
        f"Divergence rate     {b} {_pct(mb.get('divergence_rate', 0.0))}   "
        f"{a} {_pct(ma.get('divergence_rate', 0.0))}"
    )

    # `median_delta` is the median of the per-task differences, which is not the difference of the medians and
    # can legitimately be zero while the medians differ. Labelling it explicitly avoids a reader treating the
    # line as self-contradictory.
    t = c["tokens"]
    lines.append(
        f"Tokens / task       median {b} {t['median_b']:.0f}   {a} {t['median_a']:.0f}   "
        f"median per-task diff {t['median_delta']:+.0f}  Wilcoxon p={t['p']:.3g}"
    )
    tn = c["turns"]
    lines.append(
        f"Turns / task        median {b} {tn['median_b']:.1f}   {a} {tn['median_a']:.1f}   "
        f"median per-task diff {tn['median_delta']:+.1f}  Wilcoxon p={tn['p']:.3g}"
    )
    lines.append("")

    flags = ", ".join(
        f"{k}={'significant' if v['significant'] else 'not significant'} (p_adj={v['p_adjusted']:.3g})"
        for k, v in c["holm"].items()
    )
    lines.append(f"Holm-corrected across {len(c['holm'])} metrics: {flags}")

    verdict = _verdict(s, holm_success, a, b)
    lines.append("")
    lines.append(verdict)

    # Only real clusters are worth naming. "none" means these traces were never clustered -- eval traces are
    # excluded from curation, which is where clustering happens -- so reporting it as a weak cluster would point
    # a reader at a bucket that does not exist.
    weak = [w for w in (c.get("weakest_clusters") or []) if w["delta"] < 0 and w["cluster"] != "none"]
    if weak:
        lines.append("")
        lines.append("Weakest clusters for " + a + ":")
        for w in weak:
            lines.append(
                f"  #{w['cluster']}  {a} {_pct(w['success_a'])} vs {b} {_pct(w['success_b'])}  "
                f"({_pp(w['delta'])}, n={w['n_tasks']} tasks)"
            )
    elif any(w["cluster"] == "none" for w in (c.get("weakest_clusters") or [])):
        lines.append("")
        lines.append(
            "Per-cluster breakdown unavailable: these eval traces carry no cluster. Clustering happens during "
            "curation, which eval traces are excluded from. Assign clusters to the eval set to get this table."
        )

    replay_a = ma.get("replay") or {}
    if replay_a.get("fuzzy_share"):
        lines.append("")
        lines.append(
            f"NOTE: {_pct(replay_a['fuzzy_share'])} of {a}'s tool results were served by fuzzy replay "
            f"(lowest similarity {replay_a.get('min_fuzzy_score')}). A fuzzily replayed success is not a real "
            f"success; re-check with the predicate before reporting this number."
        )
    return "\n".join(lines)


def _verdict(s: dict, holm_success: dict, a: str, b: str) -> str:
    lo, hi = s["ci95"]
    delta = s["delta"]
    if lo <= 0 <= hi:
        return (
            f"VERDICT: the interval on the success delta includes zero, so this run does not show a difference "
            f"between {a} and {b}. That is not the same as showing they are equivalent -- with "
            f"{s['n_tasks']} tasks the interval is {_pp(lo)} to {_pp(hi)} wide."
        )
    direction = "better" if delta > 0 else "worse"
    strength = "and it survives Holm correction" if holm_success["significant"] else (
        "but it does not survive Holm correction across the reported metrics"
    )
    return f"VERDICT: {a} is {direction} than {b} by {_pp(delta)} [{_pp(lo)}, {_pp(hi)}], {strength}."


def render_run(run: dict) -> str:
    m = run["metrics"]
    lines = [
        f"{run['subject']}   eval set {run['eval_set_id']}   run {run['id']}",
        f"  tasks {m.get('n_tasks', '?')} x {m.get('n_per_task', '?')} repeats = {m.get('n_rows', '?')} rows",
        f"  success          {_pct(m.get('success', float('nan')))}",
        f"  schema validity  {_pct(m.get('schema_valid', float('nan')))}",
        f"  divergence rate  {_pct(m.get('divergence_rate', 0.0))}",
        f"  turns (median)   {m.get('turns_median', '?')}",
        f"  tokens (median)  {m.get('tokens_est_median', '?')}",
    ]
    stops = m.get("stop_reasons") or {}
    if stops:
        lines.append("  stop reasons     " + ", ".join(f"{k}={v}" for k, v in sorted(stops.items())))
    replay = m.get("replay") or {}
    if replay.get("fuzzy"):
        lines.append(
            f"  fuzzy replay     {replay['fuzzy']} results ({_pct(replay['fuzzy_share'])} of those served)"
        )
    return "\n".join(lines)


def render_run_markdown(run: dict, comparison: dict | None = None) -> str:
    """A committable report, for `examples/*/reports/`."""
    lines = [f"# Eval: {run['subject']}", "", "```", render_run(run), "```", ""]
    if comparison:
        lines += ["## Paired comparison", "", "```", render_comparison(comparison), "```", ""]
    return "\n".join(lines)


def per_task_table(rows: list[dict]) -> list[tuple[str, str, str, str]]:
    """(task, success rate, divergence rate, a failure detail) per task, for hand review."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    out = []
    for task, task_rows in sorted(by_task.items()):
        n = len(task_rows)
        succ = sum(bool(r["success"]) for r in task_rows)
        div = sum(bool(r["diverged"]) for r in task_rows)
        detail = next((r["grader_detail"] for r in task_rows if not r["success"] and r["grader_detail"]), "")
        out.append((task, f"{succ}/{n}", f"{div}/{n}", detail))
    return out


def summarize_divergences(rows: list[dict], limit: int = 10) -> list[dict]:
    """Most common divergences, with the nearest recorded call.

    A high nearest-score means an argument-phrasing gap that belongs in a per-tool normalization rule; a low one
    means the student genuinely went elsewhere.
    """
    counts: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = r.get("divergence")
        if not d:
            continue
        entry = counts.setdefault(d["tool"], {"tool": d["tool"], "n": 0, "max_score": 0.0, "example": d})
        entry["n"] += 1
        entry["max_score"] = max(entry["max_score"], d.get("nearest_score") or 0.0)
    return sorted(counts.values(), key=lambda e: -e["n"])[:limit]
