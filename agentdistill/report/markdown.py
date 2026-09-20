"""The results block, and injecting it into the README.

This is the only way numbers reach the README. The rule is not bureaucratic: a number typed by hand is a number
that stays after the run that produced it is gone, and every row here carries the run id that produced it so
anyone can go and check.
"""

from __future__ import annotations

from agentdistill.report.assemble import SUBJECT_ORDER, ReportData

BEGIN = "<!-- agentdistill:results:begin -->"
END = "<!-- agentdistill:results:end -->"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def num(x: float | None, places: int = 0) -> str:
    # NaN reads as n/a too: an undefined AUROC from an older row must not print as "nan".
    return "n/a" if x is None or x != x else f"{x:,.{places}f}"


def money(x: float | None, places: int = 4) -> str:
    return "n/a" if x is None else f"${x:.{places}f}"


def ci_pp(interval) -> str:
    if not interval or interval[0] is None:
        return ""
    return f" [{interval[0] * 100:+.1f}, {interval[1] * 100:+.1f}]"


def ci_raw(interval) -> str:
    if not interval or interval[0] is None or interval[0] != interval[0]:
        return ""
    return f" [{interval[0]:+.0f}, {interval[1]:+.0f}]"


def comparison_line(name: str, p: dict) -> str:
    return f"{name}: {comparison_text(p)}"


def comparison_text(p: dict) -> str:
    """One paired comparison as a sentence. The only place either renderer formats one, so the rule that an
    insufficient-power result never carries a p-value is enforced once."""
    w = p.get("insufficient_power")
    if w:
        o = p.get("observed") or {}
        rates = ""
        if o.get("rate_a") is not None and o.get("rate_b") is not None:
            rates = f" Observed: {o['rate_a'] * 100:.1f}% vs {o['rate_b'] * 100:.1f}%, not compared."
        return f"{w['reason']}.{rates}"
    success = p["success"]
    tokens = p.get("tokens") or {}
    return (f"{success['delta'] * 100:+.1f} pp{ci_pp(success['ci95'])}, "
            f"McNemar p={p['mcnemar']['p']:.3g}; tokens {tokens.get('median_delta', 0):+.0f}"
            f"{ci_raw(tokens.get('ci95'))} per task.")


def provenance_line(prov: dict | None) -> str | None:
    """`commit <sha>[ (dirty)] config <path>@<hash>` for a recorded command, or None if nothing was recorded.

    Both renderers print it as a comment under the command so the reproduce block stays pasteable.
    """
    if not prov:
        return None
    parts = []
    if prov.get("commit") or prov.get("dirty"):
        parts.append(f"commit {prov.get('commit') or 'unknown'}" + (" (dirty)" if prov.get("dirty") else ""))
    if prov.get("config_path"):
        parts.append(f"config {prov['config_path']}@{prov.get('config_hash') or 'unhashed'}")
    return " ".join(parts) or None


def cost_unbatched(r: ReportData) -> bool:
    """Whether the student's cost comes from an unbatched measurement. Either signal is enough: the code is what
    `assemble` raised, the mode is what the cost block recorded, and a hand-built report may carry only one."""
    cost = r.cost or {}
    return "cost_unbatched" in r.warning_codes or (bool(cost) and cost.get("throughput_mode") != "batched")


def onpolicy_summary(o: dict) -> str:
    """One sentence for the latest on-policy round. Shared by both renderers so a discard reads the same in each."""
    decision = o.get("decision") or "unknown"
    verb = {"promote": "promoted", "discard": "discarded", "error": "errored"}.get(decision, decision)
    text = f"round {o.get('round_idx')} (`{o.get('round_id')}`) from `{o.get('start_adapter')}` was **{verb}**"
    if o.get("candidate_adapter"):
        text += f" (candidate `{o['candidate_adapter']}`)"
    text += f": {o.get('reason') or 'no reason recorded'}."
    if o.get("insufficient_power") and o["insufficient_power"] not in (o.get("reason") or ""):
        text += f" The comparison was underpowered: {o['insufficient_power']}."
    if decision == "discard":
        text += " One round of RFT plus DPO did not beat its starting adapter on this data; that is a result."
    return text


def onpolicy_details(o: dict) -> list[str]:
    """The round's inputs as bullet lines: rollouts, fuzzy share, and the pair statistics the cap is checked by."""
    stats = o.get("pair_stats") or {}
    lines = [f"- Rollouts: {num(o.get('n_rollouts'))}; fuzzy-replay share {pct(o.get('fuzzy_share'))}"]
    if stats:
        lines.append(
            f"- Pairs: {num(stats.get('n_pairs'))} ({num(stats.get('n_rollout'))} rollout, "
            f"{num(stats.get('n_teacher'))} teacher); max per task {num(stats.get('max_per_task'))} "
            f"(cap {num(stats.get('cap_per_task'))})"
        )
        if stats.get("per_task_histogram"):
            hist = ", ".join(f"{k}: {v}" for k, v in stats["per_task_histogram"].items())
            lines.append(f"- Pairs per task (pairs: tasks): {hist}")
        if stats.get("diff_kind"):
            lines.append("- Pair differences: " + ", ".join(f"{k} {v}" for k, v in stats["diff_kind"].items()))
        lines += [f"- Pair warning: {w}" for w in stats.get("warnings") or []]
    else:
        lines.append("- No pair statistics were recorded for this round.")
    return lines


def results_block(r: ReportData) -> str:
    """The block between the markers. Every number sits on a row with its run id."""
    lines = [BEGIN, "", f"_Generated {r.generated_at} on eval set `{r.eval_set}`._", ""]

    if r.tiny:
        lines += ["> **Tiny mode.** These numbers come from the CPU rehearsal and measure nothing but that the "
                  "pipeline runs.", ""]

    if r.subjects:
        lines += [
            "| Subject | Success | Schema valid | Divergence | Tokens/task | Run |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for name in SUBJECT_ORDER:
            s = r.subjects.get(name)
            if not s:
                continue
            lines.append(
                f"| {name} | {pct(s['success'])} | {pct(s['schema_valid'])} | {pct(s['divergence_rate'])} | "
                f"{num(s['tokens_median'])} | `{s['run_id']}` |"
            )
        lines.append("")

    paired = r.paired.get("student_vs_teacher")
    if paired:
        lines += [f"**Student vs teacher:** {comparison_text(paired)}", ""]

    cascade = (r.cost or {}).get("cascade")
    if cascade and cost_unbatched(r):
        # Escalation rate and the measurement conditions only: a saving computed from an upper-bound cost is a
        # number with a known bias and an unknown size, and it would still be quoted.
        lines += [
            f"**Cascade** at threshold {cascade['threshold']:.2f}: success {pct(cascade['success'])}, "
            f"escalation {pct(cascade['escalation_rate'])}. Not priced against the teacher: student throughput "
            f"was measured under \"{r.cost.get('throughput_conditions', 'unstated')}\", so its cost per token is "
            f"an upper bound.",
            "",
        ]
    elif cascade:
        lines += [
            f"**Cascade** at threshold {cascade['threshold']:.2f}: success {pct(cascade['success'])}, "
            f"escalation {pct(cascade['escalation_rate'])}, "
            f"{money(cascade['cost_per_task'])}/task against the teacher's "
            f"{money(r.cost.get('teacher_cost_per_task'))} "
            f"({pct(cascade['saving_frac'])} saving). Break-even "
            f"{num(cascade['breakeven_tasks_per_day'])} tasks/day on the configured GPU "
            f"(throughput: {r.cost.get('throughput_conditions', 'unstated')}).",
            "",
        ]

    if r.onpolicy:
        lines += [f"**On-policy round:** {onpolicy_summary(r.onpolicy)}", ""]

    cal = r.calibration
    if cal and cal.get("holdout"):
        verdict = cal.get("verdict")
        tail = "" if verdict in (None, "usable") else f"; verdict **{verdict}**, so every turn escalates"
        if cal.get("verdict_reason"):
            tail += f" ({cal['verdict_reason']})"
        lines += [
            f"**Gate:** holdout AUROC {num(cal['holdout'].get('auroc'), 3)}, "
            f"ECE {num(cal['holdout'].get('ece'), 3)}, threshold {num(cal.get('threshold'), 2)}{tail} "
            f"(`{cal['id']}`).",
            "",
        ]

    if r.quantization and r.quantization.get("delta_pp") is not None:
        lines += [
            f"**Quantization** ({r.quantization['method']}): {r.quantization['delta_pp']:+.1f} pp against bf16 "
            f"(`{r.quantization['run_id']}`).",
            "",
        ]

    weak = [row for row in r.per_cluster if row.get("routing", "").startswith("teacher")]
    if weak:
        lines += [
            f"**{len(weak)} of {len(r.per_cluster)} clusters** sit below the router floor and route to the "
            f"teacher.",
            "",
        ]

    if r.warnings:
        lines += ["**Warnings:**", ""]
        lines += [f"- {w}" for w in r.warnings]
        lines.append("")

    lines.append(END)
    return "\n".join(lines)


def inject(readme_text: str, block: str) -> str:
    """Replace the block between the markers, or append a Results section if there are none.

    Idempotent: injecting twice leaves one block.
    """
    if BEGIN in readme_text and END in readme_text:
        head = readme_text[: readme_text.index(BEGIN)]
        tail = readme_text[readme_text.index(END) + len(END) :]
        return head + block + tail
    return readme_text.rstrip() + "\n\n## Results\n\n" + block + "\n"


def full_markdown(r: ReportData) -> str:
    """The standalone markdown report: the results block plus the detail that does not belong in a README."""
    lines = [f"# {r.project}: results", "", results_block(r), ""]

    if r.per_cluster:
        lines += ["## Per cluster", "",
                  "| Cluster | Tasks | Base | Student | Teacher | Routing |",
                  "|---|---:|---:|---:|---:|---|"]
        for row in r.per_cluster:
            lines.append(
                f"| {row['cluster']} | {num(row.get('n_tasks'))} | {pct(row.get('base'))} | "
                f"{pct(row.get('student'))} | {pct(row.get('teacher'))} | {row.get('routing', '')} |"
            )
        lines.append("")

    if r.onpolicy:
        lines += ["## On-policy round", "", onpolicy_summary(r.onpolicy), ""]
        lines += onpolicy_details(r.onpolicy)
        lines.append("")

    lin = r.lineage or {}
    if lin:
        lines += ["## Lineage", ""]
        adapter = lin.get("adapter", {})
        lines.append(f"- Adapter `{adapter.get('id')}` ({adapter.get('name')} v{adapter.get('version')}), "
                     f"status {adapter.get('status')}, base `{adapter.get('base_model')}`")
        dataset = lin.get("dataset")
        if dataset:
            lines.append(f"- Dataset `{dataset['name']}` v{dataset['version']}, {dataset['n_samples']} samples, "
                         f"hash `{dataset['content_hash'][:12]}`")
        run = lin.get("training_run")
        if run:
            lines.append(f"- Training run `{run['id']}` ({run['method']})")
        for parent in lin.get("parents", []):
            lines.append(f"- Parent `{parent['id']}` ({parent['name']} v{parent['version']})")
        lines.append("")

    if r.commands:
        lines += ["## How to reproduce", "", "```bash"]
        for c in r.commands:
            lines.append(c["command"])
            note = provenance_line(c.get("provenance"))
            if note:
                lines.append(f"# {note}")
        lines += ["```", ""]

    return "\n".join(lines)
