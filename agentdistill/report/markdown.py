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
    return "n/a" if x is None else f"{x:,.{places}f}"


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
        success = paired["success"]
        tokens = paired.get("tokens") or {}
        lines += [
            f"**Student vs teacher:** {success['delta'] * 100:+.1f} pp{ci_pp(success['ci95'])}, "
            f"McNemar p={paired['mcnemar']['p']:.3g}; tokens {tokens.get('median_delta', 0):+.0f}"
            f"{ci_raw(tokens.get('ci95'))} per task.",
            "",
        ]

    cascade = (r.cost or {}).get("cascade")
    if cascade:
        lines += [
            f"**Cascade** at threshold {cascade['threshold']:.2f}: success {pct(cascade['success'])}, "
            f"escalation {pct(cascade['escalation_rate'])}, "
            f"{money(cascade['cost_per_task'])}/task against the teacher's "
            f"{money(r.cost.get('teacher_cost_per_task'))} "
            f"({pct(cascade['saving_frac'])} saving). Break-even "
            f"{num(cascade['breakeven_tasks_per_day'])} tasks/day on the configured GPU.",
            "",
        ]

    cal = r.calibration
    if cal and cal.get("holdout"):
        lines += [
            f"**Gate:** holdout AUROC {num(cal['holdout'].get('auroc'), 3)}, "
            f"ECE {num(cal['holdout'].get('ece'), 3)}, threshold {num(cal.get('threshold'), 2)} "
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
        lines += [c["command"] for c in r.commands]
        lines += ["```", ""]

    return "\n".join(lines)
