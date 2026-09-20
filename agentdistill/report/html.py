"""The HTML report.

One self-contained file: inline CSS, inline SVG, no JavaScript, no external assets. It has to survive being
attached to an email and opened on a machine with no network, because that is how a cost report actually gets
read.

Built with `html.escape` and f-strings rather than a template engine so there is no separate template file to
drift from the data it renders.
"""

from __future__ import annotations

from html import escape
from typing import Any

from agentdistill.report.assemble import SUBJECT_ORDER, ReportData
from agentdistill.report.markdown import (
    comparison_text,
    cost_unbatched,
    money,
    num,
    onpolicy_details,
    onpolicy_summary,
    pct,
    provenance_line,
)
from agentdistill.report.svg import cost_success_chart, reliability_chart

CSS = """
:root { --fg:#1a1a1a; --muted:#666; --line:#e3e3e3; --warn-bg:#fff8e1; --warn-br:#e6c200; --bad:#c0392b; }
* { box-sizing: border-box; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--fg); max-width: 60rem; margin: 2.5rem auto; padding: 0 1.25rem; }
h1 { font-size: 1.6rem; margin-bottom: .2rem; }
h2 { font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 1px solid var(--line); padding-bottom: .3rem; }
.sub { color: var(--muted); margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .92rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }
th { font-weight: 600; color: var(--muted); font-size: .82rem; text-transform: uppercase; letter-spacing: .03em; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
code { font: .85em ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); }
.warn { background: var(--warn-bg); border-left: 4px solid var(--warn-br); padding: .75rem 1rem; margin: 1rem 0; }
.warn ul { margin: .4rem 0 0; padding-left: 1.1rem; }
.headline { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0 0; }
.metric { min-width: 9rem; }
.metric .v { font-size: 1.5rem; font-variant-numeric: tabular-nums; }
.metric .k { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }
.charts { display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: flex-start; }
.note { color: var(--muted); font-size: .88rem; }
pre { background: #fafafa; border: 1px solid var(--line); padding: .8rem; overflow-x: auto; font-size: .84rem; }
.below { color: var(--bad); }
"""


def render(r: ReportData) -> str:
    """The whole report as one HTML string."""
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        f"<title>{escape(r.project)} — agentdistill report</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>{escape(r.project)}</h1>",
        f"<p class=\"sub\">Generated {escape(r.generated_at)} · eval set <code>{escape(r.eval_set)}</code></p>",
        _warnings(r),
        _headline(r),
        _subjects(r),
        _paired(r),
        _charts(r),
        _calibration(r),
        _per_cluster(r),
        _quantization(r),
        _onpolicy(r),
        _lineage(r),
        _commands(r),
        "</body></html>",
    ]
    return "".join(p for p in parts if p)


def _warnings(r: ReportData) -> str:
    if not r.warnings:
        return ""
    items = "".join(f"<li>{escape(str(w))}</li>" for w in r.warnings)
    return f'<div class="warn"><strong>Warnings</strong><ul>{items}</ul></div>'


def _metric(value: str, label: str) -> str:
    return f'<div class="metric"><div class="v">{escape(value)}</div><div class="k">{escape(label)}</div></div>'


def _headline(r: ReportData) -> str:
    student = r.subjects.get("student") or {}
    teacher = r.subjects.get("teacher") or {}
    cascade = (r.cost or {}).get("cascade") or {}
    tiles = [
        _metric(pct(student.get("success")), "student success"),
        _metric(pct(teacher.get("success")), "teacher success"),
    ]
    if cascade and cost_unbatched(r):
        # No price tiles: the student's cost is an upper bound, and a headline tile is the number people quote.
        tiles.append(_metric(pct(cascade.get("escalation_rate")), "escalation"))
        conditions = escape(str(r.cost.get("throughput_conditions", "unstated")))
        return (f'<div class="headline">{"".join(tiles)}</div>'
                f'<p class="note">The cascade is not priced against the teacher: student throughput was measured '
                f"under “{conditions}”, so its cost per token is an upper bound.</p>")
    if cascade:
        tiles += [
            _metric(money(cascade.get("cost_per_task")), "cascade $/task"),
            _metric(money(r.cost.get("teacher_cost_per_task")), "teacher $/task"),
            _metric(pct(cascade.get("saving_frac")), "saving"),
            _metric(pct(cascade.get("escalation_rate")), "escalation"),
        ]
    return f'<div class="headline">{"".join(tiles)}</div>'


def _onpolicy(r: ReportData) -> str:
    o = r.onpolicy or {}
    if not o:
        return ""
    summary = escape(onpolicy_summary(o)).replace("**", "")
    items = "".join(f"<li>{escape(line[2:])}</li>" for line in onpolicy_details(o))
    return f"<h2>On-policy round</h2><p>{_code_spans(summary)}</p><ul>{items}</ul>"


def _code_spans(escaped: str) -> str:
    """Backtick spans from the shared markdown sentence, as <code>. Input is already escaped."""
    parts = escaped.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def _subjects(r: ReportData) -> str:
    if not r.subjects:
        return ""
    rows = []
    for name in SUBJECT_ORDER:
        s = r.subjects.get(name)
        if not s:
            continue
        rows.append(
            f"<tr><td>{escape(name)}</td>"
            f'<td class="n">{pct(s["success"])}</td>'
            f'<td class="n">{pct(s["schema_valid"])}</td>'
            f'<td class="n">{pct(s["divergence_rate"])}</td>'
            f'<td class="n">{num(s["tokens_median"])}</td>'
            f"<td><code>{escape(str(s['run_id']))}</code></td></tr>"
        )
    return (
        "<h2>Subjects</h2><table><tr><th>Subject</th><th class=\"n\">Success</th>"
        "<th class=\"n\">Schema valid</th><th class=\"n\">Divergence</th>"
        "<th class=\"n\">Tokens/task</th><th>Run</th></tr>" + "".join(rows) + "</table>"
    )


def _paired(r: ReportData) -> str:
    if not r.paired:
        return ""
    blocks = []
    for name, cmp in r.paired.items():
        blocks.append(f"<p><strong>{escape(name.replace('_', ' '))}:</strong> {escape(comparison_text(cmp))}</p>")
    return "<h2>Paired comparisons</h2>" + "".join(blocks)


def _charts(r: ReportData) -> str:
    cascade = r.cascade or {}
    analytic = cascade.get("analytic") or []
    verified = cascade.get("verified") or []
    teacher_point = None
    teacher = r.subjects.get("teacher")
    if teacher and (r.cost or {}).get("teacher_cost_per_task") is not None:
        teacher_point = (r.cost["teacher_cost_per_task"], teacher.get("success") or 0.0)
    if not (analytic or verified or (r.calibration or {}).get("bins")):
        return ""
    charts = []
    if analytic or verified:
        charts.append(cost_success_chart(analytic, verified, teacher_point))
    bins = (r.calibration or {}).get("bins") or []
    if bins:
        charts.append(reliability_chart(bins))
    note = ""
    if verified:
        note = '<p class="note">Bold points are measured through the harness. The faint curve is the analytic ' \
               'estimate, which assumes an escalated turn is as good as the teacher\'s — it is not, because the ' \
               'teacher answers on a prefix the student built.</p>'
    return f'<h2>Cost and calibration</h2><div class="charts">{"".join(charts)}</div>{note}'


def _calibration(r: ReportData) -> str:
    cal = r.calibration or {}
    holdout = cal.get("holdout") or {}
    if not holdout:
        return ""
    return (
        "<h2>Gate</h2><p>"
        f"Holdout AUROC {num(holdout.get('auroc'), 3)}, ECE {num(holdout.get('ece'), 3)}, "
        f"Brier {num(holdout.get('brier'), 3)} on {num(holdout.get('n'))} turns. "
        f"Threshold {num(cal.get('threshold'), 2)}. "
        f"{_verdict_text(cal)}"
        f"<code>{escape(str(cal.get('id', '')))}</code></p>"
        '<p class="note">Metrics come from a task-disjoint split the gate never saw; in-sample calibration error '
        "is optimistic by construction.</p>"
    )


def _verdict_text(cal: dict) -> str:
    verdict = cal.get("verdict")
    reason = f" ({escape(cal['verdict_reason'])})" if cal.get("verdict_reason") else ""
    if verdict in (None, "usable"):
        return f"Verdict {escape(str(verdict))}{reason}. " if verdict else ""
    note = f" {escape(cal['note'])}." if cal.get("note") else ""
    return (f'<span class="below">Verdict {escape(verdict)}{reason}: the gateway refuses this gate and escalates '
            f"every turn.</span>{note} ")


def _per_cluster(r: ReportData) -> str:
    if not r.per_cluster:
        return ""
    rows = []
    for row in r.per_cluster:
        # Nested quotes inside an f-string expression are 3.12+; this package supports 3.11.
        routing_class = ' class="below"' if row.get("routing", "").startswith("teacher") else ""
        routing = escape(row.get("routing", ""))
        rows.append(
            f"<tr><td>{escape(str(row['cluster']))}</td>"
            f'<td class="n">{num(row.get("n_tasks"))}</td>'
            f'<td class="n">{pct(row.get("base"))}</td>'
            f'<td class="n">{pct(row.get("student"))}</td>'
            f'<td class="n">{pct(row.get("teacher"))}</td>'
            f"<td{routing_class}>{routing}</td></tr>"
        )
    return (
        "<h2>Per cluster</h2><table><tr><th>Cluster</th><th class=\"n\">Tasks</th><th class=\"n\">Base</th>"
        "<th class=\"n\">Student</th><th class=\"n\">Teacher</th><th>Routing</th></tr>"
        + "".join(rows) + "</table>"
        '<p class="note">Clusters below the router floor route to the teacher. These are where the student '
        "cannot be trusted, and where the next curation round should add data.</p>"
    )


def _quantization(r: ReportData) -> str:
    q = r.quantization or {}
    if not q:
        return ""
    delta = q.get("delta_pp")
    detail = f"{delta:+.1f} pp against bf16" if delta is not None else "not evaluated"
    return (
        f"<h2>Quantization</h2><p>{escape(str(q.get('method', '?')))}: {detail}"
        + (f" (<code>{escape(str(q['run_id']))}</code>)" if q.get("run_id") else "")
        + "</p>"
    )


def _lineage(r: ReportData) -> str:
    lin = r.lineage or {}
    if not lin:
        return ""
    items = []
    adapter = lin.get("adapter") or {}
    items.append(
        f"Adapter <code>{escape(str(adapter.get('id')))}</code> "
        f"({escape(str(adapter.get('name')))} v{adapter.get('version')}), status {escape(str(adapter.get('status')))}"
    )
    dataset = lin.get("dataset")
    if dataset:
        items.append(
            f"Dataset <code>{escape(dataset['name'])}</code> v{dataset['version']}, "
            f"{dataset['n_samples']} samples, hash <code>{escape(dataset['content_hash'][:12])}</code>"
        )
    run = lin.get("training_run")
    if run:
        items.append(f"Training run <code>{escape(str(run['id']))}</code> ({escape(str(run['method']))})")
    for parent in lin.get("parents", []):
        items.append(f"Parent <code>{escape(str(parent['id']))}</code>")
    return "<h2>Lineage</h2><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def _commands(r: ReportData) -> str:
    if not r.commands:
        return ""
    lines = []
    for c in r.commands:
        lines.append(escape(c["command"]))
        note = provenance_line(c.get("provenance"))
        if note:
            lines.append(f"# {escape(note)}")
    body = "\n".join(lines)
    return f"<h2>How to reproduce</h2><pre>{body}</pre>"


def write(r: ReportData, path: Any) -> Any:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(r))
    return p
