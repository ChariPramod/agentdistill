"""The curation report.

Every dataset carries one. It answers, in order: how many traces went in, what each filter dropped and why, what
the surviving data looks like, and which clusters are too thin to trust. A reviewer who reads only this file
should be able to say whether the dataset is worth training on.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from agentdistill.curate.pipeline import CurationResult
from agentdistill.curate.stratify import tool_sequence


def _bar(n: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return ""
    filled = round(width * n / total)
    return "█" * filled + "·" * (width - filled)


def _histogram(values: list[int], bins: list[int], label: str) -> list[str]:
    if not values:
        return [f"_no {label} recorded_"]
    counts: Counter[int] = Counter()
    for v in values:
        bucket = bins[-1]
        for b in bins:
            if v <= b:
                bucket = b
                break
        counts[bucket] += 1
    total = len(values)
    lines = ["| range | n | share | |", "|---|---:|---:|---|"]
    prev = 0
    for b in bins:
        n = counts.get(b, 0)
        rng = f"{prev + 1}–{b}" if b != bins[-1] else f"{prev + 1}+"
        lines.append(f"| {rng} | {n} | {n / total:.1%} | {_bar(n, total)} |")
        prev = b
    return lines


def render(result: CurationResult, dataset_name: str, version: int, traces: list[dict] | None = None) -> str:
    traces = traces if traces is not None else result.traces
    kept, dropped = result.n_output, result.n_input - result.n_output
    keep_rate = result.n_output / result.n_input if result.n_input else 0.0

    L: list[str] = []
    L.append(f"# Curation report: {dataset_name} v{version}")
    L.append("")
    L.append(f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_")
    L.append("")
    L.append(f"**{result.n_input} traces in → {kept} kept ({keep_rate:.1%}), {dropped} dropped.**")
    L.append("")

    if result.notes:
        L.append("## Read this first")
        L.append("")
        for note in result.notes:
            L.append(f"- {note}")
        L.append("")

    L.append("## What each filter dropped")
    L.append("")
    L.append("| filter | in | dropped | out | top reason |")
    L.append("|---|---:|---:|---:|---|")
    for s in result.stages:
        top = s.reasons.most_common(1)
        reason = f"{top[0][0]} ({top[0][1]})" if top else "—"
        L.append(f"| `{s.name}` | {s.n_in} | {s.n_dropped} | {s.n_out} | {reason} |")
    L.append("")

    detailed = [s for s in result.stages if s.n_dropped]
    if detailed:
        L.append("### Drop reasons in detail")
        L.append("")
        for s in detailed:
            L.append(f"**`{s.name}`** — {s.n_dropped} dropped")
            L.append("")
            for reason, n in s.reasons.most_common(8):
                example = s.examples.get(reason, "")
                L.append(f"- {n} × {reason}" + (f"  \n  _e.g._ `{example}`" if example else ""))
            if len(s.reasons) > 8:
                L.append(f"- _…and {len(s.reasons) - 8} more distinct reasons_")
            L.append("")

    L.append("## What survived")
    L.append("")
    turns = [t.get("n_turns") or 0 for t in traces]
    calls = [t.get("n_tool_calls") or 0 for t in traces]
    L.append("### Assistant turns per trace")
    L.append("")
    L.extend(_histogram(turns, [1, 2, 4, 8, 16, 32, 64], "turns"))
    L.append("")
    L.append("### Tool calls per trace")
    L.append("")
    L.extend(_histogram(calls, [0, 1, 2, 4, 8, 16, 32], "tool calls"))
    L.append("")

    tool_freq: Counter[str] = Counter()
    for t in traces:
        for m in t["messages"]:
            for c in m.get("tool_calls") or []:
                tool_freq[c["function"]["name"]] += 1
    if tool_freq:
        L.append("### Tool-call frequency")
        L.append("")
        L.append("| tool | calls | share | |")
        L.append("|---|---:|---:|---|")
        total_calls = sum(tool_freq.values())
        for tool_name, n in tool_freq.most_common(20):
            L.append(f"| `{tool_name}` | {n} | {n / total_calls:.1%} | {_bar(n, total_calls)} |")
        L.append("")

    seq_freq: Counter[tuple[str, ...]] = Counter(tool_sequence(t) for t in traces)
    if seq_freq:
        L.append("### Most common tool sequences")
        L.append("")
        for seq, n in seq_freq.most_common(10):
            rendered = " → ".join(seq) if seq else "_(no tool calls)_"
            L.append(f"- {n} × {rendered}")
        L.append("")

    teachers = Counter(t.get("teacher_model") or "(unrecorded)" for t in traces)
    if len(teachers) > 1 or "(unrecorded)" not in teachers:
        L.append("### Teacher models")
        L.append("")
        for teacher_name, n in teachers.most_common():
            L.append(f"- {n} × `{teacher_name}`")
        L.append("")

    cov = result.coverage
    if cov:
        L.append("## Cluster coverage")
        L.append("")
        L.append(
            f"{cov['n_clusters']} clusters; **{cov['n_sparse']} have fewer than {cov['sparse_threshold']} "
            f"samples** ({cov['sparse_share']:.0%}). Thin clusters are where the student fails first: they need "
            f"more data, a per-cluster eval, and the router floor."
        )
        L.append("")
        if result.embedder_name:
            L.append(f"_Embedder: `{result.embedder_name}`_")
            L.append("")
        L.append("| cluster | before cap | after cap | | top tool sequence |")
        L.append("|---:|---:|---:|---|---|")
        biggest = max((c["after"] for c in cov["clusters"]), default=1) or 1
        for c in sorted(cov["clusters"], key=lambda c: -c["after"]):
            top = c["top_tool_sequences"]
            top_seq = " → ".join(top[0]["sequence"]) if top else "—"
            flag = " ⚠️" if c["sparse"] else ""
            L.append(
                f"| {c['cluster']}{flag} | {c['before']} | {c['after']} | {_bar(c['after'], biggest)} | {top_seq} |"
            )
        L.append("")

    L.append("## Reproducing this dataset")
    L.append("")
    L.append("The filter config below, applied to the same trace corpus, reproduces this dataset exactly.")
    L.append("")
    L.append("```json")
    import json

    L.append(json.dumps(result.filter_config, indent=2, sort_keys=True))
    L.append("```")
    L.append("")
    return "\n".join(L)


def write(result: CurationResult, dataset_name: str, version: int, path: Any, traces: list[dict] | None = None) -> Any:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(result, dataset_name, version, traces))
    return p
