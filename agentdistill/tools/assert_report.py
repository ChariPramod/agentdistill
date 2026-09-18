"""Assert that a report says what a clean run should say.

The clean rehearsal ends by checking the report, not by eyeballing it: every required subject has a run id,
every required section is populated, and the only warnings are the disclosures tiny mode is expected to make.
It reads the `report.json` sidecar that `agentdistill report` writes, so the assertions are against data rather
than parsed HTML.

Warnings are matched by code first, because prose drifts. A pattern matches a warning when it equals the code
once spaces become underscores ("no run found" -> `no_run_found`), or when it is a case-insensitive substring
of the message. With no `--allow-warning` every warning not forbidden is tolerated; with at least one, any
warning matching none of them fails, which is how "only tiny-mode disclosures" is enforced.

    python -m agentdistill.tools.assert_report artifacts/gpu_day/report.html \\
        --require-subjects base,student,teacher --require-sections calibration,cascade,cost,quantization \\
        --forbid-warning "no run found" --allow-warning "tiny mode" --allow-warning "replay stub"

Exit 0 when every check passes, 1 with one line per failure otherwise, 2 when the sidecar cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: str | Path) -> dict:
    """The report data for a report path: the file itself if it is JSON, else the sibling `report.json`."""
    p = Path(path)
    if p.suffix != ".json":
        p = p.parent / "report.json"
    return json.loads(p.read_text())


def warnings_of(data: dict) -> list[tuple[str, str]]:
    """(code, message) pairs. A warning added without a code reads as `uncoded`, which no code pattern allows."""
    messages = [str(m) for m in data.get("warnings") or []]
    codes = list(data.get("warning_codes") or [])
    return [(codes[i] if i < len(codes) and codes[i] else "uncoded", m) for i, m in enumerate(messages)]


def matches(pattern: str, code: str, message: str) -> bool:
    normalized = pattern.strip().lower().replace(" ", "_").replace("-", "_")
    return normalized == code.lower() or pattern.strip().lower() in message.lower()


def section_problem(name: str, data: dict) -> str | None:
    """Why a required section does not count as populated, or None if it does."""
    if name == "calibration":
        cal = data.get("calibration") or {}
        if not cal.get("holdout"):
            return "calibration has no holdout metrics"
        if not cal.get("verdict"):
            return "calibration has no verdict"
        return None
    if name == "cascade":
        cascade = data.get("cascade") or {}
        # A measured point, even an escalate-everything one: an analytic estimate alone is not a cascade result.
        if not cascade.get("verified"):
            return "cascade has no verified point"
        if cascade.get("threshold") is None:
            return "cascade has no threshold"
        return None
    if name == "cost":
        cost = data.get("cost") or {}
        if cost.get("teacher_cost_per_task") is None:
            return "cost has no teacher_cost_per_task"
        if cost.get("student_cost_per_mtok") is None:
            return "cost has no student_cost_per_mtok"
        return None
    if name == "quantization":
        if (data.get("quantization") or {}).get("delta_pp") is None:
            return "quantization has no delta_pp"
        return None
    if not data.get(name):
        return f"section {name} is empty"
    return None


def check(
    data: dict,
    require_subjects: list[str] = (),
    require_sections: list[str] = (),
    forbid: list[str] = (),
    allow: list[str] = (),
) -> list[str]:
    """Every failure as one line; empty means the report passes."""
    failures: list[str] = []
    subjects = data.get("subjects") or {}
    for name in require_subjects:
        s = subjects.get(name)
        if not s:
            failures.append(f"subject {name} is missing")
        elif not s.get("run_id"):
            failures.append(f"subject {name} has no run id")
    for name in require_sections:
        problem = section_problem(name, data)
        if problem:
            failures.append(problem)
    for code, message in warnings_of(data):
        hit = next((p for p in forbid if matches(p, code, message)), None)
        if hit:
            failures.append(f"forbidden warning ({hit!r}) [{code}]: {message}")
        elif allow and not any(matches(p, code, message) for p in allow):
            failures.append(f"unexpected warning [{code}]: {message}")
    return failures


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentdistill.tools.assert_report", description=__doc__.split(
        "\n\n")[0])
    parser.add_argument("report", help="report.html (its sibling report.json is read) or report.json")
    parser.add_argument("--require-subjects", type=_csv, default=[], help="comma-separated, e.g. base,student")
    parser.add_argument("--require-sections", type=_csv, default=[],
                        help="comma-separated: calibration, cascade, cost, quantization, or any report field")
    parser.add_argument("--forbid-warning", action="append", default=[], help="code or message substring")
    parser.add_argument("--allow-warning", action="append", default=[], help="code or message substring")
    args = parser.parse_args(argv)

    try:
        data = load(args.report)
    except (OSError, ValueError) as e:
        print(f"cannot read report data for {args.report}: {e}", file=sys.stderr)
        return 2

    failures = check(data, args.require_subjects, args.require_sections, args.forbid_warning, args.allow_warning)
    for line in failures:
        print(line)
    if failures:
        return 1
    codes = sorted({code for code, _ in warnings_of(data)})
    print(f"report ok: subjects {sorted(data.get('subjects') or {})}, "
          f"{len(data.get('warnings') or [])} warning(s) {codes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
