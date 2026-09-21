"""Every warning code is placed: known to `ReportData.warn`, and allowed or forbidden by the rehearsal gate.

`assert_report` fails an unexpected warning only when it is given allow patterns, and a forbidden one only when it
is listed. A new code in neither list of `clean_rehearsal.sh` would pass the rehearsal silently -- which is exactly
the class of silence this project keeps finding.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentdistill.report.assemble import WARNING_CODES, ReportData
from agentdistill.report.warnings import CODES

ROOT = Path(__file__).resolve().parents[1]


def _script_lists() -> tuple[set[str], set[str]]:
    text = (ROOT / "scripts" / "clean_rehearsal.sh").read_text()

    def bash_array(name: str) -> set[str]:
        m = re.search(rf"^{name}=\(([^)]*)\)", text, re.MULTILINE | re.DOTALL)
        assert m, f"clean_rehearsal.sh defines no {name} array"
        return set(m.group(1).split())

    return bash_array("ALLOW"), bash_array("FORBID")


def test_the_code_list_has_one_home():
    """`warnings.CODES` is the list; `assemble.WARNING_CODES` re-exports it so existing imports keep working."""
    assert WARNING_CODES is CODES


def test_every_code_the_report_emits_is_declared():
    source = "\n".join(p.read_text() for p in (ROOT / "agentdistill" / "report").glob("*.py"))
    emitted = set(re.findall(r'\.warn\(\s*"([a-z_]+)"', source))
    assert emitted, "found no warn() calls; the audit's pattern no longer matches the code"
    assert emitted <= set(WARNING_CODES), f"undeclared: {sorted(emitted - set(WARNING_CODES))}"


def test_every_declared_code_is_placed_exactly_once_in_the_rehearsal_gate():
    allow, forbid = _script_lists()
    assert not (allow & forbid), f"both allowed and forbidden: {sorted(allow & forbid)}"
    unplaced = set(WARNING_CODES) - allow - forbid
    assert not unplaced, f"in neither list of clean_rehearsal.sh, so they pass silently: {sorted(unplaced)}"
    assert (allow | forbid) <= set(WARNING_CODES), f"stale codes in the script: {sorted((allow | forbid) - set(WARNING_CODES))}"


def test_an_unknown_code_is_refused_at_the_source():
    with pytest.raises(ValueError, match=r"clean_rehearsal\.sh"):
        ReportData(generated_at="t", project="p", eval_set="e").warn("made_up_code", "x")


def test_the_gpu_day_forbids_what_tiny_mode_allows():
    """Tiny mode's disclosures are findings on a real run: degenerate labels, a replay teacher, unbatched cost."""
    text = (ROOT / "docs" / "gpu-day.md").read_text()
    for code in ("gate_degenerate", "replay_teacher", "cost_unbatched"):
        assert f"--forbid-warning {code}" in text, f"docs/gpu-day.md does not forbid {code} on the day"
