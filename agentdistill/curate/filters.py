"""Per-trace filter predicates.

Each returns `(keep, reason)`. The reason is what the curation report prints, so it names the specific problem
rather than the filter: "4 consecutive tool errors" is actionable, "no_error_loops" is not.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agentdistill.curate.schema import tool_calls_valid, tool_results_paired
from agentdistill.ingest.normalize import canonical_arguments

#: Substrings that mark a tool result as an error. Deliberately narrow: a result *about* an error ("the customer
#: reported an error") is not a failed call, and over-matching would throw away good traces.
_ERROR_PREFIXES = ("error:", "error ", "exception:", "traceback", "failed:", "tool error")


def is_error_result(message: dict) -> bool:
    content = (message.get("content") or "").strip()
    if not content:
        return False
    low = content.lower()
    if low.startswith(_ERROR_PREFIXES):
        return True
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    if isinstance(parsed, dict):
        if parsed.get("error") not in (None, False, "", []):
            return True
        status = parsed.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            return True
        if isinstance(parsed.get("status_code"), int) and parsed["status_code"] >= 400:
            return True
    return False


def outcome(trace: dict, *, require: bool = True) -> tuple[bool, str]:
    """SFT learns from successes only. A trace with no recorded outcome is not evidence of anything."""
    if not require:
        return True, ""
    if trace.get("success") is True:
        return True, ""
    if trace.get("success") is None:
        return False, "no recorded outcome"
    return False, "task failed"


def schema_valid(trace: dict) -> tuple[bool, str]:
    ok, problems = tool_calls_valid(trace)
    if not ok:
        return False, problems[0]
    ok, problems = tool_results_paired(trace)
    if not ok:
        return False, problems[0]
    return True, ""


def no_error_loops(trace: dict, *, max_consecutive: int = 3, max_repeats: int = 2) -> tuple[bool, str]:
    """Drop traces that flail: a run of failing tool calls, or the same call issued over and over.

    Both patterns teach the student to keep hammering a tool that is not working.
    """
    run = 0
    for m in trace["messages"]:
        if m["role"] != "tool":
            continue
        if is_error_result(m):
            run += 1
            if run >= max_consecutive:
                return False, f"{run} consecutive tool errors"
        else:
            run = 0

    calls: Counter[tuple[str, str]] = Counter()
    for m in trace["messages"]:
        for c in m.get("tool_calls") or []:
            key = (c["function"]["name"], canonical_arguments(c["function"]["arguments"]))
            calls[key] += 1
            if calls[key] >= max_repeats:
                return False, f"identical call to {key[0]} repeated {calls[key]} times"
    return True, ""


def length(trace: dict, *, min_turns: int = 2, max_turns: int = 40) -> tuple[bool, str]:
    """Turn-count bounds. The token bound needs a tokenizer and is applied at dataset build, where a too-long
    trajectory becomes turn_window samples instead of being dropped."""
    n = trace.get("n_turns") or sum(1 for m in trace["messages"] if m["role"] == "assistant")
    if n < min_turns:
        return False, f"{n} assistant turns, below min_turns={min_turns}"
    if n > max_turns:
        return False, f"{n} assistant turns, above max_turns={max_turns}"
    return True, ""


def teacher(trace: dict, *, models: list[str]) -> tuple[bool, str]:
    if not models:
        return True, ""
    tm = trace.get("teacher_model")
    if tm in models:
        return True, ""
    return False, f"teacher_model {tm!r} not in {models}"


def quality_judge(trace: dict, *, min_score: float = 3.0, scores: dict[str, float] | None = None) -> tuple[bool, str]:
    """Optional. Reads a precomputed judge score; the judging itself is a separate, paid step.

    A trace with no score is kept rather than dropped, so that enabling the filter without having run the judge
    does not silently empty the dataset.
    """
    if not scores:
        return True, ""
    s = scores.get(trace["id"])
    if s is None:
        return True, ""
    if s < min_score:
        return False, f"judge score {s} below {min_score}"
    return True, ""


#: Filters that look at one trace at a time. The set-level ones (dedupe, decontaminate, stratify) live in the
#: pipeline because they need the whole corpus.
PER_TRACE: dict[str, Any] = {
    "outcome": outcome,
    "schema_valid": schema_valid,
    "no_error_loops": no_error_loops,
    "length": length,
    "teacher": teacher,
    "quality_judge": quality_judge,
}
