"""Grading in the replay setting.

The predicates inspect a CRM database, but during replay there is no database -- tool results came from the
recording. The end state is reconstructed instead: build a fresh CRM from the task's seed and apply the student's
tool calls to it, in order.

This is exact for this project, and the reason is worth stating. Results during replay were served from the
recording, so a student that made the same calls as the teacher reaches the same state. A student that made
*different* calls diverged, and the predicate then sees whatever state its own calls produced -- which is the
right thing to grade, because those are the calls it would really have made.

The one thing this cannot capture: a call that was refused during reconstruction but would have succeeded
against the live service, or vice versa. Since reconstruction replays against the same seeded database the
recording was made from, that can only happen if the student's calls reach a state the recording never visited --
and those runs have already been counted as divergences.
"""

from __future__ import annotations

import json
from typing import Any

from examples.support_agent import scenarios
from examples.support_agent.crm import CRM


class UngradeableTrace(ValueError):
    """The trace does not carry what is needed to rebuild its task."""


def task_for_trace(trace: dict) -> Any:
    """Rebuild the scenario task a trace came from."""
    meta = trace.get("metadata") or {}
    scenario = meta.get("scenario")
    seed = meta.get("db_seed")
    if scenario is None or seed is None:
        raise UngradeableTrace(
            f"trace {trace.get('id')} has no scenario/db_seed in metadata, so its task cannot be rebuilt. "
            f"Traces recorded by examples/support_agent/record.py always carry both."
        )
    return scenarios.build_task(scenario, int(seed))


def apply_calls(crm: CRM, calls: list[tuple[str, dict]]) -> list[dict]:
    """Replay tool calls against a fresh CRM, recording what each one did.

    Refusals are captured rather than raised: a student that tried to refund an ineligible order made that
    mistake, and the resulting state -- no refund -- is what the predicate should judge.
    """
    log = []
    for name, args in calls:
        try:
            result = crm.call(name, args)
            log.append({"tool": name, "args": args, "ok": True, "result": result})
        except Exception as e:
            log.append({"tool": name, "args": args, "ok": False, "error": str(e)})
    return log


def predicate_from_calls(
    trace: dict, calls: list[tuple[str, dict]], final_text: str
) -> tuple[bool, str, dict]:
    """Grade a replayed trajectory. Returns (success, detail, extra)."""
    task = task_for_trace(trace)
    crm = task.fresh_crm()
    log = apply_calls(crm, calls)
    success, detail = task.predicate(crm, final_text)
    return (
        bool(success),
        detail,
        {
            "n_calls_applied": len(log),
            "n_calls_refused": sum(1 for entry in log if not entry["ok"]),
            "final_state_hash": crm.state_hash(),
            "scenario": task.scenario,
        },
    )


def grade_outcome(trace: dict, outcome: Any) -> tuple[bool, dict]:
    """Grader with the signature the eval runner expects: `(trace, outcome) -> (success, detail_dict)`."""
    from agentdistill.eval.harness import tool_calls_made

    success, detail, extra = predicate_from_calls(trace, tool_calls_made(outcome), outcome.final_text)
    return success, {"detail": detail, **extra}


def live_outcome(trace: dict) -> bool:
    """The success label recorded when the trace was made, for the equivalence test."""
    return bool(trace["success"])


def calls_from_trace(trace: dict) -> list[tuple[str, dict]]:
    """The tool calls the recorded trajectory made, in order."""
    out: list[tuple[str, dict]] = []
    for m in trace["messages"]:
        for c in m.get("tool_calls") or []:
            raw = c["function"]["arguments"]
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            out.append((c["function"]["name"], args))
    return out
