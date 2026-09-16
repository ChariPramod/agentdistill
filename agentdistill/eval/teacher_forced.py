"""Teacher-forced next-action accuracy.

Given the teacher's own prefix at each assistant turn, does the student choose the same action? It needs no tool
mocking, runs per turn, and is the first real quality signal available -- loss is a proxy, this is not.

It is also **not** end-to-end success. A student can score well here and still fail on its own trajectories,
because at inference the prefix is its own, mistakes included. That gap is exposure bias, and closing it is
Milestone 4's job. Report both numbers; a high score here alone proves nothing about the agent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from agentdistill.canonical import args_hash


class TurnClient(Protocol):
    """Anything that can produce one assistant turn from a prefix."""

    def next_turn(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return an assistant message: {'content': str|None, 'tool_calls': [...]|None}."""
        ...


@dataclass
class TurnResult:
    task_id: str
    turn_idx: int
    teacher_kind: str  # 'tool' | 'text'
    student_kind: str
    name_match: bool  # same set of tool names, order-insensitive
    args_match: bool  # same canonical argument hashes for every call
    kind_match: bool  # both chose to call tools, or both chose to answer

    @property
    def full_match(self) -> bool:
        return self.kind_match and self.name_match and self.args_match


def turn_signature(message: dict) -> tuple[str, frozenset[str], frozenset[str]]:
    """(kind, tool names, canonical arg hashes).

    Names and hashes are sets: a model issuing the same parallel calls in a different order did the same thing.
    Arguments go through `canonical.args_hash`, so formatting differences are not scored as disagreement.
    """
    calls = message.get("tool_calls") or []
    names, hashes = set(), set()
    for c in calls:
        fn = c["function"]
        raw = fn["arguments"]
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            # Unparseable arguments are a real difference from the teacher's valid ones, and must not collide
            # with any well-formed call.
            args = {"__unparseable__": str(raw)}
        names.add(fn["name"])
        hashes.add(args_hash(fn["name"], args))
    return ("tool" if calls else "text", frozenset(names), frozenset(hashes))


def compare_turns(teacher: dict, student: dict, task_id: str, turn_idx: int) -> TurnResult:
    tk, tn, th = turn_signature(teacher)
    sk, sn, sh = turn_signature(student)
    return TurnResult(
        task_id=task_id,
        turn_idx=turn_idx,
        teacher_kind=tk,
        student_kind=sk,
        name_match=tn == sn,
        args_match=th == sh,
        kind_match=tk == sk,
    )


def _prefixes(trace: dict, max_turns_per_trace: int | None) -> list[tuple[int, dict]]:
    """Every assistant turn with the prefix that precedes it."""
    out: list[tuple[int, dict]] = []
    for i, m in enumerate(trace["messages"]):
        if m["role"] != "assistant":
            continue
        if max_turns_per_trace is not None and len(out) >= max_turns_per_trace:
            break
        out.append((i, m))
    return out


def teacher_forced(
    traces: list[dict], client: TurnClient, max_turns_per_trace: int | None = None
) -> list[TurnResult]:
    """Score every assistant turn one at a time. Simple and slow; see `teacher_forced_batched`."""
    results: list[TurnResult] = []
    for trace in traces:
        messages, tools = trace["messages"], trace.get("tools") or []
        task_id = trace.get("task_id") or trace["id"]
        for i, teacher_msg in _prefixes(trace, max_turns_per_trace):
            student = client.next_turn(messages[:i], tools)
            results.append(compare_turns(teacher_msg, student, task_id, i))
    return results


def teacher_forced_batched(
    traces: list[dict], client: Any, max_turns_per_trace: int | None = None
) -> list[TurnResult]:
    """Collect every (prefix, tools) pair first and generate in one batch.

    With vLLM and prefix caching, turns from the same task share nearly all their prompt, so a few hundred turns
    take well under a minute. Falls back to the per-turn path for clients without `next_turns_batch`.
    """
    batch = getattr(client, "next_turns_batch", None)
    if batch is None:
        return teacher_forced(traces, client, max_turns_per_trace)

    prompts: list[tuple[list[dict], list[dict]]] = []
    keys: list[tuple[str, int, dict]] = []
    for trace in traces:
        messages, tools = trace["messages"], trace.get("tools") or []
        task_id = trace.get("task_id") or trace["id"]
        for i, teacher_msg in _prefixes(trace, max_turns_per_trace):
            prompts.append((messages[:i], tools))
            keys.append((task_id, i, teacher_msg))
    if not prompts:
        return []
    students = batch(prompts)
    if len(students) != len(keys):
        raise ValueError(
            f"client returned {len(students)} turns for {len(keys)} prompts; a batched client must preserve order "
            f"and length"
        )
    return [compare_turns(t, s, task, i) for (task, i, t), s in zip(keys, students, strict=True)]


def _bootstrap_ci(
    per_task: np.ndarray, iters: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile CI over a task-level bootstrap.

    Resampling *tasks*, not turns, is what makes the interval honest: turns within a task are correlated, so a
    turn-level bootstrap would report an interval several times too narrow.
    """
    if len(per_task) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(per_task), size=(iters, len(per_task)))
    boots = per_task[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (float(lo), float(hi))


def summarize(results: list[TurnResult], iters: int = 2000, seed: int = 0) -> dict:
    """Aggregate with task-clustered bootstrap intervals on the rates that carry one."""
    if not results:
        return {
            "n_turns": 0, "n_tasks": 0,
            "kind_match": (float("nan"), (float("nan"), float("nan"))),
            "name_match_on_tool_turns": float("nan"),
            "args_match_on_tool_turns": float("nan"),
            "full_match": (float("nan"), (float("nan"), float("nan"))),
            "n_tool_turns": 0,
        }

    by_task: dict[str, list[TurnResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    tasks = sorted(by_task)
    rng = np.random.default_rng(seed)

    def rate(fn: Callable[[TurnResult], bool]) -> tuple[float, tuple[float, float]]:
        per_task = np.array([np.mean([fn(r) for r in by_task[t]]) for t in tasks], dtype=float)
        return float(per_task.mean()), _bootstrap_ci(per_task, iters, rng)

    tool_turns = [r for r in results if r.teacher_kind == "tool"]
    return {
        "n_turns": len(results),
        "n_tasks": len(tasks),
        "n_tool_turns": len(tool_turns),
        "kind_match": rate(lambda r: r.kind_match),
        # Conditional on the teacher having called a tool: "did it pick the right tool" is only a question when
        # there was a tool to pick. Averaging over text turns would inflate it.
        "name_match_on_tool_turns": (
            float(np.mean([r.name_match for r in tool_turns])) if tool_turns else float("nan")
        ),
        # Conditional again on the name matching: this separates "knows what to do" from "knows how to fill it in".
        "args_match_on_tool_turns": (
            float(np.mean([r.args_match for r in tool_turns if r.name_match]))
            if any(r.name_match for r in tool_turns)
            else float("nan")
        ),
        "full_match": rate(lambda r: r.full_match),
    }


def format_summary(s: dict) -> str:
    """One line for a training log."""
    if not s["n_turns"]:
        return "next_action: no turns scored"
    full, (lo, hi) = s["full_match"]
    return (
        f"next_action full={full:.3f} [{lo:.3f}, {hi:.3f}] "
        f"name={s['name_match_on_tool_turns']:.3f} args={s['args_match_on_tool_turns']:.3f} "
        f"({s['n_turns']} turns, {s['n_tasks']} tasks)"
    )
