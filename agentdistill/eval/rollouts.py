"""Bridge to Milestone 4: on-policy rollout collection.

With the harness in place, on-policy training is three functions. This module is the first: run the student on
*training* tasks, many times, grade each rollout, and hand back traces the existing dataset and pair builders
already understand.

One rule that is easy to get wrong: rollouts must use the **fuzzy** replay policy. An on-policy trajectory drifts
from the teacher's argument phrasing, and strict mode would stop most rollouts at the first divergence, leaving a
tiny and badly biased set. The price is that a fuzzily served result is not the result the student's call would
really have produced, so the fuzzy-hit share is reported on every collection and belongs in any report built on
these rollouts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentdistill.eval.harness import TaskOutcome, run_task
from agentdistill.eval.replay import ReplayStats, ReplayToolProvider


@dataclass
class RolloutSet:
    """Graded rollouts, plus the replay statistics needed to judge whether to trust them."""

    rollouts: list[dict] = field(default_factory=list)
    replay: dict = field(default_factory=dict)
    n_tasks: int = 0
    k: int = 0

    @property
    def success_rate(self) -> float:
        if not self.rollouts:
            return float("nan")
        return sum(bool(r["success"]) for r in self.rollouts) / len(self.rollouts)

    def successes(self) -> list[dict]:
        return [r for r in self.rollouts if r["success"]]

    def failures(self) -> list[dict]:
        return [r for r in self.rollouts if not r["success"]]

    def warnings(self) -> list[str]:
        out = []
        share = self.replay.get("fuzzy_share") or 0.0
        if share > 0.25:
            out.append(
                f"{share:.0%} of tool results were served by fuzzy replay. A fuzzily replayed success is not a "
                f"real success; lower the threshold and re-check with the predicate before training on these."
            )
        if self.rollouts and self.success_rate < 0.05:
            out.append(
                "almost no rollout succeeded, so there is nothing to do rejection sampling with. Check the "
                "student actually loaded, and read a rollout by hand."
            )
        if self.rollouts and self.success_rate > 0.95:
            out.append(
                "almost every rollout succeeded, so there are no failures to pair against. These tasks are too "
                "easy to improve the student on."
            )
        return out


def outcome_to_trace(
    outcome: TaskOutcome, source_trace: dict, adapter_id: str | None, round_idx: int
) -> dict:
    """Turn a rollout into a trace the dataset and pair builders already accept."""
    from agentdistill.ingest.normalize import normalize_trace

    raw = {
        "task_id": source_trace.get("task_id") or source_trace["id"],
        "task_input": source_trace.get("task_input"),
        "messages": outcome.messages,
        "tools": source_trace.get("tools") or [],
        "success": outcome.success,
        "grader": "rollout",
        "score": 1.0 if outcome.success else 0.0,
        "metadata": {
            **(source_trace.get("metadata") or {}),
            "rollout_of": source_trace["id"],
            "parent_adapter_id": adapter_id,
            "round": round_idx,
            "repeat_idx": outcome.repeat_idx,
            "diverged": outcome.diverged,
            "replay_stats": outcome.replay_stats,
            "grader_detail": outcome.grader_detail,
        },
    }
    trace = normalize_trace(raw, source="rollout")
    trace["cluster"] = source_trace.get("cluster")
    return trace


def collect_rollouts(
    traces: list[dict],
    client: Any,
    grader: Callable[[dict, TaskOutcome], tuple[bool, dict]],
    k: int = 8,
    policy: str = "fuzzy",
    fuzzy_threshold: float = 0.92,
    max_turns: int = 12,
    adapter_id: str | None = None,
    round_idx: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> RolloutSet:
    """Run the student `k` times on each task and grade every rollout."""
    out = RolloutSet(n_tasks=len(traces), k=k)
    replay = ReplayStats()
    total = len(traces) * k
    done = 0
    for trace in traces:
        for i in range(k):
            provider = ReplayToolProvider(trace, policy=policy, fuzzy_threshold=fuzzy_threshold)
            if hasattr(client, "reset"):
                client.reset()
            outcome = run_task(trace, client, provider, repeat_idx=i, max_turns=max_turns)
            success, detail = grader(trace, outcome)
            outcome.success = success
            outcome.grader_detail = str(detail.get("detail", ""))[:500]
            replay.add(outcome.replay_stats)
            out.rollouts.append(outcome_to_trace(outcome, trace, adapter_id, round_idx))
            done += 1
            if progress:
                progress(done, total)
    out.replay = replay.to_dict()
    return out


def build_rft(rollouts: RolloutSet, cap_per_task: int = 2) -> list[dict]:
    """Successful rollouts as SFT traces, capped per task.

    The cap matters: without it, easy tasks that succeed every time dominate the set, and the student is trained
    hardest on what it already does well.
    """
    by_task: dict[str, list[dict]] = {}
    for r in rollouts.successes():
        by_task.setdefault(r.get("task_id") or r["id"], []).append(r)
    out: list[dict] = []
    for task in sorted(by_task):
        out.extend(by_task[task][:cap_per_task])
    return out


def build_pairs(rollouts: RolloutSet, teacher_traces: list[dict], max_pairs_per_task: int = 4) -> list[dict]:
    """Preference pairs from rollouts: success versus failure on the same task, plus failure versus teacher."""
    from agentdistill.data.pairs import pairs_against_teacher, pairs_from_traces

    same_task, _ = pairs_from_traces(rollouts.rollouts, max_pairs_per_task=max_pairs_per_task)
    versus_teacher = pairs_against_teacher(rollouts.rollouts, teacher_traces)
    return same_task + versus_teacher
