"""DPO pair construction.

A preference pair is only informative if both sides answer the *same* question. These builders pair at the first
assistant turn where two trajectories diverge, and only when every message before that turn is identical --
otherwise the "preference" is confounded by a different prefix, and DPO learns nothing about the decision.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from agentdistill.ingest.normalize import canonical_arguments


def turn_key(m: dict) -> str:
    """Identifies an assistant turn by what it does, not by call ids or key order."""
    calls = [
        (c["function"]["name"], canonical_arguments(c["function"]["arguments"])) for c in m.get("tool_calls") or []
    ]
    payload = json.dumps([m.get("content") or "", calls], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def first_divergent_pair(success: dict, failure: dict) -> dict | None:
    """Return {prompt, chosen, rejected, tools, task_id} at the first assistant turn where the trajectories differ.

    Returns None when they never diverge, when they diverge first on a non-assistant message (a different tool
    result or user turn means the environments differ, not the decisions), or when one is a strict prefix of the
    other.
    """
    s, f = success["messages"], failure["messages"]
    for i in range(min(len(s), len(f))):
        if s[i]["role"] != f[i]["role"]:
            return None
        if s[i]["role"] != "assistant":
            if turn_key(s[i]) != turn_key(f[i]):
                return None  # tool results or user turns differ before any assistant divergence
            continue
        if turn_key(s[i]) == turn_key(f[i]):
            continue
        return {
            "prompt": s[:i],
            "chosen": [s[i]],
            "rejected": [f[i]],
            "tools": success.get("tools") or [],
            "task_id": success.get("task_id"),
            "chosen_trace_id": success.get("id"),
            "rejected_trace_id": failure.get("id"),
        }
    return None


def pairs_from_traces(traces: list[dict], max_pairs_per_task: int = 4) -> tuple[list[dict], dict[str, int]]:
    """Build offline pairs from teacher traces: for each task with both a success and a failure, pair them.

    Returns (pairs, stats). Tasks are capped so that one heavily-repeated task cannot dominate the pair set.
    """
    by_task: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"success": [], "failure": []})
    for t in traces:
        task = t.get("task_id")
        if task is None or t.get("success") is None:
            continue
        by_task[task]["success" if t["success"] else "failure"].append(t)

    pairs: list[dict] = []
    stats = {
        "tasks_seen": len(by_task),
        "tasks_with_both": 0,
        "pairs_built": 0,
        "no_divergence": 0,
        "divergent_prefix": 0,
        "capped": 0,
    }
    for _task, sides in sorted(by_task.items()):
        if not sides["success"] or not sides["failure"]:
            continue
        stats["tasks_with_both"] += 1
        built = 0
        for good in sides["success"]:
            for bad in sides["failure"]:
                if built >= max_pairs_per_task:
                    stats["capped"] += 1
                    break
                pair = first_divergent_pair(good, bad)
                if pair is None:
                    # Distinguishing the two failure modes tells a user whether to collect more rollouts or to
                    # fix the harness determinism.
                    if turn_key_sequence(good) == turn_key_sequence(bad):
                        stats["no_divergence"] += 1
                    else:
                        stats["divergent_prefix"] += 1
                    continue
                pairs.append(pair)
                built += 1
            if built >= max_pairs_per_task:
                break
        stats["pairs_built"] += built
    return pairs, stats


def turn_key_sequence(trace: dict) -> list[str]:
    return [turn_key(m) for m in trace["messages"] if m["role"] == "assistant"]


def pairs_against_teacher(student_rollouts: list[dict], teacher_traces: list[dict]) -> list[dict]:
    """Pair a failed student rollout against the teacher's successful trajectory on the same task.

    This is the on-policy half of section 6.2 step 4: the student's own mistake is the rejected side, so the
    gradient acts on a prefix the student actually produces at inference.
    """
    teacher_by_task: dict[str, dict] = {}
    for t in teacher_traces:
        if t.get("task_id") and t.get("success"):
            teacher_by_task.setdefault(t["task_id"], t)
    out: list[dict] = []
    for roll in student_rollouts:
        if roll.get("success"):
            continue
        task_id = roll.get("task_id")
        if task_id is None:
            continue
        teacher = teacher_by_task.get(task_id)
        if teacher is None:
            continue
        pair = first_divergent_pair(teacher, roll)
        if pair is not None:
            pair["source"] = "on_policy_vs_teacher"
            # Without this, `balance_kinds` files a teacher pair as a rollout pair and the teacher cap never bites.
            pair["pair_kind"] = "teacher"
            out.append(pair)
    return out


def write_pairs_jsonl(pairs: list[dict], path: Any) -> Any:
    """TRL reads conversational DPO records as JSONL with prompt/chosen/rejected."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return p


def write_pairs_dataset(
    pairs: list[dict], cfg: Any, registry: Any, name: str, version: int, tag: str | None = None
) -> str:
    """Write a DPO pair set as a registered, content-hashed dataset.

    Registered rather than left on disk, because an on-policy round records the dataset it trained on and the
    `rounds` table has a foreign key to it. A fabricated id means a round that cannot be written down at all,
    which is the worst possible outcome for a stage whose whole job is to be auditable.

    Content-hashed for the same reason SFT datasets are: two rounds that produced identical pairs should be
    recognisably the same input, and a report that cites a pair set should cite something immutable.
    """
    from pathlib import Path

    from agentdistill.registry.base import utcnow

    payload = "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True) for p in pairs)
    content_hash = hashlib.sha256(payload.encode()).hexdigest()
    dataset_id = f"ds_{content_hash[:16]}"

    existing = registry.get_dataset(dataset_id)
    if existing is not None:
        return str(existing["id"])

    out_dir = Path(cfg.artifacts_dir) / "datasets" / f"{name}-v{version}"
    path = write_pairs_jsonl(pairs, out_dir / "pairs.jsonl")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"kind": "dpo", "name": name, "version": version, "n_pairs": len(pairs),
             "content_hash": content_hash, "tag": tag},
            indent=2, sort_keys=True,
        )
        + "\n"
    )

    registry.insert_dataset({
        "id": dataset_id,
        "name": name,
        "version": version,
        "kind": "dpo",
        "filter_config": {"source": "on-policy rollouts", "tag": tag},
        "n_samples": len(pairs),
        # Pairs are not tokenized here, so there is no honest token count to report.
        "n_tokens": 0,
        "content_hash": content_hash,
        "path": str(path.parent),
        "report_path": None,
        "created_at": utcnow(),
    })
    return dataset_id
