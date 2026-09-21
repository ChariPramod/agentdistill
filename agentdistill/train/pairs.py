"""On-policy preference pairs, capped per task and sampled rather than enumerated.

The rehearsal built 532 pairs from a tiny corpus. That is a cross product: eight rollouts per task, every
failure paired with the teacher, and a "cap" that capped nothing because teacher pairs were filed as rollout
pairs. A pair set that size from that little data is a handful of decisions repeated hundreds of times, and DPO
on it overfits the handful.

The rules here:

- At most `cap_per_task` pairs per task, whatever the number of rollouts.
- Rollout pairs (student success versus student failure) are *sampled*: `cap_per_task * 3` draws, not the
  success-by-failure product.
- Teacher pairs only for tasks where the student never succeeded, since those are the tasks with no on-policy
  success to prefer. They are capped overall at `teacher_pair_ratio` times the rollout pairs, so the set does not
  turn into imitation learning wearing a DPO loss.
- Statistics go on the round row, including the per-task histogram, so the cap is checkable after the fact.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any

from agentdistill.data.pairs import first_divergent_pair, turn_key
from agentdistill.train.dpo_data import diff_kind, pair_is_valid

#: Above this share of prose-only pairs, the round row carries a warning: the set teaches phrasing, not decisions.
MAX_TEXT_SHARE = 0.7


def pair_hash(side: list[dict]) -> str:
    return hashlib.sha256(json.dumps([turn_key(m) for m in side], sort_keys=True).encode()).hexdigest()


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for p in pairs:
        key = (pair_hash(p["chosen"]), pair_hash(p["rejected"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _candidate(p: dict | None, kind: str, task_id: str) -> dict | None:
    if not p or not pair_is_valid(p)[0]:
        return None
    return {**p, "pair_kind": kind, "task_id": task_id, "diff_kind": diff_kind(p)}


def build_pairs(rollouts_by_task: dict[str, list[dict]], teacher_by_task: dict[str, dict],
                cap_per_task: int = 3, teacher_pair_ratio: float = 1.0, seed: int = 0) -> tuple[list[dict], dict]:
    """Capped, sampled, deduplicated pairs, plus the statistics that prove the cap held."""
    rng = random.Random(seed)
    rollout_pairs: list[dict] = []
    teacher_pairs: list[dict] = []
    for task_id, rolls in sorted(rollouts_by_task.items()):
        good = [r for r in rolls if r["success"]]
        bad = [r for r in rolls if not r["success"]]
        cands: list[dict] = []
        if good and bad:
            # Sample pairs rather than enumerating the cross product. Dedupe happens as candidates arrive, so the
            # cap counts distinct pairs, not repeated draws of the same one.
            for _ in range(min(cap_per_task * 3, len(good) * len(bad))):
                c = _candidate(first_divergent_pair(rng.choice(good), rng.choice(bad)), "rollout", task_id)
                if c:
                    cands = dedupe_pairs([*cands, c])
                if len(cands) >= cap_per_task:
                    break
        elif bad and task_id in teacher_by_task:
            for r in rng.sample(bad, min(cap_per_task, len(bad))):
                c = _candidate(first_divergent_pair(teacher_by_task[task_id], r), "teacher", task_id)
                if c:
                    cands.append(c)
        cands = dedupe_pairs(cands)[:cap_per_task]
        for c in cands:
            (rollout_pairs if c["pair_kind"] == "rollout" else teacher_pairs).append(c)

    max_teacher = int(len(rollout_pairs) * teacher_pair_ratio) if rollout_pairs else cap_per_task * 5
    if len(teacher_pairs) > max_teacher:
        teacher_pairs = rng.sample(teacher_pairs, max_teacher)
    pairs = rollout_pairs + teacher_pairs

    # Recount after the teacher cap, so the histogram describes the pairs that are actually trained on.
    kept = Counter(p["task_id"] for p in pairs)
    diff = dict(Counter(p["diff_kind"] for p in pairs))
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "n_pairs": len(pairs), "n_rollout": len(rollout_pairs), "n_teacher": len(teacher_pairs),
        "cap_per_task": cap_per_task, "teacher_pair_ratio": teacher_pair_ratio,
        "tasks_with_pairs": sum(1 for v in kept.values() if v), "max_per_task": max(kept.values(), default=0),
        "diff_kind": diff, "per_task_histogram": {str(k): v for k, v in sorted(Counter(kept.values()).items())},
        "warnings": warnings,
    }
    if pairs and diff.get("text", 0) / len(pairs) > MAX_TEXT_SHARE:
        warnings.append(
            f"{diff['text'] / len(pairs):.0%} of pairs differ only in prose (above {MAX_TEXT_SHARE:.0%}); this set "
            f"teaches phrasing rather than decisions"
        )
    return pairs, stats
