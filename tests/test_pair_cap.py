"""The pair set is capped per task and sampled, and its statistics prove it.

The rehearsal's 532 pairs from a tiny corpus were a cross product. The invariants here are the ones the plan
names: `n_pairs <= cap * n_tasks * (1 + teacher_ratio)` and `max_per_task <= cap`, for any rollout mix.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agentdistill.data.pairs import pairs_against_teacher
from agentdistill.train.dpo_data import balance_kinds
from agentdistill.train.pairs import build_pairs
from tests.conftest import make_call

TOOLS = ["refund_order", "cancel_order", "escalate", "lookup_order", "update_address", "close_ticket"]


def _traj(task: str, idx: int, success: bool, action: int) -> dict:
    """Shared first turn and tool result, then a decision that varies with `action`."""
    tool = TOOLS[action % len(TOOLS)]
    return {
        "id": f"{task}-{idx}", "task_id": task, "success": success, "tools": [],
        "messages": [
            {"role": "user", "content": f"help with {task}"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [make_call("c1", "lookup", {"t": task})]},
            {"role": "tool", "tool_call_id": "c1", "content": "found"},
            {"role": "assistant", "content": "Acting.",
             "tool_calls": [make_call("c2", tool, {"t": task, "variant": action})]},
        ],
    }


def _corpus(n_tasks: int, k: int, success_every: int) -> tuple[dict, dict]:
    rollouts = {}
    for t in range(n_tasks):
        task = f"task{t}"
        rollouts[task] = [_traj(task, i, success=(success_every > 0 and i % success_every == 0), action=i)
                          for i in range(k)]
    teacher = {f"task{t}": _traj(f"task{t}", 99, True, 0) for t in range(n_tasks)}
    return rollouts, teacher


@settings(max_examples=60, deadline=None)
@given(n_tasks=st.integers(1, 12), k=st.integers(1, 10), success_every=st.integers(0, 4),
       cap=st.integers(1, 5), ratio=st.sampled_from([0.0, 0.5, 1.0, 2.0]))
def test_the_cap_holds_for_any_rollout_mix(n_tasks, k, success_every, cap, ratio):
    rollouts, teacher = _corpus(n_tasks, k, success_every)
    pairs, stats = build_pairs(rollouts, teacher, cap_per_task=cap, teacher_pair_ratio=ratio)
    assert stats["n_pairs"] == len(pairs)
    assert stats["max_per_task"] <= cap
    assert len(pairs) <= cap * n_tasks * (1 + ratio) + (cap * 5 if stats["n_rollout"] == 0 else 0)
    assert stats["n_rollout"] + stats["n_teacher"] == len(pairs)
    if stats["n_rollout"]:
        assert stats["n_teacher"] <= int(stats["n_rollout"] * ratio)


def test_a_tiny_corpus_lands_in_the_low_tens_not_the_hundreds():
    """Ten tasks at k=8, the tiny rehearsal's shape. The uncapped builders produced hundreds here."""
    rollouts, teacher = _corpus(n_tasks=10, k=8, success_every=3)
    pairs, stats = build_pairs(rollouts, teacher)
    assert 10 <= len(pairs) <= 30
    assert stats["max_per_task"] == 3, "the cap must actually bind when the data allows it"
    assert stats["per_task_histogram"] == {"3": 10}


def test_the_old_builders_on_the_same_corpus_were_a_cross_product():
    """Kept as the regression this phase fixes: teacher pairs uncapped, and mislabelled as rollout pairs."""
    rollouts, teacher = _corpus(n_tasks=10, k=8, success_every=0)
    flat = [r for rs in rollouts.values() for r in rs]
    old = pairs_against_teacher(flat, list(teacher.values()))
    assert len(old) == 70, "every failed rollout paired with the teacher"
    assert all(p["pair_kind"] == "teacher" for p in old), "labelled, so balance_kinds can cap them"
    kept, _ = balance_kinds(old, max_teacher_ratio=1.0)
    new, stats = build_pairs(rollouts, teacher)
    assert len(new) == 15 and stats["n_teacher"] == 15, "teacher-only sets are capped at cap_per_task * 5"
    assert len(new) < len(kept)


def test_teacher_pairs_only_where_the_student_never_succeeded():
    rollouts, teacher = _corpus(n_tasks=4, k=6, success_every=2)
    pairs, stats = build_pairs(rollouts, teacher)
    assert stats["n_teacher"] == 0, "every task had an on-policy success to prefer"
    assert all(p["pair_kind"] == "rollout" for p in pairs)


def test_pairs_are_deterministic_for_a_seed():
    rollouts, teacher = _corpus(n_tasks=6, k=8, success_every=3)
    a, _ = build_pairs(rollouts, teacher, seed=7)
    b, _ = build_pairs(rollouts, teacher, seed=7)
    assert [(p["chosen_trace_id"], p["rejected_trace_id"]) for p in a] == \
        [(p["chosen_trace_id"], p["rejected_trace_id"]) for p in b]


def test_no_duplicate_pairs():
    rollouts, teacher = _corpus(n_tasks=3, k=8, success_every=3)
    pairs, _ = build_pairs(rollouts, teacher, cap_per_task=5)
    keys = [(p["chosen_trace_id"], p["rejected_trace_id"]) for p in pairs]
    assert len(keys) == len(set(keys))


def test_a_prose_dominated_set_warns():
    rollouts = {}
    for t in range(5):
        task = f"t{t}"
        rolls = []
        for i in range(4):
            rolls.append({"id": f"{task}-{i}", "task_id": task, "success": i == 0, "tools": [], "messages": [
                {"role": "user", "content": "q"}, {"role": "assistant", "content": f"answer number {i}"}]})
        rollouts[task] = rolls
    _, stats = build_pairs(rollouts, {})
    assert stats["diff_kind"] == {"text": stats["n_pairs"]}
    assert any("differ only in prose" in w for w in stats["warnings"])


def test_pair_stats_land_on_the_round_row(registry):
    from agentdistill.registry.select import rounds_for_tag

    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": "2026-09-18T00:00:00+00:00"})
    registry.insert_adapter({"id": "ad1", "training_run_id": "tr1", "name": "s", "version": 1, "base_model": "m",
                             "path": "/tmp/a"})
    _, stats = build_pairs(*_corpus(4, 8, 3))
    registry.record_round({"round_idx": 0, "start_adapter": "ad1", "tag": "t", "pair_kinds": stats,
                           "decision": "discard", "reason": "x"})
    row = rounds_for_tag(registry, "t")[0]
    assert row["pair_stats"]["max_per_task"] == stats["max_per_task"]
    assert row["pair_stats"]["per_task_histogram"] == stats["per_task_histogram"]
