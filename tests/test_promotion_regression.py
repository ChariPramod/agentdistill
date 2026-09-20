"""Below the power floor, every promotion check fails closed.

Three places decide whether an adapter moves forward on a paired comparison: the lifecycle's promotion checks,
retrain's `not_worse_than_prod` gate, and the on-policy loop's `decide`. Each was taught about
`insufficient_power` separately, which is exactly how one of them could forget. This holds all three to it with
one real comparison built from two real eval runs -- and the candidate looks far better than prod in the observed
rates, so a check that read the rates instead of the marker would promote an unmeasured adapter.
"""

from __future__ import annotations

import pytest

from agentdistill.eval.runner import RunSpec, compare, label_grader, run_eval
from agentdistill.eval.stats import MIN_REPEATS, MIN_TASKS
from agentdistill.registry.lifecycle import promotion_checks, transition
from agentdistill.retrain import gate_not_worse_than_prod
from agentdistill.train.onpolicy import RoundCfg, decide
from tests.test_eval_runner import PerTaskRecorded, make_eval_trace


class Unhelpful:
    """Prod, on this data: answers straight away with the wrong text, so it fails every task."""

    def next_turn(self, messages, tools):
        return {"role": "assistant", "content": "I cannot help with that.", "tool_calls": None}


@pytest.fixture
def cfg(project_config):
    project_config.eval.eval_set = "holdout"
    return project_config


def _setup(registry, n_tasks: int, n_repeats: int) -> tuple[str, str]:
    """Candidate and prod adapters, each with a real eval run on the same small set. Returns (cand, prod) runs."""
    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 10, "n_tokens": 100, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": "2026-09-18T00:00:00+00:00"})
    for aid, status in (("cand", "candidate"), ("prod", "prod")):
        registry.insert_adapter({"id": aid, "training_run_id": "tr1", "name": aid, "version": 1,
                                 "base_model": "m", "path": f"/tmp/{aid}", "status": status})

    traces = [make_eval_trace(i) for i in range(n_tasks)]
    registry.insert_traces(traces)
    registry.insert_eval_set({"id": "es_holdout", "name": "holdout", "trace_ids": [t["id"] for t in traces],
                              "grader": {"type": "label"}})
    by_task = {t["id"]: t for t in traces}
    es = registry.get_eval_set("holdout")
    cand = run_eval(registry, es, by_task, PerTaskRecorded(by_task), label_grader,
                    RunSpec(subject="cand", eval_set="holdout", n_per_task=n_repeats))
    prod = run_eval(registry, es, by_task, Unhelpful(), label_grader,
                    RunSpec(subject="prod", eval_set="holdout", n_per_task=n_repeats))
    return cand, prod


def _metrics(registry, run_id: str) -> dict:
    return registry.get_eval_run(run_id)["metrics"]


def _assert_every_check_fails_closed(registry, cfg, cmp: dict, cand: str, prod: str) -> None:
    ok, detail = gate_not_worse_than_prod({"comparison": cmp})
    assert not ok, "retrain's gate promoted on an underpowered comparison"
    assert detail == cmp["insufficient_power"]["reason"]

    decision, reason = decide(cmp, _metrics(registry, prod), _metrics(registry, cand), RoundCfg())
    assert decision == "discard", "the on-policy loop kept a round on an underpowered comparison"
    assert reason == cmp["insufficient_power"]["reason"]

    checks = promotion_checks(registry, "cand", "prod", cfg)
    assert checks["has_eval"].ok and checks["schema_valid"].ok, "the other checks pass, so only power can stop it"
    assert not checks["not_worse_than_prod"].ok, "the lifecycle promoted on an underpowered comparison"
    assert checks["not_worse_than_prod"].detail == cmp["insufficient_power"]["reason"]


def test_insufficient_power_never_promotes(registry, cfg, monkeypatch):
    cand, prod = _setup(registry, n_tasks=6, n_repeats=2)
    cmp = compare(registry, cand, prod)
    assert cmp["insufficient_power"]
    assert cmp["observed"] == {"rate_a": 1.0, "rate_b": 0.0}, "the candidate must look better, or this proves little"

    # promotion_checks computes its own comparison; hand it this exact dict so all three decide on one object.
    import agentdistill.eval.runner as runner

    seen: list[tuple[str, str]] = []

    def same_dict(registry_, a, b, alpha=0.05):
        seen.append((a, b))
        return cmp

    monkeypatch.setattr(runner, "compare", same_dict)
    _assert_every_check_fails_closed(registry, cfg, cmp, cand, prod)
    assert seen == [(cand, prod)], "promotion_checks must compare the candidate's run against prod's"

    # And the transition it guards refuses, leaving the adapter where it was.
    result = transition(registry, "cand", "prod", cfg)
    assert not result.ok and "not_worse_than_prod" in result.failed


@pytest.mark.parametrize(("n_tasks", "n_repeats"), [
    (3, 1), (5, 3), (MIN_TASKS - 1, MIN_REPEATS), (MIN_TASKS, MIN_REPEATS - 1), (MIN_TASKS + 4, 1), (10, 2),
])
def test_insufficient_power_never_promotes_across_the_grid(registry, cfg, n_tasks, n_repeats):
    """Every grid point is below the floor on at least one axis; the real `compare` runs inside the lifecycle."""
    assert n_tasks < MIN_TASKS or n_repeats < MIN_REPEATS
    cand, prod = _setup(registry, n_tasks=n_tasks, n_repeats=n_repeats)
    cmp = compare(registry, cand, prod)
    assert cmp["insufficient_power"]
    assert "success" not in cmp
    _assert_every_check_fails_closed(registry, cfg, cmp, cand, prod)


def test_the_floor_itself_is_where_statistics_start(registry, cfg):
    """The control: at the floor the same setup does produce statistics, so the refusals above are the floor's
    doing and not something else in the setup."""
    cand, prod = _setup(registry, n_tasks=MIN_TASKS, n_repeats=MIN_REPEATS)
    cmp = compare(registry, cand, prod)
    assert "insufficient_power" not in cmp
    assert cmp["success"]["delta"] == pytest.approx(1.0)
