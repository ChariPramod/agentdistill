"""Router persistence and the warm start.

Two rules here, and both are about not learning the wrong thing:

- A warm start runs only on an empty table. Re-running it would overwrite what production taught the router with
  an offline eval, which is the less reliable of the two.
- A promotion resets the posteriors. The old adapter's record is not evidence about the new one.
"""

from __future__ import annotations

import pytest

from agentdistill.registry.base import utcnow
from agentdistill.report.registry_views import router_state
from agentdistill.router.store import RouterStore, flush_router, load_router, reset_router
from agentdistill.router.thompson import ArmState, ThompsonRouter


def _adapter(registry, name: str, status: str) -> str:
    aid = f"ad_{name}"
    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 10, "n_tokens": 100, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "base", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": utcnow()})
    registry.insert_adapter(
        {"id": aid, "training_run_id": "tr1", "name": name, "version": 1,
         "base_model": "base", "path": f"/tmp/{name}", "status": status}
    )
    return aid


def _eval_run(registry, subject: str, per_cluster: dict, eval_set: str = "holdout") -> str:
    registry.insert_eval_set({"id": f"es_{eval_set}", "name": eval_set, "trace_ids": [], "grader": {}})
    run_id = f"run_{subject}"
    registry.start_eval_run(run_id, f"es_{eval_set}", subject, n_per_task=1)
    registry.finish_eval_run(run_id, {"success": 0.8}, per_cluster=per_cluster)
    return run_id


@pytest.fixture
def cfg(project_config):
    project_config.eval.eval_set = "holdout"
    return project_config


def test_empty_table_warm_starts_from_eval_counts_and_flushes(registry, cfg):
    prod = _adapter(registry, "student-v1", "prod")
    _eval_run(registry, prod, {"0": {"n_tasks": 10, "success": 0.8}, "1": {"n_tasks": 10, "success": 0.4}})

    router = load_router(registry, cfg)

    # 8 of 10 in cluster 0, on top of the Beta(1,1) prior.
    assert router.arm(0, "student").alpha == pytest.approx(9.0)
    assert router.arm(0, "student").beta == pytest.approx(3.0)
    assert router.arm(1, "student").mean == pytest.approx(5 / 12)
    # And it persisted, so a restart does not warm-start a second time.
    assert {(r["cluster_id"], r["arm"]) for r in router_state(registry)} == {(0, "student"), (1, "student")}


def test_a_non_empty_table_loads_without_warm_starting(registry, cfg):
    prod = _adapter(registry, "student-v1", "prod")
    _eval_run(registry, prod, {"0": {"n_tasks": 10, "success": 0.8}})

    live = ThompsonRouter(state={(0, "student"): ArmState(50.0, 2.0)})
    flush_router(registry, live)

    router = load_router(registry, cfg)

    # What production learned survives; the eval counts do not overwrite it.
    assert router.arm(0, "student").alpha == pytest.approx(50.0)
    assert router.arm(0, "student").beta == pytest.approx(2.0)


def test_reset_then_load_restarts_from_the_new_adapters_eval(registry, cfg):
    prod = _adapter(registry, "student-v2", "prod")
    _eval_run(registry, prod, {"0": {"n_tasks": 10, "success": 0.9}})
    flush_router(registry, ThompsonRouter(state={(0, "student"): ArmState(2.0, 90.0)}))

    reset_router(registry)
    assert router_state(registry) == []

    router = load_router(registry, cfg)
    # The predecessor's 90 failures are gone; the new adapter starts from its own eval.
    assert router.arm(0, "student").mean == pytest.approx(10 / 12)


def test_no_prod_adapter_and_no_evals_leaves_the_router_blind_but_working(registry, cfg):
    router = load_router(registry, cfg)
    assert router.snapshot() == []
    assert router.choose(0) in ("student", "teacher")


def test_flush_is_idempotent_and_overwrites_rather_than_duplicating(registry, cfg):
    router = ThompsonRouter(state={(0, "student"): ArmState(3.0, 1.0)})
    flush_router(registry, router)
    router.update(0, "student", True)
    flush_router(registry, router)

    rows = router_state(registry)
    assert len(rows) == 1
    assert rows[0]["alpha"] == pytest.approx(3.0 * 0.995 + 1.0)


def test_store_handle_flushes_and_resets(registry, cfg):
    store = RouterStore(registry)
    router = ThompsonRouter(state={(1, "teacher"): ArmState(5.0, 5.0)})
    store.flush(router)
    assert len(router_state(registry)) == 1
    store.reset()
    assert router_state(registry) == []


def test_updates_persist_across_a_reload(registry, cfg):
    router = load_router(registry, cfg)
    for _ in range(20):
        router.update(3, "student", True)
    flush_router(registry, router)

    reloaded = load_router(registry, cfg)
    assert reloaded.arm(3, "student").mean == pytest.approx(router.arm(3, "student").mean)
    assert reloaded.arm(3, "student").mean > 0.9
