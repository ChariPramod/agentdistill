"""Adapter lifecycle.

The promotion checks are the only thing standing between "the loss went down" and an adapter serving real
traffic. They must be able to fail, and a failure must be recorded rather than argued away.
"""

from __future__ import annotations

import pytest

from agentdistill.config import ProjectConfig
from agentdistill.registry.lifecycle import (
    MAX_SUCCESS_REGRESSION_PP,
    TRANSITIONS,
    IllegalTransition,
    adapter,
    compare_live,
    events,
    promotion_checks,
    transition,
)


@pytest.fixture
def cfg(project_config):
    project_config.eval.eval_set = "holdout"
    return project_config


def add_adapter(registry, aid: str, status: str = "candidate", version: int = 1, **over):
    registry.insert_adapter({
        "id": aid, "training_run_id": "tr1", "name": aid, "version": version, "base_model": "m",
        "path": f"/tmp/{aid}", "status": status, **over,
    })


def add_eval(registry, run_id: str, subject: str, success: float, schema: float = 1.0, n_tasks: int = 12):
    registry.start_eval_run(run_id, "es_holdout", subject, 3)
    registry.finish_eval_run(run_id, {"success": success, "schema_valid": schema, "n_tasks": n_tasks})


@pytest.fixture
def seeded(registry):
    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 10, "n_tokens": 100, "content_hash": "h", "path": "/tmp/d"})
    registry.insert_training_run({"id": "tr1", "dataset_id": "ds1", "base_model": "m", "method": "sft",
                                  "config": {}, "status": "succeeded", "started_at": "2026-09-10T00:00:00+00:00"})
    registry.insert_eval_set({"id": "es_holdout", "name": "holdout", "trace_ids": [], "grader": {}})
    return registry


# --------------------------------------------------------------------------------------------------------------
# legality
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("frm", "to"), sorted(TRANSITIONS))
def test_legal_transitions_are_allowed(seeded, cfg, frm, to):
    add_adapter(seeded, "a1", status=frm)
    add_eval(seeded, "ev1", "a1", 0.8)
    # Forced, because legality is what is under test here, not the evidence.
    result = transition(seeded, "a1", to, cfg, force=True)
    assert result.ok
    assert adapter(seeded, "a1")["status"] == to


@pytest.mark.parametrize(("frm", "to"), [("retired", "prod"), ("prod", "canary"), ("canary", "candidate")])
def test_illegal_transitions_are_refused(seeded, cfg, frm, to):
    add_adapter(seeded, "a1", status=frm)
    with pytest.raises(IllegalTransition, match="not a legal transition"):
        transition(seeded, "a1", to, cfg, force=True)


def test_unknown_adapter(seeded, cfg):
    with pytest.raises(LookupError, match="no adapter"):
        transition(seeded, "nope", "canary", cfg)


# --------------------------------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------------------------------


def test_an_adapter_with_no_eval_cannot_be_promoted(seeded, cfg):
    add_adapter(seeded, "a1")
    result = transition(seeded, "a1", "canary", cfg)
    assert not result.ok
    assert "has_eval" in result.failed
    assert adapter(seeded, "a1")["status"] == "candidate", "a refused promotion must not move the adapter"


def test_schema_floor_blocks_promotion(seeded, cfg):
    add_adapter(seeded, "a1")
    add_eval(seeded, "ev1", "a1", 0.8, schema=0.80)
    checks = promotion_checks(seeded, "a1", "canary", cfg)
    assert not checks["schema_valid"].ok
    assert checks["schema_valid"].value == 0.80


def test_missing_calibration_blocks_canary(seeded, cfg):
    """A cascade with no gate escalates everything, which is not what a canary is for."""
    add_adapter(seeded, "a1")
    add_eval(seeded, "ev1", "a1", 0.8)
    checks = promotion_checks(seeded, "a1", "canary", cfg)
    assert not checks["calibrated"].ok
    assert "escalate everything" in checks["calibrated"].detail


def test_no_incumbent_means_nothing_to_compare_against(seeded, cfg):
    add_adapter(seeded, "a1")
    add_eval(seeded, "ev1", "a1", 0.8)
    checks = promotion_checks(seeded, "a1", "canary", cfg)
    assert checks["not_worse_than_prod"].ok


def test_an_incumbent_without_an_eval_blocks_the_comparison(seeded, cfg):
    """Refusing when the comparison is missing, not just when it is bad."""
    add_adapter(seeded, "old", status="prod")
    add_adapter(seeded, "a1", version=2)
    add_eval(seeded, "ev1", "a1", 0.8)
    checks = promotion_checks(seeded, "a1", "canary", cfg)
    assert not checks["not_worse_than_prod"].ok
    assert "nothing to compare" in checks["not_worse_than_prod"].detail


# --------------------------------------------------------------------------------------------------------------
# force, and what it records
# --------------------------------------------------------------------------------------------------------------


def test_force_promotes_and_records_that_it_was_forced(seeded, cfg):
    add_adapter(seeded, "a1")
    result = transition(seeded, "a1", "canary", cfg, force=True, actor="alice")
    assert result.ok and result.forced
    assert adapter(seeded, "a1")["status"] == "canary"

    history = events(seeded, "a1")
    assert len(history) == 1
    assert history[0]["checks"]["forced"] is True
    assert history[0]["actor"] == "alice"
    assert "has_eval" in [k for k, v in history[0]["checks"]["checks"].items() if not v["ok"]]


def test_a_clean_promotion_is_not_marked_forced(seeded, cfg):
    add_adapter(seeded, "a1")
    add_eval(seeded, "ev1", "a1", 0.8)
    result = transition(seeded, "a1", "retired", cfg)
    assert result.ok and not result.forced
    assert events(seeded, "a1")[0]["checks"]["forced"] is False


# --------------------------------------------------------------------------------------------------------------
# prod is singular
# --------------------------------------------------------------------------------------------------------------


def test_promoting_to_prod_retires_the_incumbent(seeded, cfg):
    add_adapter(seeded, "old", status="prod")
    add_adapter(seeded, "new", version=2)
    add_eval(seeded, "ev_old", "old", 0.7)
    add_eval(seeded, "ev_new", "new", 0.8)
    transition(seeded, "new", "prod", cfg, force=True)
    assert adapter(seeded, "old")["status"] == "retired"
    assert adapter(seeded, "new")["status"] == "prod"


def test_the_retirement_records_why(seeded, cfg):
    add_adapter(seeded, "old", status="prod")
    add_adapter(seeded, "new", version=2)
    transition(seeded, "new", "prod", cfg, force=True)
    assert "superseded by new" in events(seeded, "old")[0]["checks"]["reason"]


def test_exactly_one_adapter_is_prod_afterwards(seeded, cfg):
    from agentdistill.registry.select import prod_adapter

    add_adapter(seeded, "old", status="prod")
    add_adapter(seeded, "new", version=2)
    transition(seeded, "new", "prod", cfg, force=True)
    assert prod_adapter(seeded)["id"] == "new"


# --------------------------------------------------------------------------------------------------------------
# live comparison
# --------------------------------------------------------------------------------------------------------------


def _log_request(registry, adapter_id: str, cluster: int, outcome: bool, i: int):
    from sqlalchemy import text

    from agentdistill.registry.base import utcnow

    with registry.engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO requests (id, received_at, cluster_id, arm, adapter_id, escalated, outcome)
                    VALUES (:id, :t, :c, 'student', :a, 0, :o)"""),
            {"id": f"rq_{adapter_id}_{cluster}_{i}", "t": utcnow(), "c": cluster, "a": adapter_id,
             "o": outcome},
        )


def test_live_comparison_needs_shared_clusters(seeded):
    add_adapter(seeded, "prod", status="prod")
    add_adapter(seeded, "canary", status="canary", version=2)
    for i in range(10):
        _log_request(seeded, "canary", 1, True, i)
    assert compare_live(seeded, "prod", "canary") is None, "nothing to pair against"


def test_live_comparison_detects_a_better_canary(seeded):
    add_adapter(seeded, "prod", status="prod")
    add_adapter(seeded, "canary", status="canary", version=2)
    for cluster in range(4):
        for i in range(20):
            _log_request(seeded, "canary", cluster, i < 18, i)
            _log_request(seeded, "prod", cluster, i < 12, 100 + i)
    live = compare_live(seeded, "prod", "canary")
    assert live is not None
    assert live["delta"] > 0
    assert live["n_clusters"] == 4
    assert live["canary_requests"] == 80


def test_live_comparison_without_a_prod_adapter(seeded):
    add_adapter(seeded, "canary", status="canary")
    assert compare_live(seeded, None, "canary") is None


def test_promoting_a_canary_to_prod_requires_live_evidence(seeded, cfg):
    add_adapter(seeded, "old", status="prod")
    add_adapter(seeded, "canary", status="canary", version=2)
    add_eval(seeded, "ev_old", "old", 0.7)
    add_eval(seeded, "ev_can", "canary", 0.8)
    checks = promotion_checks(seeded, "canary", "prod", cfg)
    assert "live_not_worse" in checks
    assert not checks["live_not_worse"].ok, "no graded traffic yet"


def test_regression_bound_is_stated_in_the_check(seeded, cfg):
    add_adapter(seeded, "a1")
    add_eval(seeded, "ev1", "a1", 0.8)
    checks = promotion_checks(seeded, "a1", "canary", cfg)
    assert str(MAX_SUCCESS_REGRESSION_PP) in checks["not_worse_than_prod"].detail or \
        checks["not_worse_than_prod"].detail == "no incumbent to compare against"


def test_config_is_a_real_project_config(cfg):
    assert isinstance(cfg, ProjectConfig)
