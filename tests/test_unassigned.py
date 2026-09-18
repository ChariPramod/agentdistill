"""An unassigned cluster is a health state, not a routing decision.

The rehearsal gateway had no cluster model at all -- curation computed centroids and threw them away -- so every
request went to the router with no context. These tests pin the replacement: centroids persist, the gateway loads
them or says why not, unplaceable requests route on the pooled posterior without the floor and are counted, and
/healthz names what is missing on the first request anyone makes.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agentdistill.gateway import app as app_module
from agentdistill.gateway.health import HealthTracker
from agentdistill.router.clusters import (
    UNASSIGNED,
    ClusterAssigner,
    NoClusterModel,
    embedder_spec,
    load_cluster_assigner,
    save_cluster_model,
)
from agentdistill.router.thompson import ArmState, ThompsonRouter
from tests.test_gateway import TEACHER_NAME, broken_student, build_state

# --------------------------------------------------------------------------------------------------------------
# the assigner
# --------------------------------------------------------------------------------------------------------------


def test_no_cluster_model_assigns_everything_unassigned():
    m = NoClusterModel("no file")
    assert m.assign([{"role": "user", "content": "refund"}]) == UNASSIGNED
    assert m.describe() == {"state": "missing", "reason": "no file"}


def test_an_assigner_with_no_centroids_is_unassigned():
    assert ClusterAssigner(np.zeros((0, 4)), embedder=None).assign([{"role": "user", "content": "x"}]) == UNASSIGNED


def _seed_dataset(registry):
    registry.insert_dataset({"id": "ds1", "name": "d", "version": 1, "kind": "sft", "filter_config": {},
                             "n_samples": 1, "n_tokens": 1, "content_hash": "h", "path": "/tmp/d"})


def test_saved_centroids_round_trip_and_place_requests_like_curation(registry, project_config, tmp_path):
    from agentdistill.curate.stratify import HashEmbedder

    _seed_dataset(registry)
    emb = HashEmbedder(dim=project_config.curate.embeddings.dim)
    refunds = emb.embed(["refund my order please", "i want a refund for my order"]).mean(axis=0)
    shipping = emb.embed(["where is my package shipping", "track my shipping package"]).mean(axis=0)
    model_id = save_cluster_model(registry, tmp_path, "ds1", np.vstack([refunds, shipping]),
                                  embedder_spec(project_config.curate.embeddings, emb.name))

    assigner = load_cluster_assigner(registry)
    assert assigner.describe()["state"] == "loaded"
    assert assigner.describe()["id"] == model_id and assigner.describe()["k"] == 2
    ask = [{"role": "system", "content": "You are support."}, {"role": "user", "content": "refund my order"}]
    assert assigner.assign(ask) == 0
    assert assigner.assign([{"role": "user", "content": "track my package shipping"}]) == 1


def test_loading_with_no_row_names_the_fix(registry):
    m = load_cluster_assigner(registry)
    assert isinstance(m, NoClusterModel)
    assert "agentdistill curate" in m.reason


def test_loading_with_missing_centroids_says_where(registry, project_config, tmp_path):
    _seed_dataset(registry)
    save_cluster_model(registry, tmp_path, "ds1", np.ones((2, 4)),
                       embedder_spec(project_config.curate.embeddings, "hash-4"))
    for f in (tmp_path / "clusters").iterdir():
        f.unlink()
    m = load_cluster_assigner(registry)
    assert isinstance(m, NoClusterModel)
    assert "unreadable" in m.reason


# --------------------------------------------------------------------------------------------------------------
# pooled routing
# --------------------------------------------------------------------------------------------------------------


def test_the_pooled_posterior_sums_evidence_across_clusters():
    r = ThompsonRouter(state={(0, "student"): ArmState(5, 3), (1, "student"): ArmState(3, 5),
                              (0, "teacher"): ArmState(9, 1)})
    pooled = r.pooled("student")
    assert (pooled.alpha, pooled.beta) == (7.0, 7.0)


def test_pooled_routing_ignores_the_floor():
    """Every cluster alone is below the floor, which would force the teacher. Unassigned traffic is not
    evidence of a bad cluster, so it is routed on the pooled comparison -- and here the teacher is expensive."""
    state = {(c, "student"): ArmState(4.0, 8.0) for c in range(3)}
    state.update({(c, "teacher"): ArmState(5.0, 7.0) for c in range(3)})
    r = ThompsonRouter(state=state, cost={"student": 0.0, "teacher": 1.0}, lam=20.0, seed=0)
    assert all(r.choose(c) == "teacher" for c in range(3)), "the floor engages per cluster"
    assert {r.choose_pooled() for _ in range(50)} == {"student"}, "pooled routing must not apply the floor"


# --------------------------------------------------------------------------------------------------------------
# the health tracker
# --------------------------------------------------------------------------------------------------------------


def test_a_quiet_window_is_healthy_whatever_the_rate():
    t = HealthTracker()
    for i in range(10):
        t.record(fallback=True, unassigned=True, now=1000 + i)
    snap = t.snapshot(now=1010)
    assert snap["ok"] and snap["fallback_rate"] == 1.0, "ten requests is noise"


def test_sustained_unassigned_traffic_is_a_problem():
    t = HealthTracker()
    for i in range(30):
        t.record(fallback=False, unassigned=i % 3 != 0, now=1000 + i)
    snap = t.snapshot(now=1030)
    assert not snap["ok"]
    assert any("unassigned cluster rate" in p for p in snap["problems"])


def test_old_events_leave_the_window():
    t = HealthTracker(window_s=60)
    for i in range(30):
        t.record(fallback=True, unassigned=False, now=1000 + i)
    assert t.snapshot(now=2000)["requests"] == 0


# --------------------------------------------------------------------------------------------------------------
# the gateway
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def unplaced(registry):
    state = build_state(registry)
    state.clusters = NoClusterModel("no cluster model in the registry")
    state.router = ThompsonRouter(seed=0)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        yield c, state


def _ask(c, model=TEACHER_NAME):
    return c.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hi"}]})


def test_an_unassigned_request_is_logged_with_its_reason(unplaced):
    c, state = unplaced
    assert _ask(c).status_code == 200
    row = state.log.recent()[0]
    assert row["cluster_id"] is None
    assert row["routing_reason"] == "no_cluster_model"


def test_healthz_names_the_missing_cluster_model_and_calibration(registry):
    state = build_state(registry, threshold=None)
    state.clusters = NoClusterModel("no cluster model in the registry")
    state.calibration_state = {"state": "missing", "reason": "calibration cal_1 has verdict 'uninformative'"}
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        health = c.get("/healthz").json()
    assert health["cluster_model"] == {"state": "missing", "reason": "no cluster model in the registry"}
    assert health["calibration"]["state"] == "missing"
    assert "uninformative" in health["calibration"]["reason"]


def test_healthz_reports_a_loaded_calibration_with_its_threshold(registry):
    state = build_state(registry)
    state.calibration_state = {"state": "loaded"}
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        health = c.get("/healthz").json()
    assert health["calibration"] == {"state": "loaded", "threshold": 0.5}


def test_sustained_unassigned_traffic_fails_healthz(unplaced):
    c, _ = unplaced
    for _ in range(25):
        _ask(c)
    health = c.get("/healthz").json()
    assert health["traffic"]["unassigned_rate"] == 1.0
    assert health["ok"] is False
    assert any("cluster model may be missing" in n for n in health["notes"])


def test_the_tracker_is_fed_on_the_fallback_path(registry):
    """A tracker fed only on the happy path says ok while everything escalates."""
    state = broken_student(build_state(registry))
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        for _ in range(3):
            _ask(c, model="student")
    assert state.tracker.snapshot()["fallback_rate"] == 1.0


def test_load_state_attaches_the_cluster_model_or_a_reason(registry, project_config):
    from agentdistill.gateway.backends import StubBackend
    from agentdistill.gateway.state import load_state

    state = load_state(project_config, registry, student=StubBackend([]), teacher=StubBackend([]))
    assert isinstance(state.clusters, NoClusterModel)
    assert any("no cluster model" in n for n in state.notes)
    assert state.health()["cluster_model"]["state"] == "missing"
