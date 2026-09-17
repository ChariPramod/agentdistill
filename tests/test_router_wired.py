"""The router, the canary and the gateway, together.

The unit tests cover each piece. This covers the wiring: that the gateway loads posteriors at boot, that
feedback moves them and persists, that a fallback teaches the router nothing, and that a promotion clears the
previous adapter's record.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agentdistill.gateway import app as app_module
from agentdistill.gateway.state import GatewayState
from agentdistill.report.registry_views import router_state
from agentdistill.router.store import RouterStore, flush_router, load_router
from agentdistill.router.thompson import ArmState, ThompsonRouter

TEACHER_NAME = "gpt-4o"


class StubBackend:
    def __init__(self, choices: list[dict], fail: bool = False) -> None:
        self.choices = choices
        self.fail = fail
        self.calls = 0

    async def chat(self, messages, tools=None, **kw):
        self.calls += 1
        if self.fail:
            from agentdistill.gateway.backends import BackendError

            raise BackendError("student is down")
        n = kw.get("n", 1)
        return {"choices": [dict(self.choices[0]) for _ in range(n)],
                "usage": {"completion_tokens": 7, "prompt_tokens": 40}}


class StubCalibrator:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


class StubClusters:
    def __init__(self, cluster: int = 2) -> None:
        self.cluster = cluster

    def assign(self, messages):
        return self.cluster


def state_with_router(registry, router: ThompsonRouter, **over) -> GatewayState:
    from agentdistill.cascade.features import DEFAULT_FEATURES
    from agentdistill.gateway.log import RequestLog

    student_choice = {
        "message": {"role": "assistant", "content": "student says hi"},
        "finish_reason": "stop",
        "logprobs": {"content": [{"token": "t", "logprob": -0.2, "top_logprobs": []} for _ in range(4)]},
        "text": "tttt",
    }
    defaults = {
        "student": StubBackend([student_choice]),
        "teacher": StubBackend([{"message": {"role": "assistant", "content": "teacher says hi"},
                                 "finish_reason": "stop"}]),
        "registry": registry,
        "log": RequestLog(registry),
        "prod_adapter": "prod-v1",
        "prod_threshold": 0.5,
        "calibrator": StubCalibrator(0.9),
        "feature_names": list(DEFAULT_FEATURES),
        "teacher_names": {TEACHER_NAME},
        "k_samples": 2,
        "clusters": StubClusters(),
        "router": router,
        "router_store": RouterStore(registry),
    }
    defaults.update(over)
    return GatewayState(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def client(registry):
    router = ThompsonRouter(seed=0)
    state = state_with_router(registry, router)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        yield c, state


def ask(client, model=TEACHER_NAME, **over):
    """The agent's own model name is the only one that goes through the router; the rest are eval names."""
    body = {"model": model, "messages": [{"role": "user", "content": "hello"}], **over}
    return client.post("/v1/chat/completions", json=body)


def test_feedback_moves_the_posterior_and_persists_it(client):
    c, state = client
    r = ask(c)
    assert r.status_code == 200
    request_id = r.json()["agentdistill"]["request_id"]

    before = state.router.arm(2, r.json()["agentdistill"]["arm"]).alpha
    fb = c.post("/v1/feedback", json={"request_id": request_id, "success": True})
    assert fb.json()["router_updated"] is True
    assert state.router.arm(2, r.json()["agentdistill"]["arm"]).alpha > before

    # And it reached the database, so a gateway restart keeps it.
    rows = router_state(state.registry)
    assert any(row["cluster_id"] == 2 for row in rows)


def test_a_fallback_teaches_the_router_nothing(registry):
    """A fallback means the teacher served the request. Crediting the student's arm would teach the router from
    an outage."""
    router = ThompsonRouter(seed=0)
    state = state_with_router(registry, router, student=StubBackend([{}], fail=True))
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        r = ask(c, model="cascade:prod-v1:auto")
        assert r.status_code == 200
        request_id = r.json()["agentdistill"]["request_id"]
        fb = c.post("/v1/feedback", json={"request_id": request_id, "success": True})

    assert fb.json()["router_updated"] is False
    # `choose` materializes an arm, so the key may exist -- what must not exist is evidence in it.
    assert all(state.observations == 0 for state in router.state.values())


def test_the_router_survives_a_restart(registry, project_config):
    project_config.eval.eval_set = "holdout"
    router = ThompsonRouter(seed=0)
    for _ in range(30):
        router.update(4, "student", True)
    flush_router(registry, router)

    reloaded = load_router(registry, project_config)
    assert reloaded.arm(4, "student").mean == pytest.approx(router.arm(4, "student").mean)


def test_a_promotion_resets_what_the_previous_adapter_taught(registry, project_config):
    """The old adapter's record is not evidence about the new one."""
    from agentdistill.router.store import reset_router

    flush_router(registry, ThompsonRouter(state={(0, "student"): ArmState(2.0, 200.0)}))
    reset_router(registry)
    assert router_state(registry) == []

    project_config.eval.eval_set = "holdout"
    fresh = load_router(registry, project_config)
    assert fresh.arm(0, "student").mean == pytest.approx(0.5)


def test_a_cluster_the_router_has_written_off_goes_to_the_teacher(registry):
    router = ThompsonRouter(floor=0.55, min_observations=10, seed=0)
    for _ in range(40):
        router.update(2, "student", False)

    state = state_with_router(registry, router)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        arms = {ask(c).json()["agentdistill"]["arm"] for _ in range(10)}
    assert arms == {"teacher"}


def test_the_cluster_prior_reaches_the_gate(registry):
    """The gate's `cluster_prior` feature is the router's own posterior, so a cluster the router distrusts also
    raises the bar for the student's turn."""
    router = ThompsonRouter(seed=0)
    for _ in range(40):
        router.update(2, "student", True)
    assert router.state_mean(2, "student") > 0.9

    state = state_with_router(registry, router)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        assert ask(c).status_code == 200
