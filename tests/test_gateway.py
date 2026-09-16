"""The gateway: model-name policies, dialects, routing, and the request log.

Everything runs in-process against the fake vLLM and a stub teacher, so the whole gateway is exercised on CPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agentdistill.gateway import app as app_module
from agentdistill.gateway.backends import StubBackend
from agentdistill.gateway.dialect import (
    anthropic_stream_events,
    from_anthropic_request,
    from_openai_request,
    openai_stream_events,
    to_anthropic_response,
    to_openai_response,
)
from agentdistill.gateway.resolve import Route, UnknownModel, resolve
from agentdistill.gateway.state import GatewayState
from tests.conftest import make_call

TEACHER_NAME = "frontier-model-v1"


# --------------------------------------------------------------------------------------------------------------
# model-name policies
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("teacher", Route("teacher")),
        ("student", Route("student", "prod-v1")),
        ("student:other", Route("student", "other")),
        ("cascade:a:0.62", Route("cascade", "a", 0.62)),
        ("cascade:a:auto", Route("cascade", "a", 0.5)),
        ("cascade::auto", Route("cascade", "prod-v1", 0.5)),
        (TEACHER_NAME, Route("router", "prod-v1", 0.5)),
    ],
)
def test_every_model_name_policy(model, expected):
    assert resolve(model, "prod-v1", 0.5, {TEACHER_NAME}) == expected


def test_the_agents_own_model_passes_through_when_nothing_is_deployed():
    """The gateway must never be the reason an agent starts behaving differently."""
    assert resolve(TEACHER_NAME, None, None, {TEACHER_NAME}).mode == "teacher"


def test_unknown_model_names_the_alternatives():
    with pytest.raises(UnknownModel, match="Known:"):
        resolve("some-other-model", "prod-v1", 0.5, {TEACHER_NAME})


@pytest.mark.parametrize("model", ["cascade:a", "cascade:a:b:c", "cascade:a:high", "cascade:a:1.5"])
def test_malformed_cascade_names_are_rejected(model):
    with pytest.raises(UnknownModel):
        resolve(model, "prod-v1", 0.5, {TEACHER_NAME})


# --------------------------------------------------------------------------------------------------------------
# dialects
# --------------------------------------------------------------------------------------------------------------


def test_anthropic_request_becomes_openai_messages():
    body = {
        "model": "m", "max_tokens": 100, "system": "You are support.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "refund please"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "refund", "input": {"id": "o1"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
        ],
        "tools": [{"name": "refund", "input_schema": {"type": "object"}}],
    }
    req = from_anthropic_request(body)
    assert [m["role"] for m in req["messages"]] == ["system", "user", "assistant", "tool"]
    assert req["tools"][0]["type"] == "function"


def test_openai_choice_becomes_anthropic_tool_use():
    choice = {"message": {"role": "assistant", "content": None,
                          "tool_calls": [make_call("call_1", "refund", {"id": "o1"})]}}
    out = to_anthropic_response(choice, "m", {"prompt_tokens": 10, "completion_tokens": 5})
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0]["type"] == "tool_use"
    assert out["content"][0]["input"] == {"id": "o1"}
    assert out["content"][0]["id"].startswith("toolu_"), "the Anthropic SDK expects its own id shape"
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_anthropic_text_response():
    choice = {"message": {"role": "assistant", "content": "It shipped."}}
    out = to_anthropic_response(choice, "m", {})
    assert out["stop_reason"] == "end_turn"
    assert out["content"] == [{"type": "text", "text": "It shipped."}]


def test_malformed_tool_arguments_are_surfaced_not_dropped():
    """Losing a call silently is worse than surfacing a bad one."""
    choice = {"message": {"role": "assistant", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{broken"}}]}}
    out = to_anthropic_response(choice, "m", {})
    assert out["content"][0]["input"] == {"_raw": "{broken"}


def test_openai_response_always_marks_tool_calls_as_functions():
    choice = {"message": {"role": "assistant", "tool_calls": [
        {"id": "c", "function": {"name": "f", "arguments": "{}"}}]}}
    out = to_openai_response(choice, "m", {})
    assert out["choices"][0]["message"]["tool_calls"][0]["type"] == "function"
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_transport_only_keys_never_reach_the_client():
    choice = {"message": {"role": "assistant", "content": "x", "_raw": {"secret": 1}}}
    assert "_raw" not in to_openai_response(choice, "m", {})["choices"][0]["message"]


def test_openai_request_accepts_either_max_tokens_spelling():
    assert from_openai_request({"model": "m", "messages": [], "max_completion_tokens": 42})["max_tokens"] == 42


def test_usage_totals_are_filled_in():
    out = to_openai_response({"message": {"role": "assistant", "content": "x"}}, "m",
                             {"prompt_tokens": 3, "completion_tokens": 4})
    assert out["usage"]["total_tokens"] == 7


# --------------------------------------------------------------------------------------------------------------
# streaming frames
# --------------------------------------------------------------------------------------------------------------


def test_openai_stream_ends_with_done():
    events = openai_stream_events({"message": {"role": "assistant", "content": "hi"}}, "m", {})
    assert events[-1] == "data: [DONE]\n\n"
    assert any('"content": "hi"' in e or '"content":"hi"' in e for e in events)


def test_anthropic_stream_has_the_expected_event_sequence():
    events = anthropic_stream_events({"message": {"role": "assistant", "content": "hi"}}, "m", {})
    names = [e.split("\n")[0].removeprefix("event: ") for e in events]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert "content_block_start" in names and "content_block_stop" in names


def test_stream_carries_tool_calls():
    choice = {"message": {"role": "assistant", "content": None,
                          "tool_calls": [make_call("c", "refund", {"id": "o1"})]}}
    assert any("tool_calls" in e for e in openai_stream_events(choice, "m", {}))
    assert any("input_json_delta" in e for e in anthropic_stream_events(choice, "m", {}))


# --------------------------------------------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------------------------------------------


class StubCalibrator:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


def build_state(registry, *, p: float = 0.9, threshold: float | None = 0.5, prod: str | None = "prod-v1",
                teacher_reply: dict | None = None, student_reply: dict | None = None) -> GatewayState:
    from agentdistill.cascade.features import DEFAULT_FEATURES
    from agentdistill.gateway.log import RequestLog

    student_choice = {
        "message": student_reply or {"role": "assistant", "content": "student says hi"},
        "finish_reason": "stop",
        "logprobs": {"content": [{"token": "t", "logprob": -0.2, "top_logprobs": []} for _ in range(4)]},
        "text": "tttt",
    }
    teacher_choice = {"message": teacher_reply or {"role": "assistant", "content": "teacher says hi"},
                      "finish_reason": "stop"}
    return GatewayState(
        student=StubBackend([student_choice]),
        teacher=StubBackend([teacher_choice]),
        registry=registry,
        log=RequestLog(registry),
        prod_adapter=prod,
        prod_threshold=threshold,
        calibrator=StubCalibrator(p) if threshold is not None else None,
        feature_names=list(DEFAULT_FEATURES),
        teacher_names={TEACHER_NAME},
        k_samples=2,
    )


@pytest.fixture
def client(registry):
    state = build_state(registry)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        yield c, state


def ask(client, model=TEACHER_NAME, **over):
    body = {"model": model, "messages": [{"role": "user", "content": "hello"}], **over}
    return client.post("/v1/chat/completions", json=body)


def test_student_route_returns_the_students_answer(client):
    c, _ = client
    r = ask(c, model="student")
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "student says hi"
    assert r.json()["agentdistill"]["arm"] == "student"


def test_teacher_route_returns_the_teachers_answer(client):
    c, _ = client
    r = ask(c, model="teacher")
    assert r.json()["choices"][0]["message"]["content"] == "teacher says hi"
    assert r.headers["x-agentdistill-arm"] == "teacher"


def test_confident_cascade_keeps_the_student(client):
    c, _ = client
    r = ask(c, model="cascade:prod-v1:0.5")
    assert r.json()["choices"][0]["message"]["content"] == "student says hi"
    assert r.headers["x-agentdistill-escalated"] == "false"


def test_unconfident_cascade_escalates(registry):
    state = build_state(registry, p=0.05, threshold=0.5)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        r = ask(c, model="cascade:prod-v1:0.5")
    assert r.json()["choices"][0]["message"]["content"] == "teacher says hi"
    assert r.headers["x-agentdistill-escalated"] == "true"


def test_a_gateway_without_a_calibration_escalates_everything(registry):
    """The documented default: a cascade with no gate is worse than no cascade."""
    state = build_state(registry, threshold=None)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        r = ask(c, model=TEACHER_NAME)
    assert r.json()["choices"][0]["message"]["content"] == "teacher says hi"


def test_unknown_model_is_a_400(client):
    c, _ = client
    assert ask(c, model="no-such-model").status_code == 400


def test_the_request_log_records_every_request(client):
    c, state = client
    ask(c, model="student")
    ask(c, model="teacher")
    rows = state.log.recent()
    assert len(rows) == 2
    assert {r["arm"] for r in rows} == {"student", "teacher"}


def test_wasted_tokens_are_logged_on_an_escalation(registry):
    state = build_state(registry, p=0.05, threshold=0.5)
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        ask(c, model="cascade:prod-v1:0.5")
    row = state.log.recent()[0]
    assert row["escalated"]
    assert json.loads(row["payload"])["wasted_student_tokens"] == 4


def test_feedback_sets_the_outcome(client):
    c, state = client
    request_id = ask(c, model="student").json()["agentdistill"]["request_id"]
    assert c.post("/v1/feedback", json={"request_id": request_id, "success": True}).status_code == 200
    assert state.log.recent()[0]["outcome"]


def test_feedback_for_an_unknown_request_is_404(client):
    c, _ = client
    assert c.post("/v1/feedback", json={"request_id": "nope", "success": True}).status_code == 404


def test_feedback_requires_a_request_id(client):
    c, _ = client
    assert c.post("/v1/feedback", json={"success": True}).status_code == 400


def test_student_failure_falls_back_to_the_teacher(registry):
    """The gateway must not be the reason a working agent breaks."""
    from agentdistill.gateway.backends import BackendError

    state = build_state(registry)

    async def broken(*a, **kw):
        raise BackendError("student is unreachable")

    state.student.chat = broken
    app_module.set_state(state)
    with TestClient(app_module.app) as c:
        r = ask(c, model="student")
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "teacher says hi"
    assert state.log.recent()[0]["arm"] == "teacher"


def test_anthropic_endpoint_round_trips(client):
    c, _ = client
    r = c.post("/v1/messages", json={
        "model": TEACHER_NAME, "max_tokens": 100,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    })
    body = r.json()
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"
    assert r.headers["x-agentdistill-arm"]


def test_streaming_is_marked_as_buffered(client):
    """The cascade cannot stream honestly, so a client can tell it watched a replay."""
    c, _ = client
    with c.stream("POST", "/v1/chat/completions",
                  json={"model": "student", "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        assert r.headers["x-agentdistill-buffered"] == "true"
        body = "".join(r.iter_text())
    assert body.rstrip().endswith("data: [DONE]")


def test_healthz_reports_what_is_loaded(client):
    c, _ = client
    health = c.get("/healthz").json()
    assert health["ok"] and health["prod_adapter"] == "prod-v1"
    assert health["cascade_available"] is True


def test_models_lists_the_eval_names(client):
    c, _ = client
    ids = {m["id"] for m in c.get("/v1/models").json()["data"]}
    assert {"teacher", "student", "student:prod-v1", "cascade:prod-v1:auto", TEACHER_NAME} <= ids
