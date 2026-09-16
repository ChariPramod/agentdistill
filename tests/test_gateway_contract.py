"""Contract tests: the official SDKs against the gateway.

The whole promise is "drop-in": the agent keeps its SDK and its model name and points `base_url` at the gateway.
A response the gateway thinks is well-formed but the SDK refuses to parse breaks that promise, and no amount of
internal testing catches it -- only the real client does.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agentdistill.gateway import app as app_module
from agentdistill.gateway.backends import StubBackend
from agentdistill.gateway.state import GatewayState
from tests.conftest import make_call

openai_sdk = pytest.importorskip("openai", reason="the openai package is required for the contract test")
anthropic_sdk = pytest.importorskip("anthropic", reason="the anthropic package is required for the contract test")

TEACHER_NAME = "frontier-model-v1"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "refund_order",
        "description": "Refund an order.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]},
    },
}]

ANTHROPIC_TOOLS = [{
    "name": "refund_order",
    "description": "Refund an order.",
    "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
}]


class StubCalibrator:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


@pytest.fixture
def gateway(registry):
    """A gateway whose teacher answers with a tool call, so both dialects carry one."""
    from agentdistill.cascade.features import DEFAULT_FEATURES
    from agentdistill.gateway.log import RequestLog

    tool_choice = {
        "message": {"role": "assistant", "content": None,
                    "tool_calls": [make_call("call_1", "refund_order", {"order_id": "o_1"})]},
        "finish_reason": "tool_calls",
    }
    state = GatewayState(
        student=StubBackend([{**tool_choice,
                              "logprobs": {"content": [{"token": "t", "logprob": -0.2, "top_logprobs": []}]},
                              "text": "t"}]),
        teacher=StubBackend([tool_choice]),
        registry=registry,
        log=RequestLog(registry),
        prod_adapter="prod-v1",
        prod_threshold=0.5,
        calibrator=StubCalibrator(),
        feature_names=list(DEFAULT_FEATURES),
        teacher_names={TEACHER_NAME},
    )
    app_module.set_state(state)
    with TestClient(app_module.app) as http:
        yield http, state


# --------------------------------------------------------------------------------------------------------------
# the OpenAI SDK
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def openai_client(gateway):
    http, _ = gateway
    return openai_sdk.OpenAI(api_key="none", base_url=f"{http.base_url}/v1", http_client=http)


def test_openai_sdk_parses_a_tool_call(openai_client):
    response = openai_client.chat.completions.create(
        model=TEACHER_NAME,
        messages=[{"role": "user", "content": "refund order o_1"}],
        tools=TOOLS,
    )
    call = response.choices[0].message.tool_calls[0]
    assert call.function.name == "refund_order"
    assert json.loads(call.function.arguments) == {"order_id": "o_1"}
    assert call.type == "function"
    assert response.choices[0].finish_reason == "tool_calls"


def test_openai_sdk_reads_usage(openai_client):
    response = openai_client.chat.completions.create(
        model="student", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


def test_openai_sdk_keeps_working_with_the_agents_own_model_name(openai_client):
    """The drop-in promise: the agent changes base_url and nothing else."""
    response = openai_client.chat.completions.create(
        model=TEACHER_NAME, messages=[{"role": "user", "content": "hi"}], tools=TOOLS
    )
    assert response.choices[0].message.tool_calls


def test_openai_sdk_streaming(openai_client):
    stream = openai_client.chat.completions.create(
        model="student", messages=[{"role": "user", "content": "hi"}], tools=TOOLS, stream=True
    )
    chunks = list(stream)
    assert chunks, "the SDK parsed no chunks"
    assert any(c.choices and c.choices[0].delta.tool_calls for c in chunks)
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_openai_sdk_rejects_an_unknown_model(openai_client):
    with pytest.raises(openai_sdk.BadRequestError):
        openai_client.chat.completions.create(
            model="not-a-known-model", messages=[{"role": "user", "content": "hi"}]
        )


# --------------------------------------------------------------------------------------------------------------
# the Anthropic SDK
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def anthropic_client(gateway):
    http, _ = gateway
    return anthropic_sdk.Anthropic(api_key="none", base_url=str(http.base_url), http_client=http)


def test_anthropic_sdk_parses_a_tool_use_block(anthropic_client):
    message = anthropic_client.messages.create(
        model=TEACHER_NAME,
        max_tokens=256,
        messages=[{"role": "user", "content": "refund order o_1"}],
        tools=ANTHROPIC_TOOLS,
    )
    block = message.content[0]
    assert block.type == "tool_use"
    assert block.name == "refund_order"
    assert block.input == {"order_id": "o_1"}
    assert block.id.startswith("toolu_")
    assert message.stop_reason == "tool_use"


def test_anthropic_sdk_reads_usage(anthropic_client):
    message = anthropic_client.messages.create(
        model="student", max_tokens=64, messages=[{"role": "user", "content": "hi"}]
    )
    assert message.usage.input_tokens >= 0 and message.usage.output_tokens >= 0


def test_anthropic_sdk_round_trips_a_tool_result(anthropic_client):
    """A full agent turn: the SDK sends back a tool_result and the gateway converts it."""
    message = anthropic_client.messages.create(
        model=TEACHER_NAME,
        max_tokens=256,
        messages=[
            {"role": "user", "content": "refund order o_1"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_prev", "name": "refund_order", "input": {"order_id": "o_1"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_prev", "content": '{"ok": true}'}]},
        ],
        tools=ANTHROPIC_TOOLS,
    )
    assert message.content


def test_anthropic_sdk_streaming(anthropic_client):
    with anthropic_client.messages.stream(
        model="student", max_tokens=64,
        messages=[{"role": "user", "content": "hi"}], tools=ANTHROPIC_TOOLS,
    ) as stream:
        final = stream.get_final_message()
    assert final.content[0].type == "tool_use"
    assert final.stop_reason == "tool_use"


# --------------------------------------------------------------------------------------------------------------
# what the gateway told the caller
# --------------------------------------------------------------------------------------------------------------


def test_routing_detail_reaches_the_caller_in_both_dialects(gateway):
    http, _ = gateway
    openai_response = http.post("/v1/chat/completions", json={
        "model": "student", "messages": [{"role": "user", "content": "hi"}]})
    assert openai_response.json()["agentdistill"]["arm"] == "student"

    anthropic_response = http.post("/v1/messages", json={
        "model": "student", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]})
    assert anthropic_response.headers["x-agentdistill-arm"] == "student"
    assert "agentdistill" not in anthropic_response.json(), "the Anthropic schema is closed; detail goes in headers"
