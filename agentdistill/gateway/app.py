"""The gateway.

An OpenAI- and Anthropic-compatible endpoint that the agent points `base_url` at. The agent keeps its SDK and its
model name; routing, escalation, and the cascade are invisible to it.

Two rules shape the error handling. The gateway sits in front of a working agent, so it never becomes the reason
that agent stops working: when the student is unreachable it falls back to the teacher rather than failing the
request. And it never silently changes behaviour: every response carries an `agentdistill` block (or headers, in
the Anthropic dialect) saying which arm answered and whether the gate escalated.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from agentdistill.gateway.backends import BackendError
from agentdistill.gateway.dialect import (
    anthropic_stream_events,
    from_anthropic_request,
    from_openai_request,
    openai_stream_events,
    to_anthropic_response,
    to_openai_response,
)
from agentdistill.gateway.resolve import UnknownModel, resolve
from agentdistill.gateway.state import GatewayState
from agentdistill.router.canary import use_canary

logger = logging.getLogger(__name__)

#: Sustained fallbacks above this fail the health check and log at error level.
FALLBACK_ALERT_RATE = 0.20

app = FastAPI(title="agentdistill gateway")
gw: GatewayState = GatewayState.uninitialized()


def set_state(state: GatewayState) -> None:
    """Install the gateway's state. Called at boot, and by tests."""
    global gw
    gw = state


def _canary_for(request_id: str) -> bool:
    """Deterministic per-request split, so a retry of the same request lands on the same adapter."""
    return use_canary(request_id, gw.canary_share, gw.canary_adapter)


async def handle(req: dict) -> tuple[dict, dict, dict]:
    """Route one request. Returns (choice, usage, meta)."""
    started = time.time()
    request_id = uuid.uuid4().hex
    try:
        route = resolve(req["model"], gw.prod_adapter, gw.prod_threshold, gw.teacher_names)
    except UnknownModel as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cluster = gw.clusters.assign(req["messages"]) if gw.clusters else None
    meta: dict[str, Any] = {
        "id": request_id, "cluster_id": cluster, "route": route.mode, "adapter": route.adapter,
    }

    if route.mode == "router":
        arm = gw.router.choose(cluster) if (gw.router and cluster is not None) else "student"
        meta["router_arm"] = arm
        if arm == "teacher":
            route.mode = "teacher"
        else:
            route.mode = "cascade"
            if _canary_for(request_id):
                route.adapter = gw.canary_adapter
                meta["canary"] = True

    cluster_prior = gw.router.state_mean(cluster, "student") if (gw.router and cluster is not None) else 0.5

    try:
        choice, usage, arm_meta = await _dispatch(route, req, cluster_prior)
    except BackendError as e:
        if route.mode in ("student", "cascade") and gw.teacher is not None:
            # The gateway must not be the reason a working agent breaks.
            logger.warning("student backend failed (%s); falling back to the teacher", e)
            data = await gw.teacher.chat(req["messages"], req["tools"], temperature=req["temperature"])
            choice, usage = data["choices"][0], data.get("usage", {})
            arm_meta = {
                "arm": "teacher", "escalated": True, "fallback": True, "fallback_reason": str(e),
                "teacher_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
            await _check_fallback_rate()
        else:
            raise HTTPException(status_code=502, detail=str(e)) from e

    meta.update(arm_meta)
    meta["latency_ms"] = int((time.time() - started) * 1000)
    meta.setdefault("adapter", route.adapter)
    if gw.log is not None:
        await gw.log.write(meta, req, choice, usage)
    return choice, usage, meta


async def _dispatch(route: Any, req: dict, cluster_prior: float) -> tuple[dict, dict, dict]:
    if route.mode == "teacher":
        if gw.teacher is None:
            raise HTTPException(status_code=503, detail="no teacher is configured")
        data = await gw.teacher.chat(req["messages"], req["tools"], temperature=req["temperature"])
        usage = data.get("usage", {})
        return data["choices"][0], usage, {
            "arm": "teacher", "escalated": False, "teacher_tokens": usage.get("completion_tokens", 0),
        }

    if route.mode == "student":
        data = await gw.student.chat(
            req["messages"], req["tools"],
            model=f"student:{route.adapter}" if route.adapter else "student",
            temperature=req["temperature"],
        )
        usage = data.get("usage", {})
        return data["choices"][0], usage, {
            "arm": "student", "escalated": False, "student_tokens": usage.get("completion_tokens", 0),
        }

    return await gw.cascade_turn(
        req["messages"], req["tools"], route.adapter, route.threshold, cluster_prior, req["temperature"]
    )


# --------------------------------------------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, response: Response):
    body = await request.json()
    req = from_openai_request(body)
    choice, usage, meta = await handle(req)

    if req["stream"]:
        return _stream(openai_stream_events(choice, body["model"], usage), meta)

    out = to_openai_response(choice, body["model"], usage)
    # Non-standard, and namespaced so a strict client ignores it: which arm answered, for the feedback call.
    out["agentdistill"] = {
        "request_id": meta["id"], "arm": meta.get("arm"), "escalated": meta.get("escalated"),
        "adapter": meta.get("adapter"), "confidence": meta.get("confidence"),
    }
    _set_headers(response, meta)
    return out


@app.post("/v1/messages")
async def messages(request: Request, response: Response):
    body = await request.json()
    req = from_anthropic_request(body)
    choice, usage, meta = await handle(req)

    if req["stream"]:
        return _stream(anthropic_stream_events(choice, body["model"], usage), meta)

    # The Anthropic response schema is closed, so the routing detail goes in headers rather than the body.
    _set_headers(response, meta)
    return to_anthropic_response(choice, body["model"], usage)


def _set_headers(response: Response, meta: dict) -> None:
    response.headers["x-agentdistill-request-id"] = str(meta.get("id", ""))
    response.headers["x-agentdistill-arm"] = str(meta.get("arm", ""))
    response.headers["x-agentdistill-escalated"] = "true" if meta.get("escalated") else "false"
    if meta.get("adapter"):
        response.headers["x-agentdistill-adapter"] = str(meta["adapter"])


def _stream(events: list[str], meta: dict) -> StreamingResponse:
    """Emit a decided turn as a stream.

    The cascade cannot stream honestly: the gate needs the whole turn before it can decide whether to keep it. So
    the turn is decided first and then replayed as events, and the response says so in a header rather than
    letting a client believe it watched the model think.
    """
    async def generate():
        for event in events:
            yield event

    headers = {
        "x-agentdistill-request-id": str(meta.get("id", "")),
        "x-agentdistill-arm": str(meta.get("arm", "")),
        "x-agentdistill-buffered": "true",
        "cache-control": "no-cache",
    }
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


async def _check_fallback_rate() -> None:
    """Log loudly when fallbacks stop being occasional.

    A single fallback is a blip. A sustained rate means the student is down and every request is being billed to
    the teacher while the gateway reports success.
    """
    if gw.log is None:
        return
    stats = gw.log.fallback_rate()
    if stats["requests"] >= 5 and stats["rate"] > FALLBACK_ALERT_RATE:
        logger.error(
            "gateway fallback rate %.0f%% over the last %ds (%d of %d requests). The student backend is failing "
            "and every one of these is a teacher call nobody chose to make.",
            stats["rate"] * 100, stats["window_seconds"], stats["fallbacks"], stats["requests"],
        )


@app.post("/v1/feedback")
async def feedback(body: dict):
    """Report whether a request's task actually worked.

    This is what turns the request log into training data, and what lets the router learn. A gateway with no
    feedback still routes, but it routes on its warm start forever.
    """
    request_id = body.get("request_id")
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id is required")
    updated = await gw.log.set_outcome(request_id, bool(body.get("success")))
    if not updated:
        raise HTTPException(status_code=404, detail=f"unknown request_id {request_id}")

    record = await gw.log.get(request_id)
    # A fallback updates neither arm. The student never ran, and the teacher was not chosen on merit -- crediting
    # either would teach the router from an outage.
    if record and record.get("fallback"):
        return {"ok": True, "request_id": request_id, "router_updated": False,
                "note": "fallback request; neither arm's posterior was updated"}
    routable = bool(
        gw.router and record
        and record.get("cluster_id") is not None
        and record.get("arm") in ("student", "teacher")
    )
    if routable:
        gw.router.update(record["cluster_id"], record["arm"], bool(body.get("success")))
        if getattr(gw, "router_store", None):
            gw.router_store.flush(gw.router)
    # Always reported, not only when it is False: a caller checking whether its feedback landed should not have
    # to infer it from the absence of a key.
    return {"ok": True, "request_id": request_id, "router_updated": routable}


@app.get("/healthz")
async def healthz():
    health = gw.health()
    if gw.log is not None:
        stats = gw.log.fallback_rate()
        health["fallback"] = stats
        if stats["requests"] >= 5 and stats["rate"] > FALLBACK_ALERT_RATE:
            health["ok"] = False
            health["notes"] = [*health.get("notes", []),
                               f"fallback rate {stats['rate']:.0%} over {stats['window_seconds']}s"]
    return health


@app.get("/v1/models")
async def models():
    """What an OpenAI client sees. The eval names are listed so a harness can point at them."""
    available = ["teacher", "student"]
    if gw.prod_adapter:
        available.append(f"student:{gw.prod_adapter}")
        if gw.prod_threshold is not None:
            available.append(f"cascade:{gw.prod_adapter}:auto")
    available.extend(sorted(gw.teacher_names))
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "agentdistill"}
                                       for m in available]}
