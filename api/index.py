"""The gateway, deployed as a serverless function, with stubs where the hardware would be.

What is real here: the FastAPI gateway itself, both API dialects, model-name resolution, the cascade's
escalation rule, the request log, the feedback endpoint that moves the router's posteriors, and /healthz
reporting what it could and could not load. Those are the pieces this project is about, and they are the same
code the GPU day serves.

What is not real, and is labelled as such on every response: the student is a canned reply, not a trained model
(a serverless function has no GPU and no vLLM), the teacher is a canned reply too (a demo must not spend anyone's
teacher budget), and the registry is created empty in /tmp on each cold start, so the request log is real but
lives as long as the instance does.

The demo therefore shows the machinery and refuses to imply a measurement: with no adapter in prod and no
calibration, /healthz says so, and every cascade turn escalates -- which is the documented default, not a bug.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# The package is bundled from the repository rather than installed, so the function carries only the seven
# dependencies the gateway actually imports instead of the full training stack.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.responses import HTMLResponse  # noqa: E402

from agentdistill.gateway import app as gateway  # noqa: E402
from agentdistill.gateway.log import RequestLog  # noqa: E402
from agentdistill.gateway.state import GatewayState  # noqa: E402
from agentdistill.registry.base import Registry  # noqa: E402
from agentdistill.router.clusters import NoClusterModel  # noqa: E402

DEMO_NOTE = (
    "demo deployment: the student and the teacher are canned replies, not models. The gateway, both dialects, "
    "the cascade's escalation rule, the request log and /healthz are the real code."
)

STUDENT_REPLY = (
    "[demo student] A trained student would answer here after calling the CRM tools. This deployment has no GPU "
    "and no model server, so this text is canned."
)
TEACHER_REPLY = (
    "[demo teacher] The gate escalated this turn, which is what happens with no usable calibration. A real "
    "deployment would call the configured teacher here; this one does not spend anyone's budget."
)


class CannedBackend:
    """A backend shaped like the student's and the teacher's, answering with fixed text.

    It reports token usage so the request log's columns fill the way they do in production, and it never calls
    out to anything: a public demo that could reach a paid API is a public demo that will.
    """

    def __init__(self, text: str, tokens: int = 48) -> None:
        self.text, self.tokens = text, tokens

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, n: int = 1,
                   **kwargs: Any) -> dict:
        choice = {
            "message": {"role": "assistant", "content": self.text},
            "finish_reason": "stop",
            # The cascade scores logprobs; without a calibrator it never reads these, but the shape must be right.
            "logprobs": {"content": [{"token": "x", "logprob": -0.2, "top_logprobs": []} for _ in range(8)]},
            "text": self.text,
        }
        prompt_tokens = sum(len(str(m.get("content") or "")) for m in messages) // 4
        return {
            "choices": [dict(choice) for _ in range(max(n, 1))],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": self.tokens,
                      "total_tokens": prompt_tokens + self.tokens},
        }


def _registry() -> Registry:
    """A registry in /tmp: the only writable place a serverless function has.

    Created empty, so the demo starts with no adapter in prod and no calibration -- which is exactly the state
    /healthz is designed to report honestly, and the state every new deployment really begins in.
    """
    path = Path(os.environ.get("AGENTDISTILL_DEMO_DB", "/tmp/agentdistill-demo.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = Registry(f"sqlite:///{path}")
    registry.migrate()
    return registry


def build_state() -> GatewayState:
    registry = _registry()
    return GatewayState(
        student=CannedBackend(STUDENT_REPLY),
        teacher=CannedBackend(TEACHER_REPLY),
        registry=registry,
        log=RequestLog(registry),
        teacher_names={"demo-teacher"},
        # Nothing is trained, nothing is calibrated, and there is no cluster model: the honest starting state.
        clusters=NoClusterModel("this demo deployment ships no cluster model; requests route on the pooled "
                                "posterior and are counted as unassigned"),
        notes=[DEMO_NOTE],
    )


gateway.set_state(build_state())
app = gateway.app


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing() -> str:
    """A page rather than a 404, because the first thing anyone does with a deployed URL is open it."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>agentdistill gateway (demo)</title>
<style>
 body {{ font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        max-width: 48rem; margin: 0 auto; padding: 3rem 1.25rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }} .sub {{ color: #666; margin-top: 0; }}
 .warn {{ background: #fff8e1; border-left: 4px solid #e6c200; padding: .85rem 1rem; margin: 1.5rem 0;
         font-size: .95rem; }}
 pre {{ background: #fafafa; border: 1px solid #e3e3e3; padding: .8rem; overflow-x: auto; font-size: .82rem;
       border-radius: 6px; }}
 code {{ font: .88em ui-monospace, SFMono-Regular, Menlo, monospace; }}
 a {{ color: #0b5bd3; }} th, td {{ text-align: left; padding: .35rem .6rem .35rem 0; font-size: .93rem; }}
</style></head><body>
<h1>agentdistill gateway</h1>
<p class="sub">An OpenAI- and Anthropic-compatible endpoint in front of a distilled student, with a calibrated
gate that escalates the turns it is not confident about.</p>

<div class="warn"><strong>This is a demo deployment.</strong> The student and the teacher are canned replies,
not models: a serverless function has no GPU, and a public URL must not spend a teacher budget. The gateway,
both dialects, the escalation rule, the request log and the health endpoint are the real code, running the same
paths the GPU day serves. Nothing here is a measurement of a model.</div>

<h2>Try it</h2>
<pre>curl -s $URL/healthz | python3 -m json.tool

curl -s $URL/v1/chat/completions -H 'content-type: application/json' \\
  -d '{{"model":"student","messages":[{{"role":"user","content":"where is order o_1?"}}]}}'

curl -s $URL/v1/messages -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \\
  -d '{{"model":"student","max_tokens":256,"messages":[{{"role":"user","content":"refund o_2 please"}}]}}'</pre>

<table>
<tr><th><a href="/healthz">/healthz</a></th><td>what loaded and what did not: cluster model, calibration,
  rolling fallback and unassigned rates</td></tr>
<tr><th><a href="/v1/models">/v1/models</a></th><td>the names this gateway answers to</td></tr>
<tr><th>/v1/chat/completions</th><td>OpenAI dialect; the reply carries an <code>agentdistill</code> block naming
  the arm that answered</td></tr>
<tr><th>/v1/messages</th><td>Anthropic dialect; the same detail travels in <code>x-agentdistill-*</code>
  headers, because that response schema is closed</td></tr>
<tr><th>/v1/feedback</th><td>report whether a request worked; this is what moves the router's posteriors</td></tr>
</table>

<p>Ask for <code>cascade::auto</code> and every turn escalates to the teacher. That is the documented default
with no usable calibration, not a failure: a cascade with a meaningless gate is worse than no cascade.</p>

<p><a href="https://charipramod.github.io/agentdistill/">The pipeline's report</a> &middot;
<a href="https://github.com/ChariPramod/agentdistill">Source</a></p>
</body></html>"""
