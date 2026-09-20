# Serving

Two processes: vLLM holds the weights, and the gateway sits in front of the agent speaking the agent's own API.

## Starting vLLM

```bash
bash scripts/serve_vllm.sh            # build the command and run it
bash scripts/serve_vllm.sh --dry-run  # print it and stop
```

Every setting comes from `project.yaml` or the registry. A serve command with its own hardcoded model name is
how a gateway ends up routing to an adapter retired three weeks ago, and how `--quantization fp8` ends up on a
command serving weights that were never quantized.

`--enable-prefix-caching` is always on. Agent prompts repeat a system prompt and tool schemas on every turn of
every task; without prefix caching the throughput figure in the cost model is wrong by a large factor.

The prod adapter and the canary are both loaded as LoRA modules when they exist. With neither, the base model
is served alone and the script says so.

## Quantization

| method | what happens |
|---|---|
| `fp8` | vLLM applies it online from bf16 weights. Nothing is written but a marker, so the registry, the serve script and anyone reading the directory agree on what is served. |
| `awq` | A real offline pass that rewrites weights to 4 bits. Much larger saving, much larger risk. |
| `gptq` | Accepted in config for forward compatibility; not implemented. |

AWQ calibrates on **real task prompts** from the training set, never generic text. AWQ decides which channels
to preserve by watching activations on what it is shown; an agent's prompts are dominated by tool schemas, and
a quantizer that never saw one will sacrifice the channels that emit them. The symptom is a model that chats
fine and produces malformed tool calls. Below 32 prompts it refuses, before loading the model, so the failure
costs seconds rather than a GPU-hour.

Quantization may cost at most 2 pp of success against the bf16 weights. Scoring *higher* is not refused: that
is evidence the eval set is too small to resolve the difference, not that quantization improved the model.

## Merging

Merging is not required — vLLM serves LoRA directly — but quantizers need a single set of weights.

Always into bf16, never into a quantized base. Merging into 4-bit means dequantize, add the delta, requantize,
and the round trip loses more than the adapter contributed. That failure is silent: the model loads, generates
fluent text, and is worse.

`adapter merge` verifies by running teacher-forced next-action prediction through both the merged weights and
the unmerged adapter on held-out turns, paired per turn, and refuses a drift beyond 2 pp. That catches a
`target_modules` list that missed a projection or a base revision that moved. Note the granularity: on 50 turns
one disagreement is already 2 pp, so at that size the check passes only an exact reproduction. The result says
so when that applies.

## The gateway

```bash
agentdistill serve --port 8710
```

Model names are the API:

| requested model | behaviour |
|---|---|
| the agent's own model name | the router decides: cascade or teacher passthrough |
| `teacher` | passthrough, for paired evals |
| `student` / `student:<adapter>` | student only, never escalates |
| `cascade:<adapter>:<tau\|auto>` | a fixed cascade at that threshold |

Only the agent's own name goes through the router. The eval names exist so a report can measure each arm in
isolation; the routed name is the one whose behaviour has to stay invisible to the caller.

**With no adapter in prod, or no usable calibration, requests pass through to the teacher.** The gateway sits
in front of a working agent and must never be the reason that agent starts behaving differently.

**The gateway scores with exactly the features the calibration was fitted on, in that order.** At boot it reads
the calibration's `feature_order` (from the registry row, checked against `calibration.json`) and compares it with
`cascade.features`. The stored order may be a subset -- calibration drops features that had no values -- and then
the gateway builds the narrower vector. If a fitted feature is not configured, or `cascade.features` lists them in
a different relative order, the calibration is refused: `/healthz` shows `calibration.state: missing` with the
fitted and configured orders in `reason`, and every turn escalates. Recalibrate after editing `cascade.features`.
`eval run cascade:...` applies the same rule, so the cascade that is measured is the one that would serve.

Both dialects are supported: OpenAI at `/v1/chat/completions` and Anthropic at `/v1/messages`. Both are tested
against the real `openai` and `anthropic` SDKs, including streaming.

## Streaming

Streaming requests are answered as a stream, but the response is **buffered**: the cascade cannot decide
whether to escalate until the student's turn is complete, and a token already sent cannot be recalled. The
response carries `x-agentdistill-buffered: true` so a caller measuring time-to-first-token knows why it moved.

## Feedback

```
POST /v1/feedback  {"request_id": "...", "success": true}
```

This is what turns the request log into training data and what lets the router learn. A gateway with no
feedback still serves and still saves money; it just never improves, and `ingest gateway` finds nothing to
retrain on.

The response reports `router_updated`. It is `false` for a fallback, for an unclustered request, and when no
router is loaded.

## Smoke test

```bash
bash scripts/serve_smoke.sh
```

Starts both processes, drives the example agent through both dialects, asserts the request log grew by at least
ten rows, and fails if *every* request fell back — without that check, a run where vLLM never served anything
would pass, because the gateway falls back, every request succeeds, and the log fills up.
