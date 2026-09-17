#!/usr/bin/env bash
# Start vLLM and the gateway, drive the example agent through both dialects, assert the request log grew, and
# tear everything down.
#
# This is the check that the serving path works end to end on real hardware. Unit tests cover each piece against
# stubs; only this catches a tool parser that disagrees with the template, a LoRA that vLLM declines to load, or
# a gateway that answers /healthz and 500s on traffic.
set -euo pipefail

CFG="${AGENTDISTILL_CONFIG:-project.yaml}"
AD="${AGENTDISTILL:-agentdistill}"
VLLM_PORT="${VLLM_PORT:-8000}"
GW_PORT="${GW_PORT:-8710}"
OUT="${OUT_DIR:-$(mktemp -d)}"
mkdir -p logs

cleanup() {
  local status=$?
  [[ -n "${GW_PID:-}"   ]] && kill "$GW_PID"   2>/dev/null || true
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
  # On failure the logs are the only evidence of what went wrong, so say where they are.
  [[ $status -ne 0 ]] && echo "failed; see logs/vllm.log and logs/gateway.log" >&2
  return $status
}
trap cleanup EXIT

wait_for() {  # wait_for <url> <attempts> <sleep> <what>
  for _ in $(seq 1 "$2"); do
    curl -sf "$1" >/dev/null && return 0
    sleep "$3"
  done
  echo "$4 did not come up at $1" >&2
  return 1
}

echo "==> starting vLLM"
bash scripts/serve_vllm.sh > logs/vllm.log 2>&1 &
VLLM_PID=$!
wait_for "http://127.0.0.1:${VLLM_PORT}/v1/models" 60 5 "vllm"

echo "==> starting the gateway"
"$AD" serve --port "$GW_PORT" --config "$CFG" > logs/gateway.log 2>&1 &
GW_PID=$!
wait_for "http://127.0.0.1:${GW_PORT}/healthz" 30 2 "gateway"

# A gateway that is up but degraded still answers /healthz. Print what it thinks of itself before trusting it.
echo "==> gateway health"
curl -s "http://127.0.0.1:${GW_PORT}/healthz" | python -m json.tool

BEFORE="$("$AD" requests count --config "$CFG")"

echo "==> driving the example agent (openai dialect)"
python -m examples.support_agent.record \
  --model "openai/cascade::auto" --base-url "http://127.0.0.1:${GW_PORT}/v1" \
  --n 5 --out "$OUT/smoke_openai.jsonl"

echo "==> driving the example agent (anthropic dialect)"
python -m examples.support_agent.record \
  --model "anthropic/cascade::auto" --base-url "http://127.0.0.1:${GW_PORT}" \
  --n 5 --out "$OUT/smoke_anthropic.jsonl"

AFTER="$("$AD" requests count --config "$CFG")"
LOGGED=$((AFTER - BEFORE))
if [[ $LOGGED -lt 10 ]]; then
  echo "expected at least 10 request-log rows, got $LOGGED" >&2
  exit 1
fi

# A run where every request fell back is a run where vLLM never served anything, and it would otherwise pass.
FALLBACKS="$(curl -s "http://127.0.0.1:${GW_PORT}/healthz" | python -c \
  'import json,sys; print(json.load(sys.stdin).get("fallback",{}).get("requests",0))')"
if [[ "$FALLBACKS" -ge $LOGGED ]]; then
  echo "every one of the $LOGGED requests fell back to the teacher; the student never served" >&2
  exit 1
fi

"$AD" requests tail --n 10 --config "$CFG"
echo "serve smoke ok: $LOGGED requests logged, $FALLBACKS fallbacks"
