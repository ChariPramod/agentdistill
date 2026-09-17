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

# What the example agent asks the gateway for. `cascade::auto` is the real thing: the gate decides per turn and
# escalates to the teacher. Tiny mode has no teacher to escalate to, so it drives the student alone -- which
# still exercises the gateway, both dialects, the request log and the fake vLLM, and is honest about not
# exercising escalation.
if [[ "${AGENTDISTILL_TINY:-0}" == "1" ]]; then
  SMOKE_MODEL="${SMOKE_MODEL:-student}"
else
  SMOKE_MODEL="${SMOKE_MODEL:-cascade::auto}"
fi
OUT="${OUT_DIR:-$(mktemp -d)}"
mkdir -p logs

cleanup() {
  local status=$?
  [[ -n "${GW_PID:-}"   ]] && kill "$GW_PID"   2>/dev/null || true
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    # Print the gateway's last error rather than only naming the file. A smoke test that fails and makes you
    # go reading is a smoke test you stop running.
    echo "failed; see logs/vllm.log and logs/gateway.log" >&2
    if [[ -s logs/gateway.log ]]; then
      echo "--- last gateway error ---" >&2
      grep -E "Error|Exception|Traceback|detail" logs/gateway.log 2>/dev/null | tail -5 >&2 || true
    fi
  fi
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
if [[ "${AGENTDISTILL_FAKE_VLLM:-0}" == "1" ]]; then
  # The test suite's fake vLLM, so the rest of this script -- the gateway, both dialects, the request log --
  # runs on a laptop. It does not rehearse vLLM; nothing on a laptop can.
  echo "    (fake vLLM: this rehearses everything except vLLM)"
  python -m tests.fake_vllm --port "$VLLM_PORT" > logs/vllm.log 2>&1 &
else
  bash scripts/serve_vllm.sh > logs/vllm.log 2>&1 &
fi
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

echo "==> driving the example agent (openai dialect) as ${SMOKE_MODEL}"
python -m examples.support_agent.record \
  --model "openai/${SMOKE_MODEL}" --base-url "http://127.0.0.1:${GW_PORT}/v1" \
  --n 5 --out "$OUT/smoke_openai.jsonl"

echo "==> driving the example agent (anthropic dialect) as ${SMOKE_MODEL}"
python -m examples.support_agent.record \
  --model "anthropic/${SMOKE_MODEL}" --base-url "http://127.0.0.1:${GW_PORT}" \
  --n 5 --out "$OUT/smoke_anthropic.jsonl"

AFTER="$("$AD" requests count --config "$CFG")"
LOGGED=$((AFTER - BEFORE))
if [[ $LOGGED -lt 10 ]]; then
  echo "expected at least 10 request-log rows, got $LOGGED" >&2
  exit 1
fi

# A run where every request fell back is a run where vLLM never served anything, and it would otherwise pass:
# the gateway answers, the agent gets replies, and the request log fills up.
#
# `fallbacks`, not `requests` -- the health block reports both, and reading the wrong one makes this fire on
# every healthy run.
FALLBACKS="$(curl -s "http://127.0.0.1:${GW_PORT}/healthz" | python -c \
  'import json,sys; print(json.load(sys.stdin).get("fallback",{}).get("fallbacks",0))')"
if [[ "$FALLBACKS" -ge $LOGGED ]]; then
  echo "every one of the $LOGGED requests fell back to the teacher; the student never served" >&2
  exit 1
fi

"$AD" requests tail --n 10 --config "$CFG"
echo "serve smoke ok: $LOGGED requests logged, $FALLBACKS fallbacks"
