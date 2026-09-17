#!/usr/bin/env bash
# Build and run the vLLM serve command from project.yaml and the registry.
#
# Every setting comes from one of those two places. A serve command with its own hardcoded model name is how a
# gateway ends up routing to an adapter that was retired three weeks ago, and how `--quantization fp8` ends up
# on a command serving weights that were never quantized.
#
# --dry-run prints the command without running it, which is what the tests check.
set -euo pipefail

CFG="${AGENTDISTILL_CONFIG:-project.yaml}"
AD="${AGENTDISTILL:-agentdistill}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

cfg() { "$AD" config get "$1" --config "$CFG" 2>/dev/null || echo "${2:-}"; }

BASE="$(cfg train.base_model)"
if [[ -z "$BASE" ]]; then
  echo "project.yaml has no train.base_model; nothing to serve" >&2
  exit 1
fi
PARSER="$(cfg train.tool_parser.name hermes)"
QUANT="$(cfg serve.quantization)"
MAXLEN="$(cfg serve.max_model_len 16384)"
PORT="$(cfg serve.vllm_port 8000)"

PROD="$("$AD" adapter path --status prod --config "$CFG" 2>/dev/null || true)"
CANARY="$("$AD" adapter path --status canary --config "$CFG" 2>/dev/null || true)"

ARGS=(serve "$BASE"
      # Agent prompts share a long prefix -- system prompt plus tool schemas, on every turn of every task.
      # Prefix caching is the single largest throughput win available here.
      --enable-prefix-caching
      --max-model-len "$MAXLEN"
      --max-num-seqs 64
      --enable-auto-tool-choice
      --tool-call-parser "$PARSER"
      --port "$PORT")

LORA_ARGS=()
[[ -n "$PROD"   ]] && LORA_ARGS+=(--lora-modules "prod=$PROD")
# The canary is loaded alongside prod, not instead of it: the split sends a share of student traffic to each,
# and both must be resident for that to be a live comparison rather than a restart.
[[ -n "$CANARY" ]] && LORA_ARGS+=(--lora-modules "canary=$CANARY")
if [[ ${#LORA_ARGS[@]} -gt 0 ]]; then
  ARGS+=(--enable-lora --max-loras 4 --max-lora-rank 64 "${LORA_ARGS[@]}")
else
  echo "note: no prod or canary adapter in the registry; serving the base model alone" >&2
fi

# fp8 is applied online from bf16 weights. AWQ weights carry their own config and must not be double-flagged.
[[ "$QUANT" == "fp8" ]] && ARGS+=(--quantization fp8)

echo "vllm ${ARGS[*]}"
[[ $DRY_RUN -eq 1 ]] && exit 0
exec vllm "${ARGS[@]}"
