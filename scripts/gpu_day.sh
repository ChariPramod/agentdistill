#!/usr/bin/env bash
# One GPU session, end to end. Resumable: each stage writes artifacts/gpu_day/<stage>.done and is skipped on
# rerun, so a failed stage costs one stage rather than the session.
#
#   bash scripts/gpu_day.sh                  # run everything not yet done
#   AGENTDISTILL_DRY_RUN=1 bash scripts/...  # print the plan, run nothing
#   rm artifacts/gpu_day/sft.done            # force one stage to run again
#
# Every command here exists and is tested on CPU. The day should be reading a log, not writing code.
set -euo pipefail
cd "$(dirname "$0")/.."

export AGENTDISTILL_CONFIG="${AGENTDISTILL_CONFIG:-examples/support_agent/project.yaml}"
CONFIG_ARG=(--config "$AGENTDISTILL_CONFIG")
TAG="${TAG:-gpu-day}"
DRY="${AGENTDISTILL_DRY_RUN:-0}"
EVAL_SET="${EVAL_SET:-support-holdout-v1}"
UNSEEN_SET="${UNSEEN_SET:-support-unseen-v1}"
CALIB_SET="${CALIB_SET:-support-calib-v1}"
N_EVAL="${N_EVAL:-5}"

mkdir -p artifacts/gpu_day logs

# In a dry run every agentdistill call is echoed instead of executed, which is what the syntax test exercises.
ad() { if [[ "$DRY" == "1" ]]; then echo "agentdistill $*"; else agentdistill "$@"; fi; }
cap() { if [[ "$DRY" == "1" ]]; then echo "<$1>"; else shift; agentdistill "$@"; fi; }

stage() {
  local name="$1"; shift
  if [[ -f "artifacts/gpu_day/$name.done" ]]; then echo "== skip $name (done)"; return 0; fi
  echo "== $name  $(date -u +%H:%M:%S)"
  if [[ "$DRY" == "1" ]]; then "$@"; else "$@" 2>&1 | tee "logs/gpu_day.$name.log"; fi
  touch "artifacts/gpu_day/$name.done"
}

BASE_MODEL="$(cap base_model config get train.base_model "${CONFIG_ARG[@]}")"
QUANT="$(cap quantization config get serve.quantization "${CONFIG_ARG[@]}")"

s_env() {
  if [[ "$DRY" == "1" ]]; then
    echo 'pip install -e ".[train,serve]" && python -c "check torch/vllm/trl/peft"'
    return 0
  fi
  pip install -e ".[train,serve]"
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import trl, peft  # noqa: F401
try:
    import vllm  # noqa: F401
    print("vllm ok")
except ImportError:
    print("vllm MISSING: eval will fall back to the hf backend and take much longer")
PY
}

s_base_check()  { ad base-check "$BASE_MODEL" "${CONFIG_ARG[@]}"; }
s_sft()         { ad train sft "$(cap dataset dataset latest --kind sft "${CONFIG_ARG[@]}")" --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_merge()       { ad adapter merge "$(cap adapter adapter latest --tag "$TAG" "${CONFIG_ARG[@]}")" "${CONFIG_ARG[@]}"; }

s_eval_base()   { ad eval run base --eval-set "$EVAL_SET" --n "$N_EVAL" --policy strict --backend vllm --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_eval_sft()    { ad eval run "$(cap adapter adapter latest --tag "$TAG" "${CONFIG_ARG[@]}")" --eval-set "$EVAL_SET" --n "$N_EVAL" --policy strict --backend vllm --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_eval_teach()  { ad eval run teacher --eval-set "$EVAL_SET" --n "$N_EVAL" --policy strict --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_cmp_sft()     { ad eval compare "$(cap ev eval latest --subject-tag "$TAG" "${CONFIG_ARG[@]}")" "$(cap ev eval latest --subject base "${CONFIG_ARG[@]}")" --out artifacts/gpu_day/cmp_sft.md "${CONFIG_ARG[@]}"; }

s_onpolicy()    { ad train onpolicy "$(cap adapter adapter latest --tag "$TAG" "${CONFIG_ARG[@]}")" --rounds 1 --tag "$TAG-r1" "${CONFIG_ARG[@]}"; }
s_eval_r1()     { ad eval run "$(cap adapter adapter latest --tag "$TAG-r1" "${CONFIG_ARG[@]}")" --eval-set "$EVAL_SET" --n "$N_EVAL" --policy strict --backend vllm --tag "$TAG-r1" "${CONFIG_ARG[@]}"; }
s_unseen()      { ad eval run "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --eval-set "$UNSEEN_SET" --n "$N_EVAL" --policy strict --backend vllm --tag "$TAG" "${CONFIG_ARG[@]}"; }

s_logprobs()    { ad eval run "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --eval-set "$CALIB_SET" --n 3 --policy fuzzy --backend vllm --logprobs --samples 3 --tag "$TAG-calib" "${CONFIG_ARG[@]}"; }
s_calibrate()   { ad calibrate "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --from-eval "$(cap ev eval latest --eval-set "$CALIB_SET" "${CONFIG_ARG[@]}")" "${CONFIG_ARG[@]}"; }
s_cascade_ver() { ad eval run "cascade:$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}"):auto" --eval-set "$EVAL_SET" --n 3 --policy fuzzy --backend vllm --verify-threshold --tag "$TAG" "${CONFIG_ARG[@]}"; }

s_quantize()    { ad adapter quantize "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --method "$QUANT" "${CONFIG_ARG[@]}"; }
s_eval_quant()  { ad eval run "$(cap adapter adapter latest --quantized "${CONFIG_ARG[@]}")" --eval-set "$EVAL_SET" --n 3 --policy strict --backend vllm --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_serve_smoke() { if [[ "$DRY" == "1" ]]; then echo "bash scripts/serve_smoke.sh"; else bash scripts/serve_smoke.sh; fi; }
s_report()      { ad report --out artifacts/gpu_day/report.html --include-run-ids "${CONFIG_ARG[@]}"; }

stage env          s_env
stage base_check   s_base_check
stage sft          s_sft
stage merge        s_merge
stage eval_base    s_eval_base
stage eval_sft     s_eval_sft
stage eval_teach   s_eval_teach
stage cmp_sft      s_cmp_sft
stage onpolicy     s_onpolicy
stage eval_r1      s_eval_r1
stage unseen       s_unseen
stage logprobs     s_logprobs
stage calibrate    s_calibrate
stage cascade_ver  s_cascade_ver
stage quantize     s_quantize
stage eval_quant   s_eval_quant
stage serve_smoke  s_serve_smoke
stage report       s_report

echo "== done  $(date -u +%H:%M:%S)"
ls -la artifacts/gpu_day
