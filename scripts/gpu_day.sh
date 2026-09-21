#!/usr/bin/env bash
# One GPU session, end to end. Resumable: each stage writes artifacts/gpu_day/<stage>.done and is skipped on
# rerun, so a failed stage costs one stage rather than the session.
#
#   bash scripts/gpu_day.sh                  # run everything not yet done
#   AGENTDISTILL_DRY_RUN=1 bash scripts/...  # print the plan, run nothing
#   AGENTDISTILL_TINY=1 bash scripts/...     # the CPU rehearsal: same path, garbage numbers
#   rm artifacts/gpu_day/sft.done            # force one stage to run again
#
# Every command here exists and is tested on CPU. The day should be reading a log, not writing code.
set -euo pipefail
cd "$(dirname "$0")/.."

# A fixed model cache inside the repo. Rented boxes often point the default cache at a scratch or tmpfs
# directory, and a clean run then re-downloads the base model -- silently, on the clock -- or loses it between
# stages. An explicit HF_HOME from the environment still wins.
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Tiny mode runs every stage on a laptop with the fixture tokenizer, five tasks and N=1. The numbers are
# garbage; the execution path is real. Run it before the GPU day, because the failures it finds -- a query that
# assumed a vLLM-only field, a path that only exists after quantization -- cost minutes here and an hour there.
if [[ "${AGENTDISTILL_TINY:-0}" == "1" ]]; then
  export AGENTDISTILL_CONFIG="${AGENTDISTILL_CONFIG:-examples/support_agent/project.tiny.yaml}"
  export AGENTDISTILL_EVAL_BACKEND="${AGENTDISTILL_EVAL_BACKEND:-hf}"
  export AGENTDISTILL_FAKE_VLLM=1
  EVAL_SET="${EVAL_SET:-support-holdout-tiny}"
  UNSEEN_SET="${UNSEEN_SET:-support-unseen-tiny}"
  CALIB_SET="${CALIB_SET:-support-calib-tiny}"
  N_EVAL="${N_EVAL:-1}"
  # The tiny config's teacher is the replay stub (teacher.backend: replay): it answers with the recorded turns at
  # fixed token counts, so the teacher row, the cost block and the cascade all execute without a key. Its run is
  # tagged teacher_backend=replay and the report discloses that its costs are structural.
  TEACHER_SUBJECT="${TEACHER_SUBJECT:-teacher}"
  SAMPLES="${SAMPLES:-1}"
  echo "== tiny mode: CPU rehearsal, the numbers are not meaningful"
  # Idempotent: builds the tiny model and the ten-task eval sets if they are not already there. Skipped in a
  # dry run, which is meant to print a plan without touching anything.
  if [[ "${AGENTDISTILL_DRY_RUN:-0}" == "1" ]]; then
    echo "bash scripts/tiny_setup.sh"
  else
    bash scripts/tiny_setup.sh
  fi
fi

export AGENTDISTILL_CONFIG="${AGENTDISTILL_CONFIG:-examples/support_agent/project.yaml}"
CONFIG_ARG=(--config "$AGENTDISTILL_CONFIG")
TAG="${TAG:-gpu-day}"
DRY="${AGENTDISTILL_DRY_RUN:-0}"
EVAL_SET="${EVAL_SET:-support-holdout-v1}"
UNSEEN_SET="${UNSEEN_SET:-support-unseen-v1}"
CALIB_SET="${CALIB_SET:-support-calib-v1}"
N_EVAL="${N_EVAL:-5}"
TEACHER_SUBJECT="${TEACHER_SUBJECT:-teacher}"
SAMPLES="${SAMPLES:-3}"
# Student evals run lockstep at this batch size, so throughput -- and the cost per token built on it -- is a
# serving figure. Only vLLM batches; tiny mode's hf client falls back and the report says unbatched.
BATCH="${BATCH:-32}"
BACKEND="${AGENTDISTILL_EVAL_BACKEND:-vllm}"
TINY="${AGENTDISTILL_TINY:-0}"

# Overridable so the test suite can exercise the script without deleting a real session's markers. A test that
# removes artifacts/gpu_day while a GPU day is running would make the next stage redo work that was paid for.
MARKERS="${AGENTDISTILL_MARKER_DIR:-artifacts/gpu_day}"
mkdir -p "$MARKERS" logs

# In a dry run every agentdistill call is echoed instead of executed, which is what the syntax test exercises.
ad() { if [[ "$DRY" == "1" ]]; then echo "agentdistill $*"; else agentdistill "$@"; fi; }
cap() { if [[ "$DRY" == "1" ]]; then echo "<$1>"; else shift; agentdistill "$@"; fi; }

# Markers only on exit 0; exit 3 (a stage that wrote nothing) stops the day without one. See stage_lib.sh.
# shellcheck source=scripts/stage_lib.sh
source "$(dirname "$0")/stage_lib.sh"

s_export() { bash scripts/export_results.sh; }

# The export also runs when a stage fails, exits 3, or the session is interrupted: a day that died at `calibrate`
# still produced eleven stages of rows and logs, and the box is about to be deleted. Guarded so the normal path
# does not export twice.
on_exit() {
  local status=$?
  # The export stage writes its marker on success, so the trap only fires when the day ended some other way:
  # a stage failed, a stage wrote no row and exited 3, or someone interrupted it.
  if [[ "$DRY" == "1" || -f "$MARKERS/export.done" ]]; then exit "$status"; fi
  echo "== export (from the exit trap, status $status)"
  bash scripts/export_results.sh || echo "export failed; the registry is still under .agentdistill/" >&2
  exit "$status"
}
trap on_exit EXIT

timing_table() {
  local file="$MARKERS/stage_seconds.tsv"
  [[ -f "$file" ]] || return 0
  local scale
  scale="$(cap timing_scale config get gpu_day.timing_scale "${CONFIG_ARG[@]}" 2>/dev/null || echo 1)"
  echo "== stage timings (expected = rehearsal x ${scale})"
  awk -v scale="$scale" -v timings="scripts/rehearsal_timings.tsv" '
    BEGIN { while ((getline line < timings) > 0) { if (line !~ /^#/ && line != "") { split(line, f, "\t"); expected[f[1]] = f[2] } } }
    { want = (expected[$1] == "" ? 0 : expected[$1] * scale)
      flag = (want > 0 && $2 > 3 * want) ? "  <- OVER 3x EXPECTED" : ""
      printf "  %-18s %6ss   expected %6ss%s\n", $1, $2, (want > 0 ? sprintf("%.0f", want) : "?"), flag }
  ' "$file"
}

BASE_MODEL="$(cap base_model config get train.base_model "${CONFIG_ARG[@]}")"
QUANT="$(cap quantization config get serve.quantization "${CONFIG_ARG[@]}")"

# Install through whichever tool manages this environment. A uv-created venv has no `pip` in it at all, which
# is how the first rehearsal died on its first stage.
install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  elif python -m pip --version >/dev/null 2>&1; then
    python -m pip install "$@"
  elif command -v pip >/dev/null 2>&1; then
    pip install "$@"
  else
    echo "no uv and no pip in this environment; install one before running the day" >&2
    return 1
  fi
}

# Version pins, so the box installs what was rehearsed rather than whatever released this morning.
requirements_arg() {
  if [[ -f requirements-gpu.txt ]]; then echo "-r requirements-gpu.txt"; fi
}

s_env() {
  if [[ "$DRY" == "1" ]]; then
    echo 'install -e ".[train,serve]" && python -c "check torch/vllm/trl/peft"'
    return 0
  fi
  if [[ "$TINY" == "1" ]]; then
    # No vLLM and no CUDA on a laptop, and asking for them would fail the rehearsal on the one thing it is not
    # rehearsing. The environment is assumed already installed here; a rehearsal that reinstalled the
    # development venv on every run would be a worse experience than the failure it is protecting against.
    python -c 'import torch, trl, peft; print("torch", torch.__version__, "trl/peft ok (tiny mode)")'
    return 0
  fi
  # shellcheck disable=SC2046
  install $(requirements_arg) -e ".[train,serve]"
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
s_merge()       { ad adapter merge "$(cap adapter adapter latest --tag "$TAG" "${CONFIG_ARG[@]}")" --backend "$BACKEND" "${CONFIG_ARG[@]}"; }
# fp8 writes a marker and nothing else, so quantize runs unchanged on a laptop. AWQ does not, and tiny mode
# configures fp8 rather than skipping the stage -- a skipped stage rehearses nothing.

s_eval_base()   { ad eval run base --eval-set "$EVAL_SET" --n "$N_EVAL" --policy strict --backend "$BACKEND" --batch "$BATCH" --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_eval_sft()    { ad eval run "$(cap adapter adapter latest --tag "$TAG" "${CONFIG_ARG[@]}")" --eval-set "$EVAL_SET" --n "$N_EVAL" --policy strict --backend "$BACKEND" --batch "$BATCH" --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_eval_teach()  { ad eval run "$TEACHER_SUBJECT" --eval-set "$EVAL_SET" --n "$N_EVAL" --tools live --tag "$TAG" "${CONFIG_ARG[@]}"; }
# The same teacher, the same tasks, graded by replay. Not a second baseline: the gap between this and the live
# run above is the measured distortion of replay grading, which the report prints as its own section. Without it
# the report would have to claim replay's penalty is small rather than show it.
s_eval_teach_replay() { ad eval run "$TEACHER_SUBJECT" --eval-set "$EVAL_SET" --n "$N_EVAL" --tools replay --policy strict --tag "$TAG-replay" "${CONFIG_ARG[@]}"; }
# Both sides by subject. `eval latest --tag` returned whichever tagged run was newest -- the teacher's, once the
# teacher row existed -- and the stage compared the teacher against base under the name cmp_sft.
s_cmp_sft()     { ad eval compare "$(cap ev eval latest --subject "$(cap adapter adapter latest --tag "$TAG" "${CONFIG_ARG[@]}")" --eval-set "$EVAL_SET" "${CONFIG_ARG[@]}")" "$(cap ev eval latest --subject base --eval-set "$EVAL_SET" "${CONFIG_ARG[@]}")" --out "$MARKERS/cmp_sft.md" "${CONFIG_ARG[@]}"; }

s_onpolicy()    { ad train onpolicy "$(cap adapter adapter latest --tag "$TAG" "${CONFIG_ARG[@]}")" --rounds 1 --backend "$BACKEND" --tag "$TAG-r1" "${CONFIG_ARG[@]}"; }
s_eval_r1() {
  # A discarded on-policy round is a result, recorded on its round row with the reason; it just leaves no
  # candidate to evaluate. That is the one case this stage skips, and the skip cites the row that decided it.
  local r1="" round=""
  if [[ "$DRY" != "1" ]]; then r1="$(agentdistill adapter latest --tag "$TAG-r1" "${CONFIG_ARG[@]}" 2>/dev/null || true)"; fi
  if [[ "$DRY" != "1" && -z "$r1" ]]; then
    round="$(agentdistill train latest-round --tag "$TAG-r1" "${CONFIG_ARG[@]}")"
    echo "[stage eval_r1] SKIPPED: round ${round%% *} decided ${round#* } -- no candidate to evaluate"
    return 0
  fi
  ad eval run "$(cap adapter adapter latest --tag "$TAG-r1" "${CONFIG_ARG[@]}")" --eval-set "$EVAL_SET" --n "$N_EVAL" --policy strict --backend "$BACKEND" --batch "$BATCH" --tag "$TAG-r1" "${CONFIG_ARG[@]}"
}
s_unseen()      { ad eval run "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --eval-set "$UNSEEN_SET" --n "$N_EVAL" --policy strict --backend "$BACKEND" --batch "$BATCH" --tag "$TAG" "${CONFIG_ARG[@]}"; }

s_logprobs()    { ad eval run "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --eval-set "$CALIB_SET" --n 3 --policy fuzzy --backend "$BACKEND" --logprobs --samples "$SAMPLES" --tag "$TAG-calib" "${CONFIG_ARG[@]}"; }
s_calibrate()   { ad calibrate "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --from-eval "$(cap ev eval latest --eval-set "$CALIB_SET" "${CONFIG_ARG[@]}")" "${CONFIG_ARG[@]}"; }
s_cascade_ver() {
  # Tiny mode escalates to the replay stub rather than skipping: a skipped stage rehearses nothing, and this is
  # the stage that writes the verified cascade point the report's cost block reads.
  ad eval run "cascade:$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}"):auto" --eval-set "$EVAL_SET" --n 3 --policy fuzzy --backend "$BACKEND" --verify-threshold --tag "$TAG" "${CONFIG_ARG[@]}"
}

s_quantize()    { ad adapter quantize "$(cap adapter adapter best --tag "$TAG*" "${CONFIG_ARG[@]}")" --method "$QUANT" "${CONFIG_ARG[@]}"; }
s_eval_quant()  { ad eval run "$(cap adapter adapter latest --quantized "${CONFIG_ARG[@]}")" --eval-set "$EVAL_SET" --n 3 --policy strict --backend "$BACKEND" --batch "$BATCH" --tag "$TAG" "${CONFIG_ARG[@]}"; }
s_serve_smoke() {
  if [[ "$DRY" == "1" ]]; then echo "bash scripts/serve_smoke.sh"; return 0; fi
  if [[ "$TINY" == "1" ]]; then
    # The test suite's fake vLLM stands in for the real one, so the gateway, both dialects and the request-log
    # assertion are all still exercised. What is not exercised is vLLM itself, which is the whole point of
    # running the real smoke test on the real box afterwards.
    AGENTDISTILL_FAKE_VLLM=1 bash scripts/serve_smoke.sh
    return $?
  fi
  bash scripts/serve_smoke.sh
}
# The same tag the calibrate and quantize stages select with, so the report's student is the adapter they used.
s_report()      { ad report --out "$MARKERS/report.html" --tag "$TAG*" --include-run-ids "${CONFIG_ARG[@]}"; }

stage env          s_env
stage base_check   s_base_check
stage sft          s_sft
stage merge        s_merge
stage eval_base    s_eval_base
stage eval_sft     s_eval_sft
stage eval_teach   s_eval_teach
stage eval_teach_replay s_eval_teach_replay
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

stage export       s_export

echo "== done  $(date -u +%H:%M:%S)"
timing_table
ls -la "$MARKERS"
