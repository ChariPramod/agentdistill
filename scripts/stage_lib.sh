# Sourced by gpu_day.sh. Kept apart so the resume and exit-code rules can be tested without running a GPU day.
#
# A stage writes $MARKERS/<name>.done only when it exits 0. That includes a declared skip, which the command
# itself logs as `[stage <name>] SKIPPED: <config value>`. Exit 3 means the stage ran and wrote no row; it gets
# no marker and stops the script, so the rerun retries it instead of skipping past a hole.
#
# Expects: MARKERS, DRY. The stage name is exported as AGENTDISTILL_STAGE so the CLI's log line names the stage.

stage() {
  local name="$1"; shift
  if [[ -f "$MARKERS/$name.done" ]]; then echo "== skip $name (done)"; return 0; fi
  echo "== $name  $(date -u +%H:%M:%S)"
  local status=0
  export AGENTDISTILL_STAGE="$name"
  # The stage runs in its own subshell with errexit on. `"$@" || status=$?` would be shorter and wrong: bash
  # ignores `set -e` inside anything on the left of `||`, so a failing first command in a multi-command stage
  # would fall through to the next one instead of failing the stage.
  set +e
  if [[ "$DRY" == "1" ]]; then
    (set -e; "$@")
    status=$?
  else
    mkdir -p logs
    (set -e; "$@") 2>&1 | tee "logs/gpu_day.$name.log"
    status=${PIPESTATUS[0]}
  fi
  set -e
  unset AGENTDISTILL_STAGE
  if [[ "$status" == "3" ]]; then
    echo "== $name WROTE NOTHING (exit 3): no marker written; fix the cause and rerun to retry this stage" >&2
    exit 3
  elif [[ "$status" != "0" ]]; then
    echo "== $name FAILED (exit $status): no marker written" >&2
    exit "$status"
  fi
  touch "$MARKERS/$name.done"
}
