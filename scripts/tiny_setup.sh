#!/usr/bin/env bash
# Prepare the CPU rehearsal: a tiny model, small deterministic eval sets, an ingested registry.
#
# Idempotent, so `AGENTDISTILL_TINY=1 bash scripts/gpu_day.sh` can call it every run.
#
# Ten tasks for holdout and unseen rather than the five the plan first specified: at five, `eval compare`
# has nothing to say. It still has little to say at ten -- below 20 tasks x 3 repeats compare returns an
# insufficient-power marker instead of a p-value -- but the comparison stage runs end to end and the report
# prints the reason, which is what a rehearsal is for. Calibration gets 20 tasks (plan 3.2), so that with
# `cascade.min_turns: 20` the calibrator is actually fit rather than refused.
set -euo pipefail
cd "$(dirname "$0")/.."

EX="examples/support_agent"
CFG="$EX/project.tiny.yaml"
AD="${AGENTDISTILL:-agentdistill}"
N_TASKS="${TINY_TASKS:-10}"
N_CALIB="${TINY_CALIB_TASKS:-20}"

echo "==> corpus"
# Gitignored and generated; a fresh clone has none until this runs.
bash scripts/make_corpus.sh

echo "==> tiny model"
python scripts/make_tiny_model.py --out artifacts/tiny/model

echo "==> tiny eval sets ($N_TASKS holdout/unseen tasks, $N_CALIB calibration tasks)"
# Ranked by a hash of (set name, scenario, instance), not taken from the front of the file: a clean rehearsal
# that reorders or regenerates a source file must still pick the same tasks, and each set's printed hash is how
# two runs prove they evaluated the same thing. The salt is the set name, so overlapping pools do not pick
# correlated tasks.
#
# eval-calib.jsonl has 102 distinct tasks, none shared with holdout, unseen or traces-train, so all 20
# calibration tasks come from it; nothing is borrowed from a set the student is scored or trained on.
#
# The sets are registered frozen below. An existing tiny registry keeps whatever it froze first -- that is the
# point of freezing -- so a changed selection only takes effect on a clean run (registry deleted).
for spec in "holdout:$N_TASKS" "unseen:$N_TASKS" "calib:$N_CALIB"; do
  s="${spec%%:*}"; n="${spec##*:}"
  python -m agentdistill.evalsets.generate --src "$EX/eval-$s.jsonl" --dst "$EX/eval-$s-tiny.jsonl" \
    --n "$n" --salt "support-$s-tiny" | sed 's/^/  /'
done

echo "==> registry"
"$AD" ingest jsonl "$EX/traces-train.jsonl" --config "$CFG" >/dev/null
for s in holdout unseen calib; do
  # Already-frozen eval sets are left alone; freezing is the point of an eval set.
  "$AD" evalset add "support-$s-tiny" "$EX/eval-$s-tiny.jsonl" --config "$CFG" >/dev/null 2>&1 \
    || echo "  support-$s-tiny already registered"
done

echo "==> curate"
if "$AD" dataset latest --kind sft --config "$CFG" >/dev/null 2>&1; then
  echo "  dataset already built"
else
  "$AD" curate --config "$CFG" >/dev/null
fi

echo "tiny setup ok — the numbers this produces mean nothing; the execution path is real"
