#!/usr/bin/env bash
# A rented box, from a fresh clone to "pre-flight passes". Safe to run twice.
#
#   export ANTHROPIC_API_KEY=...        # you, in your own shell. Never in the repo, never in a prompt.
#   bash scripts/bootstrap_box.sh
#
# It ends with the pre-flight. If any line says FAIL, stop: debugging configuration on a rented meter is the
# most expensive way to find a missing YAML key.
set -euo pipefail
cd "$(dirname "$0")/.."

EX="examples/support_agent"
CFG="$EX/project.yaml"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
mkdir -p "$HF_HOME"

say() { echo; echo "== $*"; }

say "python packages"
# vLLM first: it pulls its own torch and replaces one installed before it.
if command -v uv >/dev/null 2>&1; then
  uv pip install --system -r requirements-gpu.txt
  uv pip install --system -e ".[train,serve]"
else
  python -m pip install -r requirements-gpu.txt
  python -m pip install -e ".[train,serve]"
fi

say "corpus"
bash scripts/make_corpus.sh

say "teacher pricing"
# The registry is gitignored, so a fresh box has no price on file and the cost block cannot be computed.
# Prices change: this records the one in force today, and the report cites the row it priced from.
TEACHER="$(agentdistill config get teacher.model --config "$CFG")"
agentdistill pricing set "$TEACHER" --provider "$(agentdistill config get teacher.provider --config "$CFG")" \
  --input "${TEACHER_INPUT_PER_MTOK:-5.0}" --output "${TEACHER_OUTPUT_PER_MTOK:-25.0}" \
  --cache-read "${TEACHER_CACHE_READ_PER_MTOK:-0.5}" --effective-from "$(date -u +%F)" --config "$CFG"

say "ingest, eval sets, curate"
( cd "$EX"
  agentdistill ingest jsonl traces-train.jsonl --config project.yaml >/dev/null
  for s in holdout unseen calib; do
    agentdistill evalset add "support-$s-v1" "eval-$s.jsonl" --config project.yaml >/dev/null 2>&1 \
      || echo "  support-$s-v1 already registered"
  done
  agentdistill curate --config project.yaml )

say "lock"
agentdistill ops lock check --config "$CFG"

say "pre-flight"
bash scripts/preflight.sh
