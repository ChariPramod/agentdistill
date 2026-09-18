#!/usr/bin/env bash
# Build the example corpus from committed code: record with the scripted teacher, then split.
#
# The corpus files are gitignored -- they are generated, and a scripted teacher's traces must never be mistaken
# for a teacher's -- so this recipe is what a fresh clone rebuilds them from. It is deterministic: the same commit
# produces byte-identical files, which is what lets a clean rehearsal on a fresh clone reproduce one on a laptop.
#
# The first corpus was recorded by hand with a command nobody wrote down, from an earlier scripted teacher, and
# 189 of its 800 traces could not be reproduced from any commit. This script exists so that cannot recur.
#
#   bash scripts/make_corpus.sh          # rebuild only if traces-train.jsonl is missing
#   FORCE=1 bash scripts/make_corpus.sh  # rebuild regardless
set -euo pipefail
cd "$(dirname "$0")/.."

EX="examples/support_agent"
if [[ "${FORCE:-0}" != "1" && -f "$EX/traces-train.jsonl" && -f "$EX/eval-holdout.jsonl" ]]; then
  echo "corpus present; FORCE=1 to rebuild"
  exit 0
fi

python -m examples.support_agent.record --scripted --error-rate 0.25 --n 800 --seed 7 --out "$EX/traces.jsonl"
python "$EX/split_corpus.py" --traces "$EX/traces.jsonl"
