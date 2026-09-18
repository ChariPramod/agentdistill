#!/usr/bin/env bash
# The real rehearsal: destroy every tiny-mode artifact and rerun the GPU day from nothing, then assert the report.
#
#   bash scripts/clean_rehearsal.sh
#
# Green means a clean tiny run produced a report with base, student and teacher rows, populated calibration,
# cascade, cost and quantization sections, and no warnings except tiny-mode disclosures. At that point the GPU day
# is a rerun with different config values. Run it from a fresh clone before the day (docs/gpu-day.md, pre-flight):
# a clean working tree is not the same as a clean checkout.
#
# CLEAN_ALLOW_DIRTY=1 also accepts the dirty-tree warning, for iterating on uncommitted changes. Never on the day.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AGENTDISTILL_TINY:=1}"; export AGENTDISTILL_TINY

EX="examples/support_agent"
echo "== removing every tiny-mode artifact"
rm -rf artifacts/gpu_day logs/gpu_day.* artifacts/tiny \
       "$EX/.agentdistill/registry.tiny.db" "$EX/artifacts/tiny" "$EX/reports/tiny" \
       "$EX"/eval-*-tiny.jsonl
git clean -ndx artifacts "$EX/.agentdistill" "$EX/artifacts" 2>/dev/null | sed 's/^/would remove: /' || true

bash scripts/gpu_day.sh

allow=(--allow-warning tiny_mode --allow-warning replay_teacher --allow-warning uninformative)
if [[ "${CLEAN_ALLOW_DIRTY:-0}" == "1" ]]; then
  allow+=(--allow-warning dirty_tree)
fi

python -m agentdistill.tools.assert_report artifacts/gpu_day/report.html \
  --require-subjects base,student,teacher \
  --require-sections calibration,cascade,cost,quantization \
  --forbid-warning no_run_found --forbid-warning no_calibration \
  "${allow[@]}"
echo "clean rehearsal ok"
