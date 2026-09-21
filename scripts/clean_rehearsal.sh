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

# Every code in agentdistill.report.assemble.WARNING_CODES is placed in exactly one list; a test holds this file
# to that, because a code in neither would pass silently.
#
# Allowed: the disclosures tiny mode is expected to make. A random model's calibration labels are all "bad"
# (gate_degenerate), its teacher is the replay stub, and the hf client does not batch (cost_unbatched). On the GPU
# day every one of these except tiny_mode's absence is forbidden -- see docs/gpu-day.md.
# corpus_teacher_differs is a true and permanent disclosure for this example: the corpus was recorded from a
# scripted solver, so the student imitates that solver and the teacher comparison is operational rather than
# distillation. It stays allowed until the corpus is re-recorded from the serving teacher.
ALLOW=(tiny_mode replay_teacher gate_degenerate cost_unbatched corpus_teacher_differs)
# Forbidden: a hole in the pipeline, which this rehearsal exists to prove there is none of.
FORBID=(no_eval_set no_run_found teacher_skipped no_student paired_failed no_calibration gate_not_usable
        cascade_unverified quantized_unevaluated quantization_missing no_teacher_run no_teacher_config no_pricing
        no_prompt_tokens no_throughput dirty_tree eval_mode_mismatch)
if [[ "${CLEAN_ALLOW_DIRTY:-0}" == "1" ]]; then
  ALLOW+=(dirty_tree)
  FORBID=("${FORBID[@]/dirty_tree}")
fi

args=()
for code in "${ALLOW[@]}"; do args+=(--allow-warning "$code"); done
for code in "${FORBID[@]}"; do [[ -n "$code" ]] && args+=(--forbid-warning "$code"); done

python -m agentdistill.tools.assert_report artifacts/gpu_day/report.html \
  --require-subjects base,student,teacher \
  --require-sections calibration,cascade,cost,quantization,onpolicy \
  "${args[@]}"
echo "clean rehearsal ok"
