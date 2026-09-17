#!/usr/bin/env bash
# Prepare the CPU rehearsal: a tiny model, ten-task eval sets, an ingested registry.
#
# Idempotent, so `AGENTDISTILL_TINY=1 bash scripts/gpu_day.sh` can call it every run.
#
# Ten tasks rather than the five the plan specified: `eval compare` refuses below eight, because a
# task-clustered interval over five tasks means nothing. At five, the rehearsal would silently skip the
# comparison stage -- which is one of the stages most worth rehearsing.
set -euo pipefail
cd "$(dirname "$0")/.."

EX="examples/support_agent"
CFG="$EX/project.tiny.yaml"
AD="${AGENTDISTILL:-agentdistill}"
N_TASKS="${TINY_TASKS:-10}"

echo "==> tiny model"
python scripts/make_tiny_model.py --out artifacts/tiny/model

echo "==> tiny eval sets ($N_TASKS tasks each)"
python - "$EX" "$N_TASKS" <<'PY'
import json, pathlib, sys

base, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
for src, dst in (("eval-holdout.jsonl", "eval-holdout-tiny.jsonl"),
                 ("eval-unseen.jsonl", "eval-unseen-tiny.jsonl"),
                 ("eval-calib.jsonl", "eval-calib-tiny.jsonl")):
    rows = [json.loads(l) for l in (base / src).read_text().splitlines() if l.strip()]
    # Taken from the front rather than sampled, so two rehearsals compare like with like.
    seen, picked = set(), []
    for r in rows:
        tid = r.get("task_id") or r["id"]
        if tid in seen:
            continue
        seen.add(tid)
        picked.append(r)
        if len(picked) == n:
            break
    (base / dst).write_text("\n".join(json.dumps(r) for r in picked) + "\n")
    print(f"  {dst}: {len(picked)} tasks")
PY

echo "==> registry"
"$AD" ingest jsonl "$EX/traces-train.jsonl" --config "$CFG" >/dev/null
for s in holdout unseen calib; do
  # Already-frozen eval sets are left alone; freezing is the point of an eval set.
  "$AD" evalset add "support-$s-tiny" "$EX/eval-$s-tiny.jsonl" --config "$CFG" >/dev/null 2>&1 \
    || echo "  support-$s-tiny already registered"
done

echo "tiny setup ok — the numbers this produces mean nothing; the execution path is real"
