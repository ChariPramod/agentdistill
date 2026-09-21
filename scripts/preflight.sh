#!/usr/bin/env bash
# Everything that must be true before the meter starts. Run it on the box, before `gpu_day.sh`.
#
#   bash scripts/preflight.sh                 # the repo's example project
#   bash scripts/preflight.sh --bundle-ok     # the owner chose a git bundle over a remote
#   bash scripts/preflight.sh --config path/to/project.yaml
#
# One line per check: PASS, FAIL, or SKIP, cheapest checks first, hardware last. Exit 0 only when nothing
# FAILed. A SKIP is a check that cannot run here and says why; it never stands in for a pass.
#
# If a line says FAIL, stop and fix it here. The whole point is that no failure is discovered on rented
# hardware, after the stages before it have already run and been paid for.
#
# The git and environment groups are checked here because they are the cheapest; everything that needs the
# project loaded -- config, registry, lock, tokenizer guard, version pins, hardware -- is
# `agentdistill/ops/preflight.py`, called below, so the two halves keep one order between them.
set -uo pipefail   # deliberately no -e: a failing check reports and the run continues to the next one

ROOT="${PREFLIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || { echo "FAIL preflight: cannot cd to $ROOT"; exit 2; }

CFG="${AGENTDISTILL_CONFIG:-examples/support_agent/project.yaml}"
BUNDLE_OK=0
PY="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-ok) BUNDLE_OK=1 ;;
    --config)    CFG="${2:?--config needs a path}"; shift ;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
    *)           echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

LINES="$(mktemp)"
trap 'rm -f "$LINES"' EXIT
FAILED=0

pass() { printf 'PASS %s: %s\n' "$1" "$2" | tee -a "$LINES"; }
fail() { printf 'FAIL %s: %s\n' "$1" "$2" | tee -a "$LINES"; FAILED=1; }
skip() { printf 'SKIP %s: %s\n' "$1" "$2" | tee -a "$LINES"; }

# ---------------------------------------------------------------------------------------------------------
# git: the day's results are attributed to a commit, and the commit has to exist somewhere else too
# ---------------------------------------------------------------------------------------------------------
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  fail git.clean "$ROOT is not a git repository, so nothing the day produces can be attributed to a commit"
  fail git.pushed "$ROOT is not a git repository"
else
  DIRTY="$(git status --porcelain 2>/dev/null)"
  if [[ -z "$DIRTY" ]]; then
    pass git.clean "working tree clean at $(git rev-parse --short HEAD 2>/dev/null || echo '<no commit>')"
  else
    fail git.clean "$(printf '%s' "$DIRTY" | wc -l | tr -d ' ') uncommitted change(s): $(printf '%s' "$DIRTY" | head -3 | tr '\n' ';' ) — commit or stash them; every registry row records the commit it came from and a dirty tree makes that a lie"
  fi

  if [[ -z "$(git remote 2>/dev/null)" ]]; then
    if [[ "$BUNDLE_OK" == "1" ]]; then
      pass git.pushed "no remote, and --bundle-ok says the owner chose a bundle — confirm the bundle for $(git rev-parse --short HEAD 2>/dev/null) is already off this machine (git bundle create ~/agentdistill-\$(date +%Y%m%d).bundle --all)"
    else
      fail git.pushed "this repository has no remote, so HEAD exists on one disk only. Fix: create an empty private repo and \`git remote add origin <url> && git push -u origin --all && git push --tags\`; or, if the owner chose a bundle, \`git bundle create ~/agentdistill-\$(date +%Y%m%d).bundle --all\`, copy it off, and rerun with --bundle-ok"
    fi
  else
    HEAD_SHA="$(git rev-parse HEAD 2>/dev/null)"
    REMOTE_REF="$(git branch -r --contains "$HEAD_SHA" 2>/dev/null | head -1 | xargs || true)"
    if [[ -n "$REMOTE_REF" ]]; then
      pass git.pushed "HEAD is on $REMOTE_REF"
    else
      fail git.pushed "HEAD $(git rev-parse --short HEAD) is on no remote branch; \`git push -u origin HEAD\` (and \`git push --tags\`) before the box is rented"
    fi
  fi
fi

# ---------------------------------------------------------------------------------------------------------
# environment: the key and the model cache
# ---------------------------------------------------------------------------------------------------------
# Only whether it is set. Never its length, never a prefix, never a suffix: a redaction that leaks four
# characters is still a leak, and this output gets pasted into chat logs.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  pass env.api_key "set (the value is never printed)"
else
  fail env.api_key "ANTHROPIC_API_KEY is not set in this shell. Export it yourself on the box (never in a file, never in the repo), with a hard spend cap already set on its workspace"
fi

HF="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF" 2>/dev/null
if [[ -d "$HF" ]] && touch "$HF/.preflight" 2>/dev/null; then
  rm -f "$HF/.preflight"
  pass env.hf_home "$HF is writable"
else
  fail env.hf_home "$HF is not writable; a rented box often points the default cache at a scratch mount, and the base model is then re-downloaded between stages, on the clock"
fi

# ---------------------------------------------------------------------------------------------------------
# config, registry, lock, tokenizer guard, version pins, hardware
# ---------------------------------------------------------------------------------------------------------
if ! command -v "$PY" >/dev/null 2>&1; then
  fail python "no \`$PY\` on PATH, so none of the project checks can run (set PYTHON=<interpreter>)"
else
  "$PY" -m agentdistill.ops.preflight --config "$CFG" 2>&1 | tee -a "$LINES"
  [[ "${PIPESTATUS[0]}" != "0" ]] && FAILED=1
fi

N_PASS="$(grep -c '^PASS ' "$LINES" || true)"
N_FAIL="$(grep -c '^FAIL ' "$LINES" || true)"
N_SKIP="$(grep -c '^SKIP ' "$LINES" || true)"
echo
echo "preflight: ${N_PASS} PASS, ${N_FAIL} FAIL, ${N_SKIP} SKIP  (config: $CFG)"
if [[ "$FAILED" != "0" ]]; then
  echo "preflight FAILED — do not start the GPU day. Hand this whole output to the lead."
  exit 1
fi
echo "preflight ok"
