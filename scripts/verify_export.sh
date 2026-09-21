#!/usr/bin/env bash
# Run this ON THE LAPTOP, on the tarball copied off the box, BEFORE the box is terminated.
#
#   bash scripts/verify_export.sh ~/Downloads/agentdistill-export-20260921T101500Z.tar.gz
#
# It recomputes the checksum, lists what is inside, opens the exported registry, and checks that every run id in
# report.json is in it. Nothing is deleted or terminated until this prints "export verified".
#
# The work is `agentdistill/ops/verify_export.py`, standard library only, so this runs on a laptop with nothing
# installed but Python.
set -uo pipefail

TARBALL="${1:-}"
if [[ -z "$TARBALL" ]]; then
  echo "usage: bash scripts/verify_export.sh <tarball> [checksum-file]" >&2
  exit 2
fi
shift
CHECKSUM="${1:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "no \`$PY\` on PATH (set PYTHON=<interpreter>)" >&2; exit 2; }

args=("$TARBALL")
[[ -n "$CHECKSUM" ]] && args+=(--checksum "$CHECKSUM")

cd "$ROOT" && PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m agentdistill.ops.verify_export "${args[@]}"
