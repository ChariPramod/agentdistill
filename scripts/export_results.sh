#!/usr/bin/env bash
# Everything the GPU day produced, in one tarball you can carry off the box.
#
#   bash scripts/export_results.sh                 # writes into artifacts/gpu_day
#   bash scripts/export_results.sh --out /tmp/x    # somewhere else
#
# `gpu_day.sh` runs this as its last stage AND from a trap on EXIT, so it also runs when a stage fails, when a
# stage writes no row and exits 3, and when the session is interrupted. A day that died at `calibrate` still
# produced eleven stages of registry rows and logs, and those are worth more than the box they are sitting on.
#
# What goes in, and what deliberately does not:
#
#   registry/      the registry database -- every run id, every metric, the whole provenance chain
#   report/        report.json, report.html and the comparison markdown, plus the stage markers
#   logs/          every logs/*.log, which is where a failure is diagnosed afterwards
#   adapters/      LoRA adapter directories only, found by their adapter_config.json. Merged and quantized full
#                  weights are tens of gigabytes and are reproducible from the adapter and the pinned base
#                  model; the adapter is the only thing here that is not.
#   env/           pip freeze, git HEAD and status, nvidia-smi if there is one
#   config/        the project configs, the lock file and requirements-gpu.txt
#
# The registry is gitignored and the box is disposable. If the results stay on a machine that is about to be
# deleted, the day produced nothing.
set -uo pipefail

ROOT="${EXPORT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || { echo "export: cannot cd to $ROOT" >&2; exit 2; }

CFG="${AGENTDISTILL_CONFIG:-examples/support_agent/project.yaml}"
MARKERS="${AGENTDISTILL_MARKER_DIR:-artifacts/gpu_day}"
OUT="${AGENTDISTILL_EXPORT_DIR:-$MARKERS}"
AD="${AGENTDISTILL:-agentdistill}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="${2:?--out needs a directory}"; shift ;;
    --config) CFG="${2:?--config needs a path}"; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum  >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else echo "no sha256sum and no shasum on this machine" >&2; return 1; fi
}

# The registry path, from the CLI when it is installed and from the config file when it is not -- the export
# has to work in a shell where the package is half-installed, because that is one of the ways a day fails.
registry_file() {
  local url="" raw=""
  if command -v "$AD" >/dev/null 2>&1; then
    url="$("$AD" config get registry --config "$CFG" 2>/dev/null || true)"
  fi
  if [[ -z "$url" ]]; then
    url="$(sed -n 's/^registry:[[:space:]]*//p' "$CFG" 2>/dev/null | head -1 | tr -d '"'"'"' ' )"
  fi
  [[ "$url" == sqlite:///* ]] || { echo ""; return 0; }
  raw="${url#sqlite:///}"
  if [[ "$raw" == /* ]]; then echo "$raw"; else echo "$(cd "$(dirname "$CFG")" && pwd)/$raw"; fi
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="agentdistill-export-${STAMP}"
STAGE="$(mktemp -d)/$NAME"
mkdir -p "$STAGE"/{registry,report,logs,adapters,env,config} "$OUT"

# --- registry ---------------------------------------------------------------------------------------------
REG="$(registry_file)"
if [[ -n "$REG" && -f "$REG" ]]; then
  cp "$REG" "$STAGE/registry/$(basename "$REG")"
  # A database checkpointed mid-write leaves its tail in the -wal; copying only the .db silently loses the
  # most recent rows, which are the ones the day just produced.
  for side in "-wal" "-shm"; do
    [[ -f "${REG}${side}" ]] && cp "${REG}${side}" "$STAGE/registry/$(basename "${REG}${side}")"
  done
  echo "export: registry $(basename "$REG")"
else
  echo "export: WARNING no registry at '${REG:-<unresolved>}' — the export will carry no run ids" >&2
  echo "no registry found for config $CFG" > "$STAGE/registry/MISSING.txt"
fi

# --- report and stage markers ------------------------------------------------------------------------------
for f in "$MARKERS"/report.json "$MARKERS"/report.html "$MARKERS"/*.md "$MARKERS"/*.done "$MARKERS"/*.seconds; do
  [[ -e "$f" ]] && cp "$f" "$STAGE/report/" 2>/dev/null
done
[[ -d reports ]] && cp -R reports "$STAGE/report/reports" 2>/dev/null

# --- logs ---------------------------------------------------------------------------------------------------
if compgen -G "logs/*.log" >/dev/null; then cp logs/*.log "$STAGE/logs/" 2>/dev/null; fi

# --- adapters (LoRA only) ------------------------------------------------------------------------------------
ART="$(sed -n 's/^artifacts:[[:space:]]*//p' "$CFG" 2>/dev/null | head -1 | tr -d '"'"'"' ')"
ART_DIR="$(cd "$(dirname "$CFG")" 2>/dev/null && cd "${ART:-./artifacts}" 2>/dev/null && pwd || true)"
N_ADAPTERS=0
if [[ -n "$ART_DIR" && -d "$ART_DIR" ]]; then
  while IFS= read -r cfgfile; do
    d="$(dirname "$cfgfile")"
    cp -R "$d" "$STAGE/adapters/$(basename "$d")" 2>/dev/null && N_ADAPTERS=$((N_ADAPTERS + 1))
  done < <(find "$ART_DIR" -name adapter_config.json -maxdepth 4 2>/dev/null)
fi
echo "export: $N_ADAPTERS LoRA adapter director(ies); merged and quantized weights deliberately excluded"

# --- environment ---------------------------------------------------------------------------------------------
{
  echo "# pip freeze at export time"
  if command -v uv >/dev/null 2>&1; then uv pip freeze 2>/dev/null
  elif command -v python3 >/dev/null 2>&1; then python3 -m pip freeze 2>/dev/null
  elif command -v pip >/dev/null 2>&1; then pip freeze 2>/dev/null
  else echo "(no pip in this environment)"; fi
} > "$STAGE/env/pip-freeze.txt"
{
  echo "HEAD $(git rev-parse HEAD 2>/dev/null || echo '<not a git repository>')"
  echo "described $(git describe --tags --always --dirty 2>/dev/null || true)"
  echo "# git status --porcelain"
  git status --porcelain 2>/dev/null || true
} > "$STAGE/env/git.txt"
{ command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi; } > "$STAGE/env/nvidia-smi.txt" 2>/dev/null || true

# --- configs ---------------------------------------------------------------------------------------------------
CFG_DIR="$(cd "$(dirname "$CFG")" && pwd)"
for f in "$CFG" "$CFG_DIR/project.yaml" "$CFG_DIR/project.tiny.yaml" "$CFG_DIR/gpu-day.lock.json" requirements-gpu.txt; do
  [[ -f "$f" ]] && cp "$f" "$STAGE/config/" 2>/dev/null
done

# --- manifest and tarball ----------------------------------------------------------------------------------------
{
  echo "agentdistill GPU-day export"
  echo "created_utc  $STAMP"
  echo "config       $CFG"
  echo "head         $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "host         $(hostname 2>/dev/null || echo unknown)"
  echo
  echo "contents:"
  (cd "$STAGE" && find . -type f | sort | sed 's/^\./  /')
} > "$STAGE/MANIFEST.txt"

TARBALL="$OUT/$NAME.tar.gz"
# COPYFILE_DISABLE: macOS tar otherwise adds an AppleDouble `._<name>` entry beside every file, and
# `._registry.db` ends in `.db` too -- the verifier picked that 163-byte header up as the registry and failed.
# Harmless on Linux, where the variable means nothing.
COPYFILE_DISABLE=1 tar czf "$TARBALL" -C "$(dirname "$STAGE")" "$NAME" || { echo "export: tar failed" >&2; exit 1; }
SUM="$(sha256_of "$TARBALL")"
printf '%s  %s\n' "$SUM" "$NAME.tar.gz" > "$TARBALL.sha256"
rm -rf "$(dirname "$STAGE")"

SIZE="$(du -h "$TARBALL" | awk '{print $1}')"
cat <<EOF

================================================================================
export written: $TARBALL  ($SIZE)
checksum:       $TARBALL.sha256
                $SUM

Copy both off the box now, from your laptop:

  scp '<user>@<box>:$(cd "$(dirname "$TARBALL")" && pwd)/$NAME.tar.gz*' ~/Downloads/

Then, on the laptop:

  bash scripts/verify_export.sh ~/Downloads/$NAME.tar.gz

DO NOT TERMINATE OR DELETE THIS BOX UNTIL verify_export.sh PASSES ON THE LAPTOP.
THE REGISTRY IS GITIGNORED AND THIS MACHINE IS DISPOSABLE: IF THE RESULTS ARE ONLY HERE, THEY DO NOT EXIST.
================================================================================
EOF
