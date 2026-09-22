#!/usr/bin/env bash
# The public site: the landing page plus the report the pipeline actually wrote.
#
#   bash scripts/build_site.sh            # build into site/_build
#   bash scripts/build_site.sh --publish  # also push it to the gh-pages branch
#
# The report is copied, never edited and never hand-written: a page that restates a number the pipeline produced
# is a number nobody can trace back to a run id. If there is no report to copy, this fails rather than publishing
# a site with a stale one.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="site/_build"
REPORT="${REPORT:-artifacts/gpu_day/report.html}"
SIDECAR="${REPORT%.html}.json"

if [[ ! -f "$REPORT" ]]; then
  echo "no report at $REPORT -- run 'bash scripts/clean_rehearsal.sh' first" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cp site/index.html "$OUT/index.html"
cp "$REPORT" "$OUT/report.html"
[[ -f "$SIDECAR" ]] && cp "$SIDECAR" "$OUT/report.json"
# Jekyll would otherwise swallow any file or directory starting with an underscore.
touch "$OUT/.nojekyll"
echo "built $OUT ($(find "$OUT" -type f | wc -l | tr -d ' ') files) from $REPORT"

if [[ "${1:-}" == "--publish" ]]; then
  REMOTE="$(git remote get-url origin)"
  TMP="$(mktemp -d)"
  git -C "$TMP" init -q
  git -C "$TMP" checkout -q -b gh-pages
  cp -R "$OUT"/. "$TMP"/
  git -C "$TMP" add -A
  git -C "$TMP" -c user.email="$(git config user.email)" -c user.name="$(git config user.name)" \
    commit -qm "site: built from $(git rev-parse --short HEAD)"
  git -C "$TMP" push -q --force "$REMOTE" gh-pages
  rm -rf "$TMP"
  echo "published to gh-pages"
fi
