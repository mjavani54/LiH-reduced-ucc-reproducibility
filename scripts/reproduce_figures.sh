#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/work/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"
cd "$REPO_ROOT"

python analysis/make_figures.py
python scripts/verify_package.py --skip-manifest

printf 'Regenerated Figures 1-4 in %s/figures\n' "$REPO_ROOT"
