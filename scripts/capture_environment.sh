#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPTURE_DIR="${1:-$REPO_ROOT/environment_capture}"
mkdir -p "$CAPTURE_DIR"

python --version > "$CAPTURE_DIR/python-version.txt" 2>&1
python -m pip freeze --all > "$CAPTURE_DIR/pip-freeze.txt"

if command -v conda >/dev/null 2>&1; then
  conda list --explicit > "$CAPTURE_DIR/conda-explicit.txt"
  conda env export --no-builds > "$CAPTURE_DIR/conda-environment-no-builds.yml"
fi

python - <<'PY' > "$CAPTURE_DIR/platform.txt"
import platform
print(platform.platform())
print(platform.processor())
print(platform.machine())
PY

printf 'Environment metadata written to %s\n' "$CAPTURE_DIR"

