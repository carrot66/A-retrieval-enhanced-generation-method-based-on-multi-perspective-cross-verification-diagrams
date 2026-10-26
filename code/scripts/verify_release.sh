#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/code/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m unittest discover -s "$ROOT/code/tests" -v
PYTHON="$PYTHON" RUN_ROOT="$ROOT" bash "$ROOT/code/scripts/replay_formal.sh"
(
  cd "$ROOT"
  sha256sum -c artifacts/manifests/SHA256SUMS
)
