#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"

"$PYTHON" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = root / "artifacts" / "results"
checks = {
    "mechanism": results / "MECHANISM_ABLATION.json",
    "source": results / "SOURCE_CHANNEL_ABLATION.json",
    "sensitivity": results / "SUFFICIENCY_SENSITIVITY.json",
}
for name, path in checks.items():
    if not path.is_file():
        raise SystemExit(f"missing result: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("workers") != 12:
        raise SystemExit(f"{name}: workers is not 12")
    if report.get("gold_used_for_model_inference") is True:
        raise SystemExit(f"{name}: gold was available during inference")
    if report.get("gold_used_for_decision") is True:
        raise SystemExit(f"{name}: gold was available during decision")
    audit = report.get("trace_audit", {})
    if audit.get("failures", 0) != 0 or audit.get("complete") is not True:
        raise SystemExit(f"{name}: trace audit is incomplete: {audit}")
    if report.get("historical_observation_replay") is True:
        raise SystemExit(f"{name}: historical replay is enabled")
print("release verification: PASS")
for name, path in checks.items():
    report = json.loads(path.read_text(encoding="utf-8"))
    audit = report["trace_audit"]
    print(f"{name}: samples={audit.get('samples')} requests={audit.get('requests')} failures={audit.get('failures')}")
PY
