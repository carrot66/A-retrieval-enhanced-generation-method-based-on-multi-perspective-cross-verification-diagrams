#!/usr/bin/env python3
"""Build integrity hashes and release metadata after a successful release audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def included_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[:2] == ("artifacts", "manifests"):
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_dir = root / "artifacts/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    audit = load(root / "artifacts/results/RELEASE_AUDIT.json")
    if audit.get("status") != "PASS":
        raise ValueError("release audit must pass before building the manifest")
    paper = load(root / "artifacts/results/PAPER_REPORT.json")
    config = load(root / "code/configs/experiment.json")

    files = included_files(root)
    sums_path = manifest_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(
            f"{sha256(path)}  ./{path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    claims = paper["claim_status"]
    release = {
        "method": config["method"]["name"],
        "long_name": config["method"]["long_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": config["method"]["training"],
        "zero_shot": config["method"]["zero_shot"],
        "workers": config["method"]["workers"],
        "model_dataset_pairs": audit["coverage"]["formal_pairs"],
        "formal_samples": audit["coverage"]["formal_samples"],
        "retained_formal_model_responses": audit["coverage"]["formal_requests"],
        "retained_empty_model_outputs": audit["coverage"].get(
            "formal_empty_model_outputs", 0
        ),
        "holdout_samples": audit["coverage"]["holdout_samples"],
        "holdout_requests": audit["coverage"]["holdout_requests"],
        "release_audit_checks": audit["total_checks"],
        "release_audit_status": audit["status"],
        "all_configured_gates_pass": claims[
            "all_published_point_estimate_gates_pass"
        ],
        "universal_non_conflict_improvement": claims[
            "universal_non_conflict_improvement"
        ],
        "paired_significance_against_published_systems": claims[
            "paired_significance_against_published_systems"
        ],
        "tracked_files": len(files),
        "sha256sums_sha256": sha256(sums_path),
        "paper_report_sha256": sha256(
            root / "artifacts/results/PAPER_REPORT.json"
        ),
        "release_audit_sha256": sha256(
            root / "artifacts/results/RELEASE_AUDIT.json"
        ),
        "official_evaluator_sha256": config["evaluation"][
            "official_evaluator_sha256"
        ],
    }
    release_path = manifest_dir / "RELEASE.json"
    release_path.write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
