"""Record a pre-inference hash manifest for the frozen holdout protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


MODEL_METADATA = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(base)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def tracked_files(root: Path) -> list[Path]:
    patterns = (
        "code/**/*.py",
        "code/**/*.sh",
        "code/requirements.txt",
        "data/openbookqa*.json",
        "artifacts/holdout_protocol/*.json",
        "artifacts/semantic/openbookqa*.jsonl",
        "artifacts/semantic/openbookqa*.summary.json",
    )
    files = {
        path.resolve()
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return sorted(files)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    prediction_paths = sorted(
        path
        for pattern in (
            f"artifacts/source/{args.model}/openbookqa*.jsonl",
            f"artifacts/projections/{args.model}/openbookqa*.jsonl",
        )
        for path in root.glob(pattern)
        if path.is_file()
    )
    if prediction_paths:
        raise ValueError(
            "pre-inference freeze refused because holdout predictions already exist: "
            + ", ".join(str(path) for path in prediction_paths)
        )

    model_path = args.model_path.resolve()
    model_files = [
        model_path / name
        for name in MODEL_METADATA
        if (model_path / name).is_file()
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_inference_freeze",
        "method": "UGCP-v5.1-QPCF",
        "model": args.model,
        "model_path": str(model_path),
        "workers": 12,
        "training": False,
        "gold_available_to_inference": False,
        "method_frozen_before_predictions": True,
        "tracked_files": [artifact(path, root) for path in tracked_files(root)],
        "model_metadata": [artifact(path, model_path) for path in model_files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
