"""Run the frozen source generator on an arbitrary label-isolated holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import source_runner


FORBIDDEN_LABEL_FIELDS = {"answer", "answerKey"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_label_isolated(dataset_path: Path) -> None:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("holdout inference input must be a non-empty JSON list")
    for index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise ValueError(f"holdout record {index} is not an object")
        leaked = FORBIDDEN_LABEL_FIELDS.intersection(sample)
        if leaked:
            raise ValueError(
                f"holdout record {index} contains forbidden label fields "
                f"{sorted(leaked)}"
            )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--dataset_name", default="openbookqa_holdout")
    parser.add_argument("--mode", choices=["plain", "graph"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base_input", type=Path)
    parser.add_argument("--base_url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--model", default="llama")
    parser.add_argument(
        "--prompt_format",
        choices=["auto", "llama", "qwen", "mistral"],
        default="llama",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    assert_label_isolated(dataset_path)
    if args.workers != 12:
        raise ValueError("Formal UGCP holdout runs require --workers 12")
    if args.mode == "graph" and args.base_input is None:
        raise ValueError("graph mode requires --base_input")

    source_runner.DATA_PATHS[args.dataset_name] = dataset_path
    forwarded = [
        str(Path(source_runner.__file__).resolve()),
        "--dataset",
        args.dataset_name,
        "--mode",
        args.mode,
        "--output",
        str(args.output),
        "--base_url",
        args.base_url,
        "--model",
        args.model,
        "--prompt_format",
        args.prompt_format,
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--decision_only",
    ]
    if args.base_input is not None:
        forwarded.extend(["--base_input", str(args.base_input)])
    if args.no_resume:
        forwarded.append("--no_resume")
    if args.allow_partial:
        forwarded.append("--allow_partial")

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        source_runner.run()
    finally:
        sys.argv = original_argv

    summary_path = args.output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["holdout_protocol"] = {
        "decision_only_enforced": True,
        "physical_label_isolation_enforced": True,
        "forbidden_label_fields": sorted(FORBIDDEN_LABEL_FIELDS),
        "full_split_enforced": True,
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "source_core_sha256": sha256(Path(source_runner.__file__).resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
