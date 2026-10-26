"""Separate inference inputs from evaluation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LABEL_FIELDS = {"answer", "answerKey"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_record(
    sample: dict[str, Any], index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = {"id", "question", "context", "choices", "answer"} - set(sample)
    if missing:
        raise ValueError(f"record {index} is missing {sorted(missing)}")
    choices = [str(choice) for choice in sample["choices"]]
    answer = str(sample["answer"])
    if answer not in choices:
        raise ValueError(f"record {index} answer is not one of its choices")
    inference = {
        key: value for key, value in sample.items() if key not in LABEL_FIELDS
    }
    leaked = LABEL_FIELDS.intersection(inference)
    if leaked:
        raise RuntimeError(f"record {index} retained label fields {sorted(leaked)}")
    answer_key = {
        "idx": index,
        "id": str(sample["id"]),
        "answer": answer,
    }
    return inference, answer_key


def isolate(
    source: Path,
    inference_output: Path,
    answer_key_output: Path,
    metadata_output: Path,
) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("source must be a non-empty JSON list")
    split = [split_record(sample, index) for index, sample in enumerate(data)]
    inference = [item[0] for item in split]
    answer_key = [item[1] for item in split]
    identifiers = [record["id"] for record in answer_key]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("source contains duplicate record ids")

    inference_output.parent.mkdir(parents=True, exist_ok=True)
    answer_key_output.parent.mkdir(parents=True, exist_ok=True)
    inference_output.write_text(
        json.dumps(inference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    answer_key_output.write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "records": len(data),
        "inference_output": str(inference_output.resolve()),
        "inference_sha256": sha256(inference_output),
        "answer_key_output": str(answer_key_output.resolve()),
        "answer_key_sha256": sha256(answer_key_output),
        "label_fields_removed": sorted(LABEL_FIELDS),
        "physical_label_isolation": True,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inference_output", type=Path, required=True)
    parser.add_argument("--answer_key_output", type=Path, required=True)
    parser.add_argument("--metadata_output", type=Path, required=True)
    args = parser.parse_args()
    isolate(
        args.source,
        args.inference_output,
        args.answer_key_output,
        args.metadata_output,
    )


if __name__ == "__main__":
    run()
