"""Create deterministic option-order and irrelevant-context holdout variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation_for(sample_id: str, option_count: int) -> int:
    if option_count < 2:
        raise ValueError("at least two options are required")
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:4], "big") % (option_count - 1)


def permute_options(sample: dict[str, Any]) -> dict[str, Any]:
    result = dict(sample)
    choices = [str(choice) for choice in sample["choices"]]
    shift = rotation_for(str(sample["id"]), len(choices))
    rotated = choices[shift:] + choices[:shift]
    result["choices"] = rotated
    result["answerKey"] = chr(ord("A") + rotated.index(str(sample["answer"])))
    result["perturbation"] = {
        "type": "deterministic_cyclic_option_permutation",
        "rotation": shift,
        "uses_answer_to_choose_perturbation": False,
    }
    return result


def append_distractor(
    sample: dict[str, Any], distractor: dict[str, Any]
) -> dict[str, Any]:
    result = dict(sample)
    result["context"] = (
        f"{sample['context']}\n\nUnrelated information: {distractor['context']}"
    )
    result["perturbation"] = {
        "type": "next_record_irrelevant_fact",
        "distractor_source_id": str(distractor["id"]),
        "uses_answer_to_choose_perturbation": False,
    }
    return result


def write_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--permuted_output", type=Path, required=True)
    parser.add_argument("--distractor_output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("input must be a JSON list with at least two records")
    permuted = [permute_options(sample) for sample in data]
    distracted = [
        append_distractor(sample, data[(index + 1) % len(data)])
        for index, sample in enumerate(data)
    ]
    write_dataset(args.permuted_output, permuted)
    write_dataset(args.distractor_output, distracted)

    metadata = {
        "base_dataset": str(args.input.resolve()),
        "base_sha256": sha256(args.input),
        "records": len(data),
        "selection": "all records in source order",
        "answer_dependent_routing": False,
        "permutation": {
            "rule": "non-zero cyclic rotation from SHA256(sample id)",
            "output": str(args.permuted_output.resolve()),
            "sha256": sha256(args.permuted_output),
        },
        "distractor": {
            "rule": "append the next record's fact under an unrelated-information prefix",
            "output": str(args.distractor_output.resolve()),
            "sha256": sha256(args.distractor_output),
        },
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
