"""Convert the official OpenBookQA complete test split to the UGCP schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


ARCHIVE_MEMBER = "OpenBookQA-V1-Sep2018/Data/Additional/test_complete.jsonl"
EXPECTED_RECORDS = 500


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def convert_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    question = record.get("question")
    if not isinstance(question, dict):
        raise ValueError(f"record {index} has no question object")
    raw_choices = question.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) < 2:
        raise ValueError(f"record {index} has invalid choices")

    labels = [str(choice["label"]) for choice in raw_choices]
    choices = [str(choice["text"]) for choice in raw_choices]
    if len(set(labels)) != len(labels):
        raise ValueError(f"record {index} has duplicate choice labels")
    answer_key = str(record.get("answerKey", ""))
    if answer_key not in labels:
        raise ValueError(f"record {index} has invalid answerKey {answer_key!r}")
    answer_index = labels.index(answer_key)

    return {
        "id": str(record["id"]),
        "question": str(question["stem"]),
        "answer": choices[answer_index],
        "answerKey": answer_key,
        "choices": choices,
        "context": str(record["fact1"]),
        "num of options": len(choices),
    }


def convert(source: Path, output: Path, metadata: Path, archive: Path | None) -> None:
    source_bytes = source.read_bytes()
    if archive is not None:
        with zipfile.ZipFile(archive) as bundle:
            archived_source = bundle.read(ARCHIVE_MEMBER)
        if archived_source != source_bytes:
            raise ValueError("source JSONL is not byte-identical to the archive member")

    raw_records = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_records) != EXPECTED_RECORDS:
        raise ValueError(
            f"expected {EXPECTED_RECORDS} records, found {len(raw_records)}"
        )
    converted = [
        convert_record(record, index) for index, record in enumerate(raw_records)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    provenance = {
        "dataset": "OpenBookQA",
        "release": "OpenBookQA-V1-Sep2018",
        "split": "test_complete",
        "records": len(converted),
        "sample_selection": "all official records in source order",
        "context_mapping": "fact1 copied verbatim",
        "question_mapping": "question.stem copied verbatim",
        "choice_mapping": "question.choices[*].text copied in source order",
        "answer_mapping": "answerKey mapped to matching choice text",
        "source_file": str(source.resolve()),
        "source_sha256": sha256(source),
        "archive_file": str(archive.resolve()) if archive is not None else None,
        "archive_sha256": sha256(archive) if archive is not None else None,
        "archive_member": ARCHIVE_MEMBER if archive is not None else None,
        "archive_member_verified": archive is not None,
        "conversion_script_sha256": sha256(Path(__file__).resolve()),
        "output_file": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output, args.metadata, args.archive)


if __name__ == "__main__":
    run()
