"""Compose frozen local and query semantic artifacts without reading labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .projection import load_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local = load_jsonl(args.local)
    query = load_jsonl(args.query)
    data = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    expected = set(range(len(data)))
    if set(local) != expected or set(query) != expected:
        raise ValueError("semantic source indices do not match the dataset")
    records = [
        {
            "idx": index,
            "local_alignment": {
                key: value for key, value in local[index].items() if key != "idx"
            },
            "query_entailment": {
                key: value for key, value in query[index].items() if key != "idx"
            },
        }
        for index in range(len(data))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_sha256": sha256(args.dataset_path),
        "num_samples": len(records),
        "training": False,
        "uses_gold_labels": False,
        "local_source": str(args.local.resolve()),
        "local_sha256": sha256(args.local),
        "query_source": str(args.query.resolve()),
        "query_sha256": sha256(args.query),
        "artifact_sha256": sha256(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
