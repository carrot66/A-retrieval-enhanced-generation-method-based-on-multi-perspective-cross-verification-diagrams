"""Replay QPEF-RAG decisions from model and semantic traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .projection import (
    METHOD_NAME,
    VARIANT_ORDER,
    WORKERS,
    load_jsonl,
    score_option,
    select_variants,
)


LABEL_FIELDS = {"answer", "answerKey"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_isolated(
    inference_path: Path, answer_key_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    if not isinstance(inference, list) or not isinstance(answer_key, list):
        raise ValueError("inference input and answer key must be JSON lists")
    if len(inference) != len(answer_key):
        raise ValueError("inference input and answer key have different lengths")
    for index, (sample, key) in enumerate(zip(inference, answer_key)):
        leaked = LABEL_FIELDS.intersection(sample)
        if leaked:
            raise ValueError(
                f"inference record {index} contains label fields {sorted(leaked)}"
            )
        if int(key.get("idx", -1)) != index:
            raise ValueError(f"answer key index mismatch at record {index}")
        if str(key.get("id")) != str(sample.get("id")):
            raise ValueError(f"answer key id mismatch at record {index}")
        if str(key.get("answer")) not in [str(item) for item in sample["choices"]]:
            raise ValueError(f"answer key is not a choice at record {index}")
    return inference, answer_key


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset_path", type=Path)
    inputs.add_argument("--inference_dataset_path", type=Path)
    parser.add_argument("--answer_key_path", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise ValueError(f"QPEF-RAG requires --workers {WORKERS}")

    if args.inference_dataset_path is not None:
        if args.answer_key_path is None:
            raise ValueError("--inference_dataset_path requires --answer_key_path")
        data, answer_key = load_label_isolated(
            args.inference_dataset_path, args.answer_key_path
        )
        dataset_path = args.inference_dataset_path
        physical_label_isolation = True
    else:
        if args.answer_key_path is not None:
            raise ValueError("--answer_key_path requires --inference_dataset_path")
        labeled = json.loads(args.dataset_path.read_text(encoding="utf-8"))
        data = [
            {key: value for key, value in sample.items() if key not in LABEL_FIELDS}
            for sample in labeled
        ]
        answer_key = [
            {
                "idx": index,
                "id": str(sample.get("id", index)),
                "answer": sample["answer"],
            }
            for index, sample in enumerate(labeled)
        ]
        dataset_path = args.dataset_path
        physical_label_isolation = False
    plain = load_jsonl(args.plain)
    graph = load_jsonl(args.graph)
    bridge = load_jsonl(args.bridge)
    expected = set(range(len(data)))
    if not all(set(records) == expected for records in (plain, graph, bridge)):
        raise ValueError("source indices do not match the full dataset")

    def decide(index: int) -> dict[str, Any]:
        sample = data[index]
        variants, diagnostics = select_variants(
            plain[index], graph[index], bridge[index], sample
        )
        return {"idx": index, "option_ids": variants, "decision": diagnostics}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        decisions = list(executor.map(decide, range(len(data))))

    records = []
    for decision in decisions:
        index = int(decision["idx"])
        sample = data[index]
        choices = [str(choice) for choice in sample["choices"]]
        gold = answer_key[index]["answer"]
        variants = {
            name: score_option(option_id, choices, gold)
            for name, option_id in decision["option_ids"].items()
        }
        records.append(
            {
                "idx": index,
                "variants": variants,
                "decision": decision["decision"],
                "gold": gold,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    metrics = {
        variant: {
            metric: 100
            * sum(float(record["variants"][variant][metric]) for record in records)
            / len(records)
            for metric in ("acc", "em", "f1", "option_acc")
        }
        for variant in VARIANT_ORDER
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD_NAME,
        "model": args.model,
        "dataset": args.dataset,
        "workers": args.workers,
        "zero_shot": True,
        "training": False,
        "gold_used_for_decision": False,
        "physical_label_isolation": physical_label_isolation,
        "label_fields_available_to_decision": [],
        "full_official_evaluation": True,
        "num_samples": len(records),
        "dataset_sha256": sha256(dataset_path),
        "answer_key_sha256": (
            sha256(args.answer_key_path) if args.answer_key_path is not None else None
        ),
        "source_hashes": {
            "plain": sha256(args.plain),
            "graph": sha256(args.graph),
            "bridge": sha256(args.bridge),
        },
        "output_sha256": sha256(args.output),
        "metrics": metrics,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
