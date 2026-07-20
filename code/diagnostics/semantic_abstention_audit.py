#!/usr/bin/env python3
"""Audit a model-native NLI abstention rule without changing frozen QPCF outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ugcp_v5 import projection


DATA_FILES = {
    "faitheval": "faitheval_data.json",
    "musique_golden": "musique_golden.json",
    "musique_negative": "musique_negative.json",
    "squad_golden": "squad_golden.json",
    "squad_negative": "squad_negative.json",
    "timeqa": "timeqa_2022_nota.json",
    "openbookqa_holdout": "openbookqa_test.json",
    "openbookqa_permuted": "openbookqa_test_permuted.json",
    "openbookqa_distractor": "openbookqa_test_distractor.json",
}


def evaluate_pair(
    root: Path,
    *,
    model: str,
    dataset: str,
    threshold: float,
) -> dict[str, Any]:
    data = json.loads((root / "data" / DATA_FILES[dataset]).read_text(encoding="utf-8"))
    plain = projection.load_jsonl(
        root / "artifacts" / "source" / model / f"{dataset}.plain.jsonl"
    )
    graph = projection.load_jsonl(
        root / "artifacts" / "source" / model / f"{dataset}.graph.jsonl"
    )
    bridge = projection.load_jsonl(
        root / "artifacts" / "semantic" / f"{dataset}.bridge.jsonl"
    )
    frozen = projection.load_jsonl(
        root / "artifacts" / "projections" / model / f"{dataset}.jsonl"
    )
    expected = set(range(len(data)))
    if not all(set(records) == expected for records in (plain, graph, bridge, frozen)):
        raise ValueError(f"incomplete records for {model}/{dataset}")

    original_semantic_option_id = projection.semantic_option_id

    def semantic_option_id(
        bridge_record: dict[str, Any], edge: str, number_of_choices: int
    ) -> int | None:
        if edge == "query_entailment":
            scores = [float(value) for value in bridge_record[edge]["option_scores"]]
            if max(scores) <= threshold:
                return None
        return original_semantic_option_id(bridge_record, edge, number_of_choices)

    candidate_correct = 0.0
    frozen_correct = 0.0
    plain_correct = 0.0
    abstained = 0
    changed = 0
    rescues = 0
    harms = 0
    projection.semantic_option_id = semantic_option_id
    try:
        for index, sample in enumerate(data):
            scores = [
                float(value)
                for value in bridge[index]["query_entailment"]["option_scores"]
            ]
            abstained += int(max(scores) <= threshold)
            variants, _ = projection.select_variants(
                plain[index], graph[index], bridge[index], sample
            )
            candidate = projection.score_option(
                variants["full"], [str(choice) for choice in sample["choices"]], sample["answer"]
            )
            frozen_full = frozen[index]["variants"]["full"]
            frozen_plain = frozen[index]["variants"]["plain_context"]
            candidate_correct += float(candidate["acc"])
            frozen_correct += float(frozen_full["acc"])
            plain_correct += float(frozen_plain["acc"])
            changed += int(candidate["option_id"] != frozen_full["option_id"])
            rescues += int(candidate["acc"] == 1 and frozen_full["acc"] == 0)
            harms += int(candidate["acc"] == 0 and frozen_full["acc"] == 1)
    finally:
        projection.semantic_option_id = original_semantic_option_id

    total = len(data)
    return {
        "model": model,
        "dataset": dataset,
        "n": total,
        "entailment_threshold": threshold,
        "threshold_semantics": "entailment posterior exceeds all non-entailment mass",
        "query_abstentions": abstained,
        "query_abstention_rate": 100 * abstained / total,
        "frozen_full_acc": 100 * frozen_correct / total,
        "candidate_acc": 100 * candidate_correct / total,
        "plain_acc": 100 * plain_correct / total,
        "candidate_minus_frozen_pp": 100 * (candidate_correct - frozen_correct) / total,
        "changed_decisions": changed,
        "candidate_rescues_vs_frozen": rescues,
        "candidate_harms_vs_frozen": harms,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-hoc NLI Abstention Audit",
        "",
        "This is a diagnostic candidate evaluated after OpenBookQA outcomes were observed. It is not a frozen holdout result and does not replace the reported QPCF-v5.1 method.",
        "",
        "| Model | Dataset | N | Frozen | Candidate | Plain | Candidate - frozen | Abstain rate | Rescue / harm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['n']} | "
            f"{row['frozen_full_acc']:.2f} | {row['candidate_acc']:.2f} | "
            f"{row['plain_acc']:.2f} | {row['candidate_minus_frozen_pp']:+.2f} | "
            f"{row['query_abstention_rate']:.2f} | "
            f"{row['candidate_rescues_vs_frozen']} / {row['candidate_harms_vs_frozen']} |"
        )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.threshold != 0.5:
        raise ValueError("this audit permits only the model-native 0.5 posterior rule")

    root = args.root.resolve()
    config = json.loads(
        (root / "code" / "configs" / "experiment.json").read_text(encoding="utf-8")
    )
    pairs = [
        (model, dataset)
        for model, model_config in config["models"].items()
        for dataset in model_config["datasets"]
    ]
    pairs.extend(
        ("llama", dataset)
        for dataset in (
            "openbookqa_holdout",
            "openbookqa_permuted",
            "openbookqa_distractor",
        )
    )
    report = {
        "scope": "post-hoc diagnostic after OpenBookQA inspection",
        "changes_frozen_method": False,
        "threshold": args.threshold,
        "results": [
            evaluate_pair(root, model=model, dataset=dataset, threshold=args.threshold)
            for model, dataset in pairs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    run()
