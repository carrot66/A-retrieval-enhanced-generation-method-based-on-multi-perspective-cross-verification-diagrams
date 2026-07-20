"""Evaluate paired Golden and counterfactual decisions after inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyze import (
    exact_mcnemar,
    holm_adjust,
    paired_bootstrap_interval,
    stable_seed,
)
from .projection import METHOD_NAME, VARIANT_ORDER, gold_option_ids, load_jsonl


def is_correct(
    option_id: int,
    sample: dict[str, Any],
) -> bool:
    choices = [str(choice) for choice in sample["choices"]]
    return option_id in gold_option_ids(sample["answer"], choices)


def summarize_variant(
    golden_option_ids: list[int],
    counterfactual_option_ids: list[int],
    golden_samples: list[dict[str, Any]],
    counterfactual_samples: list[dict[str, Any]],
) -> tuple[dict[str, float | int], np.ndarray]:
    total = len(golden_samples)
    if not (
        len(counterfactual_samples)
        == len(golden_option_ids)
        == len(counterfactual_option_ids)
        == total
    ):
        raise ValueError("paired inputs must have equal lengths")

    golden_correct = np.asarray(
        [
            is_correct(option_id, sample)
            for option_id, sample in zip(golden_option_ids, golden_samples)
        ],
        dtype=bool,
    )
    counterfactual_correct = np.asarray(
        [
            is_correct(option_id, sample)
            for option_id, sample in zip(
                counterfactual_option_ids, counterfactual_samples
            )
        ],
        dtype=bool,
    )
    golden_predictions = np.asarray(golden_option_ids, dtype=int)
    counterfactual_predictions = np.asarray(
        counterfactual_option_ids, dtype=int
    )
    joint = golden_correct & counterfactual_correct
    switched = golden_predictions != counterfactual_predictions
    stale_after_golden_success = golden_correct & ~switched
    golden_correct_count = int(golden_correct.sum())

    metrics: dict[str, float | int] = {
        "n": total,
        "golden_option_acc": 100 * float(golden_correct.mean()),
        "counterfactual_option_acc": 100
        * float(counterfactual_correct.mean()),
        "joint_option_acc": 100 * float(joint.mean()),
        "golden_only_rate": 100
        * float((golden_correct & ~counterfactual_correct).mean()),
        "counterfactual_only_rate": 100
        * float((~golden_correct & counterfactual_correct).mean()),
        "neither_rate": 100
        * float((~golden_correct & ~counterfactual_correct).mean()),
        "prediction_switch_rate": 100 * float(switched.mean()),
        "adaptation_given_golden_correct": (
            100 * int(joint.sum()) / golden_correct_count
            if golden_correct_count
            else 0.0
        ),
        "stale_given_golden_correct": (
            100 * int(stale_after_golden_success.sum()) / golden_correct_count
            if golden_correct_count
            else 0.0
        ),
    }
    return metrics, joint.astype(int)


def validate_pair(
    golden_samples: list[dict[str, Any]],
    counterfactual_samples: list[dict[str, Any]],
) -> dict[str, int | bool]:
    if len(golden_samples) != len(counterfactual_samples):
        raise ValueError("Golden and counterfactual datasets differ in length")
    same_questions = 0
    same_choices = 0
    disjoint_answers = 0
    changed_contexts = 0
    for golden, counterfactual in zip(golden_samples, counterfactual_samples):
        same_questions += golden["question"] == counterfactual["question"]
        same_choices += golden["choices"] == counterfactual["choices"]
        golden_ids = gold_option_ids(golden["answer"], golden["choices"])
        counterfactual_ids = gold_option_ids(
            counterfactual["answer"], counterfactual["choices"]
        )
        disjoint_answers += golden_ids.isdisjoint(counterfactual_ids)
        changed_contexts += golden["context"] != counterfactual["context"]
    total = len(golden_samples)
    valid = (
        same_questions
        == same_choices
        == disjoint_answers
        == changed_contexts
        == total
    )
    return {
        "valid": valid,
        "same_questions": same_questions,
        "same_choices": same_choices,
        "disjoint_answers": disjoint_answers,
        "changed_contexts": changed_contexts,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {METHOD_NAME} Paired Counterfactual Evaluation",
        "",
        "Joint option accuracy requires the same method variant to answer both the "
        "Golden context and its answer-changing counterfactual context correctly.",
        "",
        "| Pair | Variant | Golden | Counterfactual | Joint | Switch | Adapt given Golden | Stale given Golden |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    displayed = (
        "full",
        "plain_context",
        "majority_same_budget",
        "provenance_grouped_vote",
    )
    for row in report["results"]:
        for variant in displayed:
            metrics = row["variants"][variant]
            lines.append(
                f"| {row['label']} | {variant} | "
                f"{metrics['golden_option_acc']:.2f} | "
                f"{metrics['counterfactual_option_acc']:.2f} | "
                f"{metrics['joint_option_acc']:.2f} | "
                f"{metrics['prediction_switch_rate']:.2f} | "
                f"{metrics['adaptation_given_golden_correct']:.2f} | "
                f"{metrics['stale_given_golden_correct']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| Pair | Comparator | Joint effect | 95% bootstrap CI | Holm p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for test in report["paired_tests"]:
        lines.append(
            f"| {test['label']} | {test['comparator']} | "
            f"{test['effect_pp']:+.2f} | [{test['ci_low']:+.2f}, "
            f"{test['ci_high']:+.2f}] | {test['p_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "- This is a secondary paired option-level analysis, not the published-paper ACC gate.",
            "- Questions and choices are identical within each pair; contexts change and correct options are disjoint.",
            "- Predictions come from separately frozen per-sample traces. Gold answers are used only after all option IDs are fixed.",
            "- Only Llama is included because Qwen and Mistral were not run on Golden files under the formal protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    paired = config["paired_counterfactual_evaluation"]
    model = paired["model"]
    results: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []

    for pair in paired["pairs"]:
        golden_samples = json.loads(
            (args.root / pair["golden_data"]).read_text(encoding="utf-8")
        )
        counterfactual_samples = json.loads(
            (args.root / pair["counterfactual_data"]).read_text(
                encoding="utf-8"
            )
        )
        integrity = validate_pair(golden_samples, counterfactual_samples)
        if not integrity["valid"]:
            raise ValueError(f"invalid paired dataset: {pair['label']}")

        golden_records = load_jsonl(
            args.root
            / f"artifacts/projections/{model}/{pair['golden_dataset']}.jsonl"
        )
        counterfactual_records = load_jsonl(
            args.root
            / (
                f"artifacts/projections/{model}/"
                f"{pair['counterfactual_dataset']}.jsonl"
            )
        )
        expected = set(range(len(golden_samples)))
        if set(golden_records) != expected or set(counterfactual_records) != expected:
            raise ValueError(f"projection indices do not match: {pair['label']}")

        variants: dict[str, dict[str, float | int]] = {}
        joint_vectors: dict[str, np.ndarray] = {}
        for variant in VARIANT_ORDER:
            golden_options = [
                int(golden_records[index]["variants"][variant]["option_id"])
                for index in range(len(golden_samples))
            ]
            counterfactual_options = [
                int(
                    counterfactual_records[index]["variants"][variant][
                        "option_id"
                    ]
                )
                for index in range(len(counterfactual_samples))
            ]
            metrics, joint = summarize_variant(
                golden_options,
                counterfactual_options,
                golden_samples,
                counterfactual_samples,
            )
            variants[variant] = metrics
            joint_vectors[variant] = joint

        full = joint_vectors["full"]
        for comparator in VARIANT_ORDER:
            if comparator == "full":
                continue
            other = joint_vectors[comparator]
            better, worse, p_value = exact_mcnemar(full, other)
            ci_low, ci_high = paired_bootstrap_interval(
                full,
                other,
                args.bootstrap,
                stable_seed(args.seed, pair["label"], comparator),
            )
            tests.append(
                {
                    "label": pair["label"],
                    "comparator": comparator,
                    "n": len(full),
                    "effect_pp": 100 * float(full.mean() - other.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "full_joint_other_not": better,
                    "full_not_other_joint": worse,
                    "p_raw": p_value,
                }
            )
        results.append(
            {
                "label": pair["label"],
                "model": model,
                "n": len(golden_samples),
                "integrity": integrity,
                "variants": variants,
            }
        )

    holm_adjust(tests)
    report = {
        "method": METHOD_NAME,
        "status": "PASS",
        "scope": "secondary paired Golden-counterfactual option evaluation",
        "model": model,
        "gold_used_for_decision": False,
        "gold_used_for_evaluation": True,
        "bootstrap_repetitions": args.bootstrap,
        "holm_family_size": len(tests),
        "results": results,
        "paired_tests": tests,
    }
    output = args.root / "artifacts/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "PAIRED_COUNTERFACTUAL.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "PAIRED_COUNTERFACTUAL.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "pairs": len(results),
                "tests": len(tests),
                "full_joint_option_acc": {
                    row["label"]: row["variants"]["full"][
                        "joint_option_acc"
                    ]
                    for row in results
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
