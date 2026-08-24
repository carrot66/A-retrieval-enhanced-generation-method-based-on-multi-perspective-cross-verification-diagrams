"""Build primary, ablation, and paired-statistics reports for UGCP-v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .projection import METHOD_NAME, VARIANT_ORDER, load_jsonl


DATASET_LABELS = {
    "faitheval": "FaithEval",
    "timeqa": "RealtimeQA",
    "musique_negative": "MuSiQue-Negative",
    "squad_negative": "SQuAD-Negative",
    "musique_golden": "MuSiQue-Golden",
    "squad_golden": "SQuAD-Golden",
}
COMPARATORS = [name for name in VARIANT_ORDER if name != "full"]
INTERNAL_CONTROL_VARIANTS = {
    "plain_context",
    "majority_same_budget",
    "provenance_grouped_vote",
}


def stable_seed(*parts: str | int) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    probability = correct / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return 100 * (center - radius), 100 * (center + radius)


def exact_mcnemar(full: np.ndarray, other: np.ndarray) -> tuple[int, int, float]:
    better = int(np.sum((full == 1) & (other == 0)))
    worse = int(np.sum((full == 0) & (other == 1)))
    discordant = better + worse
    if discordant == 0:
        return better, worse, 1.0
    tail = min(better, worse)
    probability = 2 * sum(
        math.exp(
            math.lgamma(discordant + 1)
            - math.lgamma(index + 1)
            - math.lgamma(discordant - index + 1)
            - discordant * math.log(2)
        )
        for index in range(tail + 1)
    )
    return better, worse, min(1.0, probability)


def paired_bootstrap_interval(
    full: np.ndarray,
    other: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    difference = full.astype(int) - other.astype(int)
    counts = Counter(int(value) for value in difference)
    probabilities = np.asarray(
        [counts[-1], counts[0], counts[1]], dtype=float
    ) / len(difference)
    samples = np.random.default_rng(seed).multinomial(
        len(difference), probabilities, size=repetitions
    )
    effects = 100 * (samples[:, 2] - samples[:, 0]) / len(difference)
    low, high = np.quantile(effects, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(tests: list[dict[str, Any]]) -> None:
    order = sorted(range(len(tests)), key=lambda index: tests[index]["p_raw"])
    running = 0.0
    total = len(tests)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(tests[index]["p_raw"]))
        running = max(running, adjusted)
        tests[index]["p_holm"] = running
        tests[index]["holm_significant_0_05"] = running < 0.05


def metric_average(records: list[dict[str, Any]], variant: str) -> dict[str, float]:
    return {
        metric: 100
        * sum(float(record["variants"][variant][metric]) for record in records)
        / len(records)
        for metric in ("acc", "em", "f1", "option_acc")
    }


def load_external_baseline(
    root: Path, datasets: list[str]
) -> dict[str, Any]:
    comparison_path = root / "artifacts/results/MAIN_COMPARISON.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in comparison["rows"]
        if item["method"] == "rag"
        and item["model"] == "llama3.1-8b-instruct"
    )
    values = {dataset: float(row["values"][dataset]) for dataset in datasets}
    return {
        "method": "standard_rag",
        "label": "Standard RAG",
        "model": "llama3.1-8b-instruct",
        "backbone": row["backbone"],
        "datasets": datasets,
        "values": values,
        "average_acc": sum(values.values()) / len(values),
        "source": "MAIN_COMPARISON.json; local Llama zero-shot rerun",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {METHOD_NAME} Development Results",
        "",
        "## Primary Results",
        "",
        "| Model | Dataset | N | ACC | 95% CI | Target | Margin | Pass |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['model']} | {row['dataset_label']} | {row['n']} | "
            f"{row['metrics']['acc']:.2f} | [{row['ci95_low']:.2f}, "
            f"{row['ci95_high']:.2f}] | {row['target']:.2f} | "
            f"{row['margin']:+.2f} | {'yes' if row['pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## External Standard-RAG Reference",
            "",
            "The common external reference is the local Llama standard-RAG result from MAIN_COMPARISON.json. Its value is fixed before interpreting model-specific results; it is not used to recompute module ablations or same-observation controls.",
            "",
            "| Reference | Backbone | FaithEval | RealtimeQA | MuSiQue-Negative | SQuAD-Negative | Average ACC |",
            "|---|---|---:|---:|---:|---:|---:|",
            f"| Standard RAG | {report['external_baseline']['backbone']} | "
            + " | ".join(
                f"{report['external_baseline']['values'][dataset]:.2f}"
                for dataset in report["external_baseline"]["datasets"]
            )
            + f" | {report['external_baseline']['average_acc']:.2f} |",
            "",
            "## Internal Same-Observation Controls",
            "",
            "These controls reuse the same frozen per-sample observations as the full method. Their plain-context replay is an internal diagnostic and must not be confused with the external Llama standard-RAG reference above.",
            "",
            "| Model | Full QPCF | Plain-context replay | Majority | Grouped vote | Full - Plain replay | Full - Majority | Full - Grouped |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, row in report["internal_controls"].items():
        lines.append(
            f"| {model} | {row['full_acc']:.2f} | {row['plain_context_acc']:.2f} | "
            f"{row['majority_acc']:.2f} | {row['provenance_grouped_vote_acc']:.2f} | "
            f"{row['full_minus_plain_context_pp']:+.2f} | "
            f"{row['full_minus_majority_pp']:+.2f} | "
            f"{row['full_minus_provenance_grouped_vote_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Comparator Summary",
            "",
            "| Comparator | Weighted effect | Positive/Tied/Negative pairs | Holm-significant positive/negative |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["comparator_summary"]:
        lines.append(
            f"| {row['comparator']} | {row['sample_weighted_effect_pp']:+.3f} | "
            f"{row['positive_pairs']}/{row['tied_pairs']}/{row['negative_pairs']} | "
            f"{row['significant_positive']}/{row['significant_negative']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "- These are adaptive development-trace replays, not an untouched holdout.",
            "- Majority and every ablation reuse exactly the same frozen model observations.",
            "- Gold labels are read only after every variant option ID is fixed.",
            "- Published targets are point estimates; paired significance against the papers requires reproduced per-sample baseline predictions.",
            f"- External standard-RAG reference: Llama {report['external_baseline']['average_acc']:.2f}% average over {', '.join(report['external_baseline']['datasets'])}.",
            f"- Internal ablation/control scope: {', '.join(report['comparison_scope']['datasets'])}; Golden datasets remain in the primary and paired analyses but are excluded from this aggregate scope.",
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
    comparison_dataset_order = list(config["main_comparison"]["datasets"])
    comparison_datasets = set(comparison_dataset_order)
    external_baseline = load_external_baseline(args.root, comparison_dataset_order)
    comparison_scope = {
        "purpose": "internal ablation and same-observation control scope",
        "datasets": comparison_dataset_order,
        "excludes_golden": True,
    }

    rows = []
    tests = []
    correctness: dict[tuple[str, str, str], np.ndarray] = {}
    for model in config["execution_order"]:
        for dataset in config["models"][model]["datasets"]:
            path = args.root / f"artifacts/projections/{model}/{dataset}.jsonl"
            by_index = load_jsonl(path)
            records = [by_index[index] for index in sorted(by_index)]
            variants = {
                variant: metric_average(records, variant)
                for variant in VARIANT_ORDER
            }
            full = np.asarray(
                [record["variants"]["full"]["acc"] for record in records],
                dtype=int,
            )
            correctness[(model, dataset, "full")] = full
            correct = int(full.sum())
            low, high = wilson_interval(correct, len(records))
            target = float(config["models"][model]["targets"][dataset])
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "n": len(records),
                    "correct": correct,
                    "metrics": variants["full"],
                    "variants": variants,
                    "ci95_low": low,
                    "ci95_high": high,
                    "target": target,
                    "margin": variants["full"]["acc"] - target,
                    "pass": variants["full"]["acc"] > target,
                }
            )
            for comparator in COMPARATORS:
                other = np.asarray(
                    [record["variants"][comparator]["acc"] for record in records],
                    dtype=int,
                )
                correctness[(model, dataset, comparator)] = other
                better, worse, p_value = exact_mcnemar(full, other)
                ci_low, ci_high = paired_bootstrap_interval(
                    full,
                    other,
                    args.bootstrap,
                    stable_seed(args.seed, model, dataset, comparator),
                )
                tests.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "comparator": comparator,
                        "n": len(records),
                        "effect_pp": float(100 * (full.mean() - other.mean())),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "full_correct_other_wrong": better,
                        "full_wrong_other_correct": worse,
                        "p_raw": p_value,
                    }
                )
    holm_adjust(tests)
    comparison_tests = [
        test for test in tests if test["dataset"] in comparison_datasets
    ]
    # The reported ablation family follows the same four-dataset baseline scope.
    holm_adjust(comparison_tests)

    model_majority = {}
    for model in config["execution_order"]:
        pairs = [
            row
            for row in rows
            if row["model"] == model and row["dataset"] in comparison_datasets
        ]
        total = sum(row["n"] for row in pairs)
        full_correct = sum(row["correct"] for row in pairs)
        correct_by_variant = {
            variant: sum(
                int(correctness[(model, row["dataset"], variant)].sum())
                for row in pairs
            )
            for variant in (
                "plain_context",
                "majority_same_budget",
                "provenance_grouped_vote",
            )
        }
        model_majority[model] = {
            "n": total,
            "datasets": [row["dataset"] for row in pairs],
            "full_acc": 100 * full_correct / total,
            "plain_context_acc": 100 * correct_by_variant["plain_context"] / total,
            "majority_acc": 100
            * correct_by_variant["majority_same_budget"]
            / total,
            "provenance_grouped_vote_acc": 100
            * correct_by_variant["provenance_grouped_vote"]
            / total,
            "full_minus_plain_context_pp": 100
            * (full_correct - correct_by_variant["plain_context"])
            / total,
            "full_minus_majority_pp": 100
            * (full_correct - correct_by_variant["majority_same_budget"])
            / total,
            "full_minus_provenance_grouped_vote_pp": 100
            * (full_correct - correct_by_variant["provenance_grouped_vote"])
            / total,
            "effect_pp": 100
            * (full_correct - correct_by_variant["majority_same_budget"])
            / total,
        }

    comparator_summary = []
    for comparator in COMPARATORS:
        if comparator in INTERNAL_CONTROL_VARIANTS:
            continue
        relevant = [
            test
            for test in comparison_tests
            if test["comparator"] == comparator
        ]
        total_n = sum(test["n"] for test in relevant)
        weighted = sum(test["effect_pp"] * test["n"] for test in relevant) / total_n
        comparator_summary.append(
            {
                "comparator": comparator,
                "sample_weighted_effect_pp": weighted,
                "positive_pairs": sum(test["effect_pp"] > 0 for test in relevant),
                "tied_pairs": sum(test["effect_pp"] == 0 for test in relevant),
                "negative_pairs": sum(test["effect_pp"] < 0 for test in relevant),
                "significant_positive": sum(
                    test["effect_pp"] > 0 and test["holm_significant_0_05"]
                    for test in relevant
                ),
                "significant_negative": sum(
                    test["effect_pp"] < 0 and test["holm_significant_0_05"]
                    for test in relevant
                ),
            }
        )

    report = {
        "method": METHOD_NAME,
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "scope": "adaptive development-trace replay",
        "workers": 12,
        "zero_shot": True,
        "training": False,
        "gold_used_for_decision": False,
        "bootstrap_repetitions": args.bootstrap,
        "holm_family_size": len(tests),
        "results": rows,
        "model_majority": model_majority,
        "internal_controls": model_majority,
        "external_baseline": external_baseline,
        "comparison_scope": comparison_scope,
        "paired_tests": tests,
        "comparator_summary": comparator_summary,
    }
    output = args.root / "artifacts/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "DEVELOPMENT_RESULTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "DEVELOPMENT_RESULTS.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "pairs": len(rows), "tests": len(tests), "model_majority": model_majority}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("one or more paper targets failed")


if __name__ == "__main__":
    run()
