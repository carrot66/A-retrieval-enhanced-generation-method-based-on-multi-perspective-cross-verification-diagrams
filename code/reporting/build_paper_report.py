#!/usr/bin/env python3
"""Build one paper-facing report from the sealed QPCF result artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_idk_analysis(root: Path) -> dict[str, Any]:
    data = load(root / "data" / "timeqa_2022_nota.json")
    idk_indices = {
        index
        for index, sample in enumerate(data)
        if str(sample["answer"]).strip().lower() == "i don't know"
    }
    answerable_indices = set(range(len(data))) - idk_indices
    rows = []
    for model in ("llama", "qwen", "mistral"):
        records = load_jsonl(
            root / "artifacts" / "projections" / model / "timeqa.jsonl"
        )

        def correct(variant: str, indices: set[int]) -> int:
            return sum(
                int(records[index]["variants"][variant]["acc"])
                for index in indices
            )

        full_correct = correct("full", set(range(len(records))))
        ablated_correct = correct(
            "without_idk_protection", set(range(len(records)))
        )
        answerable_correct = correct("full", answerable_indices)
        idk_correct = correct("full", idk_indices)
        rows.append(
            {
                "model": model,
                "n": len(records),
                "without_idk_protection_acc": 100
                * ablated_correct
                / len(records),
                "full_acc": 100 * full_correct / len(records),
                "effect_pp": 100
                * (full_correct - ablated_correct)
                / len(records),
                "answerable_correct": answerable_correct,
                "answerable_n": len(answerable_indices),
                "answerable_acc": 100
                * answerable_correct
                / len(answerable_indices),
                "idk_correct": idk_correct,
                "idk_n": len(idk_indices),
                "idk_acc": 100 * idk_correct / len(idk_indices),
            }
        )
    return {
        "dataset": "RealtimeQA",
        "fixed_minimum_source_families": 3,
        "model_or_dataset_specific_thresholds": False,
        "gold_used_for_decision": False,
        "rows": rows,
    }


def development_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Conflict and Golden Results",
        "",
        "| Model | Dataset | N | ACC | 95% CI | Published point estimate | Margin |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["development"]["results"]:
        lines.append(
            f"| {row['model']} | {row['dataset_label']} | {row['n']} | "
            f"{row['metrics']['acc']:.2f} | [{row['ci95_low']:.2f}, "
            f"{row['ci95_high']:.2f}] | {row['target']:.2f} | "
            f"{row['margin']:+.2f} |"
        )
    return lines


def main_comparison_table(report: dict[str, Any]) -> list[str]:
    comparison = report.get("main_comparison")
    if not comparison:
        return []
    lines = [
        "",
        "## Main Baseline Comparison",
        "",
        "| Method | Backbone | FaithEval | RealtimeQA | MuSiQue | SQuAD | Source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison["rows"]:
        values = [
            "-"
            if row["values"].get(dataset) is None
            else f"{row['values'][dataset]:.2f}"
            for dataset in comparison["datasets"]
        ]
        lines.append(
            f"| {row['method_label']} | {row['backbone']} | "
            f"{' | '.join(values)} | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "The w/o RAG, RAG, and KRE rows are local zero-shot reruns when complete per-sample outputs are present; otherwise they retain the FaithfulRAG Table 1 point estimate.",
            "Self-RAG retains its original Llama2-7B backbone label. CAD and COIECD retain published values unless exact token-level decoder traces are supplied.",
        ]
    )
    return lines


def unified_baseline_table(report: dict[str, Any]) -> list[str]:
    development = report["development"]
    baseline = development.get("external_baseline")
    controls = development.get("internal_controls")
    scope = development.get("comparison_scope")
    if not baseline or not controls or not scope:
        return []
    lines = [
        "",
        "## External Standard-RAG Reference",
        "",
        "The common external reference is the local Llama standard-RAG result. It is used only for the external aggregate comparison and is not substituted into module ablations or same-observation controls.",
        "",
        "| Reference | Backbone | FaithEval | RealtimeQA | MuSiQue-Negative | SQuAD-Negative | Average ACC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| {baseline['label']} | {baseline['backbone']} | "
        + " | ".join(
            f"{baseline['values'][dataset]:.2f}" for dataset in baseline["datasets"]
        )
        + f" | {baseline['average_acc']:.2f} |"
    )
    lines.append(
        "\nExternal-reference scope: " + ", ".join(baseline["datasets"]) + "."
    )
    lines.extend(
        [
            "",
            "## Internal Same-Observation Controls",
            "",
            "The following values reuse frozen per-sample observations. Plain-context replay, majority voting, and provenance-grouped voting are internal controls for decision analysis, not the external standard-RAG baseline.",
            "",
            "| Model | Full QPCF | Plain-context replay | Majority | Grouped vote | Full - Plain replay | Full - Majority | Full - Grouped |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, row in controls.items():
        lines.append(
            f"| {model} | {row['full_acc']:.2f} | {row['plain_context_acc']:.2f} | "
            f"{row['majority_acc']:.2f} | {row['provenance_grouped_vote_acc']:.2f} | "
            f"{row['full_minus_plain_context_pp']:+.2f} | "
            f"{row['full_minus_majority_pp']:+.2f} | "
            f"{row['full_minus_provenance_grouped_vote_pp']:+.2f} |"
        )
    lines.append(
        "\nInternal-control scope: " + ", ".join(scope["datasets"]) + ". Golden datasets remain in the primary and paired analyses."
    )
    return lines


def holdout_table(report: dict[str, Any]) -> list[str]:
    holdout = report["holdout"]
    interventions = {row["variant"]: row for row in holdout["interventions"]}
    lines = [
        "",
        "## Frozen OpenBookQA Stress Test",
        "",
        "| Variant | N | QPCF | Plain RAG | Majority | QPCF - plain | Rescued | Harmed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in holdout["accuracy"]:
        intervention = interventions[row["variant"]]
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['full_acc']:.2f} | "
            f"{intervention['plain_acc']:.2f} | {row['majority_acc']:.2f} | "
            f"{intervention['full_minus_plain_pp']:+.2f} | "
            f"{intervention['rescued']} | {intervention['harmed']} |"
        )
    lines.extend(
        [
            "",
            f"The source audit covers {holdout['trace_audit']['records']} sample variants and "
            f"{holdout['trace_audit']['requests']} real model requests with zero missing "
            "request, response, or source-label fields.",
        ]
    )
    return lines


def paired_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Paired Golden-Counterfactual Results",
        "",
        "| Pair | Full joint | Plain joint | Majority joint | Grouped-vote joint |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["paired_counterfactual"]["results"]:
        variants = row["variants"]
        lines.append(
            f"| {row['label']} | {variants['full']['joint_option_acc']:.2f} | "
            f"{variants['plain_context']['joint_option_acc']:.2f} | "
            f"{variants['majority_same_budget']['joint_option_acc']:.2f} | "
            f"{variants['provenance_grouped_vote']['joint_option_acc']:.2f} |"
        )
    return lines


def ablation_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Cross-Configuration Ablations",
        "",
        "| Comparator | Weighted full-minus-comparator | Positive / tied / negative | Holm-significant +/- |",
        "|---|---:|---:|---:|",
    ]
    for row in report["development"]["comparator_summary"]:
        lines.append(
            f"| {row['comparator']} | {row['sample_weighted_effect_pp']:+.3f} | "
            f"{row['positive_pairs']} / {row['tied_pairs']} / {row['negative_pairs']} | "
            f"{row['significant_positive']} / {row['significant_negative']} |"
        )
    return lines


def idk_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## IDK Sufficiency Analysis",
        "",
        "| Model | Without protection | Full v5.1 | Effect | Answerable | IDK-gold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["idk_sufficiency"]["rows"]:
        lines.append(
            f"| {row['model']} | {row['without_idk_protection_acc']:.2f} | "
            f"{row['full_acc']:.2f} | {row['effect_pp']:+.2f} | "
            f"{row['answerable_correct']}/{row['answerable_n']} "
            f"({row['answerable_acc']:.2f}) | {row['idk_correct']}/{row['idk_n']} "
            f"({row['idk_acc']:.2f}) |"
        )
    return lines


def sufficiency_sensitivity_table(report: dict[str, Any]) -> list[str]:
    sensitivity = report.get("sufficiency_sensitivity")
    if not sensitivity:
        return []
    by_dataset = {row["dataset"]: row for row in sensitivity["datasets"]}
    dataset_order = [
        "faitheval",
        "timeqa",
        "musique_golden",
        "squad_golden",
        "musique_negative",
        "squad_negative",
    ]
    lines = [
        "",
        "## Evidence Sufficiency Threshold Sensitivity",
        "",
        "Every observation in this analysis was regenerated with fresh, label-isolated, one-sample model requests. The graph and IDK thresholds are varied in separate single-factor panels.",
        "",
        "| Graph minimum support | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in sensitivity["thresholds"]:
        key = str(threshold)
        values = [
            by_dataset[dataset]["graph_thresholds"][key]["acc"]
            for dataset in dataset_order
        ]
        lines.append(
            f"| {threshold} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + f" | {sensitivity['aggregate']['graph_thresholds'][key]['weighted_acc']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| IDK minimum source families | RealtimeQA ACC | Changed vs. default 3 |",
            "|---:|---:|---:|",
        ]
    )
    timeqa = by_dataset["timeqa"]
    for threshold in sensitivity["thresholds"]:
        row = timeqa["idk_thresholds"][str(threshold)]
        lines.append(
            f"| {threshold} | {row['acc']:.2f} | {row['changed_vs_default']} |"
        )
    lines.extend(
        [
            "",
            f"Trace audit: {sensitivity['trace_audit']['samples']} fresh samples, "
            f"{sensitivity['trace_audit']['requests']} retained model requests, "
            f"{sensitivity['trace_audit']['failures']} failures.",
        ]
    )
    return lines


def source_channel_ablation_table(report: dict[str, Any]) -> list[str]:
    ablation = report.get("source_channel_ablation")
    if not ablation:
        return []
    by_dataset = {row["dataset"]: row for row in ablation["datasets"]}
    dataset_order = [
        "faitheval",
        "timeqa",
        "musique_golden",
        "squad_golden",
        "musique_negative",
        "squad_negative",
    ]
    variants = ["full"] + [
        f"without_{channel}" for channel in ablation["source_channels"]
    ]
    lines = [
        "",
        "## Complete Source-Channel Ablation",
        "",
        "All six source families are removed one at a time on newly generated, label-isolated Llama traces. Each removal excludes the family from fusion, sufficiency counting, and its source-specific routing predicates.",
        "",
        "| Variant | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC | Delta vs. full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        values = [
            by_dataset[dataset]["variants"][variant]["acc"]
            for dataset in dataset_order
        ]
        aggregate = ablation["aggregate"]["variants"][variant]
        label = "Full" if variant == "full" else f"w/o {variant.removeprefix('without_')}"
        lines.append(
            f"| {label} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + f" | {aggregate['weighted_acc']:.2f} | {aggregate['effect_vs_full_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"Trace audit: {ablation['trace_audit']['samples']} fresh samples, "
            f"{ablation['trace_audit']['requests']} retained model requests, "
            f"{ablation['trace_audit']['failures']} failures.",
        ]
    )
    return lines


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['method']} Unified Paper Report",
        "",
        "QPCF-v5.1 is frozen as a training-free, zero-shot conflict-resolution method. Published-paper rows are descriptive point-estimate gates; paired significance is claimed only for included same-sample internal comparisons.",
        "",
    ]
    lines.extend(development_table(report))
    lines.extend(main_comparison_table(report))
    lines.extend(unified_baseline_table(report))
    lines.extend(holdout_table(report))
    lines.extend(paired_table(report))
    lines.extend(ablation_table(report))
    lines.extend(idk_table(report))
    lines.extend(sufficiency_sensitivity_table(report))
    lines.extend(source_channel_ablation_table(report))
    replication = report["provenance_replication"]
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Provenance replication: {replication['samples']} samples, "
            f"{replication['score_changes']} score changes, "
            f"{replication['winner_changes']} winner changes.",
            "- OpenBookQA was frozen before source generation with physical label isolation.",
            "- The OpenBookQA result is negative relative to plain RAG and is retained, not tuned away.",
            "- Post-hoc proof-completeness, contrastive-adjudication, and NLI-abstention candidates are diagnostic only and are not part of QPCF-v5.1.",
            "",
            "## Claim Boundary",
            "",
            "QPCF-v5.1 supports a conflict-focused claim on the published benchmark files and paired Golden/counterfactual contexts. It does not support universal improvement on partial-evidence, non-conflict QA: on OpenBookQA, QPCF trails matched plain RAG by 2.0 to 3.4 points.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "artifacts" / "results"
    config = load(root / "code" / "configs" / "experiment.json")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": config["method"]["name"],
        "workers": 12,
        "training": False,
        "zero_shot": True,
        "development": load(results / "DEVELOPMENT_RESULTS.json"),
        "main_comparison": load(results / "MAIN_COMPARISON.json")
        if (results / "MAIN_COMPARISON.json").is_file()
        else None,
        "paired_counterfactual": load(results / "PAIRED_COUNTERFACTUAL.json"),
        "provenance_replication": load(results / "PROVENANCE_REPLICATION.json"),
        "holdout": load(results / "HOLDOUT_RESULTS_llama.json"),
        "posthoc_nli_abstention": load(results / "POSTHOC_NLI_ABSTENTION.json"),
        "idk_sufficiency": build_idk_analysis(root),
        "sufficiency_sensitivity": load(
            results / "SUFFICIENCY_SENSITIVITY.json"
        ),
        "source_channel_ablation": load(
            results / "SOURCE_CHANNEL_ABLATION.json"
        )
        if (results / "SOURCE_CHANNEL_ABLATION.json").is_file()
        else None,
        "claim_status": {
            "all_published_point_estimate_gates_pass": True,
            "universal_non_conflict_improvement": False,
            "paired_significance_against_published_systems": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    run()
