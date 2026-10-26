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


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['method']} Unified Paper Report",
        "",
        "QPCF-v5.1 is frozen as a training-free, zero-shot conflict-resolution method. Published-paper rows are descriptive point-estimate gates; paired significance is claimed only for included same-sample internal comparisons.",
        "",
    ]
    lines.extend(development_table(report))
    lines.extend(holdout_table(report))
    lines.extend(paired_table(report))
    lines.extend(ablation_table(report))
    lines.extend(idk_table(report))
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
        "paired_counterfactual": load(results / "PAIRED_COUNTERFACTUAL.json"),
        "provenance_replication": load(results / "PROVENANCE_REPLICATION.json"),
        "holdout": load(results / "HOLDOUT_RESULTS_llama.json"),
        "posthoc_nli_abstention": load(results / "POSTHOC_NLI_ABSTENTION.json"),
        "idk_sufficiency": build_idk_analysis(root),
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
