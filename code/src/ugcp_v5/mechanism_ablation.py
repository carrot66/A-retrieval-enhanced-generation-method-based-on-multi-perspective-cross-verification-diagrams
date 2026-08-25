"""Score fresh per-variant Llama traces for the mechanism-level ablation study."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .statistics import exact_mcnemar, holm_adjust, paired_bootstrap_interval, stable_seed
from .projection import (
    VARIANT_ORDER,
    WORKERS,
    observation_nodes,
    project_variant,
    provenance_grouped_vote,
    score_option,
    scope_risk_projection,
    same_budget_majority,
)
from .source_channel_ablation import validate_semantic_trace
from .sufficiency_sensitivity import DATASET_LABELS, load, sha256, validate_fresh_sources


DATASETS = (
    "faitheval",
    "musique_golden",
    "squad_golden",
    "timeqa",
    "musique_negative",
    "squad_negative",
)


def select_variant(
    variant: str,
    plain_record: dict[str, Any],
    graph_record: dict[str, Any],
    bridge_record: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    choices = [str(choice) for choice in sample["choices"]]
    nodes, plain_parse = observation_nodes(
        plain_record, graph_record, bridge_record, choices
    )
    base_option_id = int(nodes["plain"])
    scope_option_id, scope = scope_risk_projection(
        graph_record, sample, base_option_id
    )
    if variant == "plain_context":
        option_id = base_option_id
        diagnostics: dict[str, Any] = {"decision": "plain_context"}
    elif variant == "scope_only":
        option_id = scope_option_id
        diagnostics = {"decision": "scope_only"}
    elif variant == "majority_same_budget":
        option_id = same_budget_majority(nodes, base_option_id)
        diagnostics = {"decision": "same_budget_majority"}
    elif variant == "provenance_grouped_vote":
        option_id = provenance_grouped_vote(nodes, base_option_id)
        diagnostics = {"decision": "provenance_grouped_vote"}
    else:
        kwargs: dict[str, Any] = {}
        if variant == "without_query_semantic":
            kwargs["include_query_semantic"] = False
        elif variant == "without_local_semantic":
            kwargs["include_local_semantic"] = False
        elif variant == "without_semantic_bridge":
            kwargs.update(
                include_local_semantic=False,
                include_query_semantic=False,
            )
        elif variant == "without_provenance_capacity":
            kwargs["provenance_capacity"] = False
        elif variant == "shared_backbone_capacity":
            kwargs["shared_backbone_capacity"] = True
        elif variant == "without_typed_topology":
            kwargs.update(provenance_capacity=False, directional_cut=False)
        elif variant == "without_directional_cut":
            kwargs["directional_cut"] = False
        elif variant == "without_risk_projection":
            kwargs["risk_projection"] = False
        elif variant == "without_idk_protection":
            kwargs["idk_protection"] = False
        elif variant != "full":
            raise ValueError(f"unsupported mechanism variant: {variant}")
        option_id, diagnostics = project_variant(
            nodes=nodes,
            plain_parse=plain_parse,
            scope_option_id=scope_option_id,
            scope=scope,
            number_of_choices=len(choices),
            **kwargs,
        )
    return {
        "option_id": option_id,
        "nodes": nodes,
        "plain_parse": plain_parse,
        "scope_option_id": scope_option_id,
        "scope": scope,
        "diagnostics": diagnostics,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def markdown(report: dict[str, Any]) -> str:
    rows = {
        (row["variant"], row["dataset"]): row for row in report["dataset_rows"]
    }
    order = (
        "faitheval",
        "timeqa",
        "musique_golden",
        "squad_golden",
        "musique_negative",
        "squad_negative",
    )
    lines = [
        "# Fresh Mechanism Ablation",
        "",
        "Every configuration below was generated from a separate, fresh Llama run. Each request contains exactly one label-isolated sample; no historical source trace or decision replay is read.",
        "",
        "| Variant | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC | Delta vs. full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in report["variants"]:
        values = [rows[(variant, dataset)]["acc"] for dataset in order]
        aggregate = report["aggregate"][variant]
        label = "Full MVCG-RAG" if variant == "full" else f"w/o {variant}"
        lines.append(
            f"| {label} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + f" | {aggregate['weighted_acc']:.2f} | {aggregate['effect_vs_full_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Trace Audit",
            "",
            f"- Fresh per-variant decisions: {report['trace_audit']['samples']}",
            f"- Fresh retained Llama requests: {report['trace_audit']['requests']}",
            f"- Source failures: {report['trace_audit']['failures']}",
            "- Historical observation replay: no",
            "- Gold labels available during model inference or decision: no",
            f"- Official evaluator SHA-256: `{report['official_evaluator_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise ValueError(f"mechanism ablation requires --workers {WORKERS}")

    root = args.root.resolve()
    run_root = args.run_root.resolve()
    input_root = run_root / "inputs"
    source_root = run_root / "source" / args.model
    semantic_root = run_root / "semantic"
    decision_root = run_root / "decisions" / args.model

    all_correctness: dict[str, list[int]] = {variant: [] for variant in VARIANT_ORDER}
    dataset_rows: list[dict[str, Any]] = []
    total_requests = 0
    total_failures = 0

    for variant in VARIANT_ORDER:
        variant_source_root = source_root / variant
        variant_semantic_root = semantic_root / variant
        variant_decision_root = decision_root / variant
        for dataset in DATASETS:
            inference_path = input_root / f"{dataset}.inference.json"
            answer_key_path = input_root / f"{dataset}.answer_key.json"
            inference = load(inference_path)
            answer_key = load(answer_key_path)
            if len(inference) != len(answer_key):
                raise ValueError(f"input/answer mismatch for {variant}:{dataset}")
            source = validate_fresh_sources(
                source_root=variant_source_root,
                dataset=dataset,
                model=args.model,
                size=len(inference),
                inference_hash=sha256(inference_path),
            )
            bridge_path = variant_semantic_root / f"{dataset}.bridge.jsonl"
            bridge = validate_semantic_trace(
                bridge_path, sha256(inference_path), dataset, len(inference)
            )

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                decisions = list(
                    executor.map(
                        lambda index: select_variant(
                            variant,
                            source["traces"]["plain"][index],
                            source["traces"]["graph"][index],
                            bridge[index],
                            inference[index],
                        ),
                        range(len(inference)),
                    )
                )
            records = []
            for index, decision in enumerate(decisions):
                choices = [str(choice) for choice in inference[index]["choices"]]
                scored = score_option(
                    int(decision["option_id"]), choices, answer_key[index]["answer"]
                )
                records.append(
                    {
                        "idx": index,
                        "id": answer_key[index]["id"],
                        "variant": variant,
                        "decision": {key: value for key, value in decision.items() if key != "option_id"},
                        "result": scored,
                        "gold": answer_key[index]["answer"],
                    }
                )
            decision_path = variant_decision_root / f"{dataset}.jsonl"
            write_jsonl(decision_path, records)
            correct = [int(record["result"]["acc"]) for record in records]
            all_correctness[variant].extend(correct)
            total_requests += source["request_counts"]["total"]
            total_failures += sum(
                int(source["summaries"][mode]["failed_samples"])
                for mode in ("plain", "graph")
            )
            dataset_rows.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "n": len(records),
                    "acc": 100 * sum(correct) / len(correct),
                    "source_hashes": {
                        "plain": sha256(variant_source_root / f"{dataset}.plain.jsonl"),
                        "graph": sha256(variant_source_root / f"{dataset}.graph.jsonl"),
                        "semantic": sha256(bridge_path),
                        "decisions": sha256(decision_path),
                    },
                    "fresh_requests": source["request_counts"],
                }
            )

    full = np.asarray(all_correctness["full"], dtype=int)
    tests = []
    for variant in VARIANT_ORDER[1:]:
        other = np.asarray(all_correctness[variant], dtype=int)
        better, worse, p_value = exact_mcnemar(full, other)
        ci_low, ci_high = paired_bootstrap_interval(
            full, other, args.bootstrap, stable_seed(args.seed, "mechanism", variant)
        )
        tests.append(
            {
                "variant": variant,
                "n": len(full),
                "effect_pp": float(100 * (full.mean() - other.mean())),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "full_correct_variant_wrong": better,
                "full_wrong_variant_correct": worse,
                "p_raw": p_value,
            }
        )
    holm_adjust(tests)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "UGCP-v5.1-QPCF",
        "analysis": "fresh per-variant mechanism ablation",
        "model": args.model,
        "variants": list(VARIANT_ORDER),
        "datasets": list(DATASETS),
        "dataset_rows": dataset_rows,
        "aggregate": {
            variant: {
                "n": len(all_correctness[variant]),
                "weighted_acc": float(100 * np.mean(all_correctness[variant])),
                "effect_vs_full_pp": float(
                    100 * (np.mean(all_correctness[variant]) - np.mean(full))
                ),
            }
            for variant in VARIANT_ORDER
        },
        "paired_tests": tests,
        "fresh_model_inference": True,
        "historical_observation_replay": False,
        "one_sample_per_request": True,
        "workers": args.workers,
        "training": False,
        "gold_used_for_model_inference": False,
        "gold_used_for_decision": False,
        "gold_used_for_evaluation_only": True,
        "trace_audit": {
            "samples": sum(len(all_correctness[variant]) for variant in VARIANT_ORDER),
            "requests": total_requests,
            "failures": total_failures,
            "complete": total_failures == 0,
        },
        "official_evaluator_sha256": sha256(root / "code/src/ugcp_v5/evaluate.py"),
    }
    output = root / "artifacts/results/MECHANISM_ABLATION.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "PASS", "trace_audit": report["trace_audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    run()
