"""Run label-isolated, per-source-family fusion ablations on fresh traces."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .statistics import (
    exact_mcnemar,
    holm_adjust,
    paired_bootstrap_interval,
    stable_seed,
)
from .projection import (
    SOURCE_CHANNELS,
    WORKERS,
    load_jsonl,
    observation_nodes,
    project_variant,
    score_option,
    scope_risk_projection,
)
from .sufficiency_sensitivity import (
    DATASET_LABELS,
    sha256,
    validate_fresh_sources,
)


DATA_FILES = {
    "faitheval": "faitheval_data.json",
    "musique_golden": "musique_golden.json",
    "squad_golden": "squad_golden.json",
    "timeqa": "timeqa_2022_nota.json",
    "musique_negative": "musique_negative.json",
    "squad_negative": "squad_negative.json",
}
DATASETS = tuple(DATA_FILES)
VARIANTS = ("full",) + tuple(f"without_{channel}" for channel in SOURCE_CHANNELS)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def enabled_channels(removed_channel: str | None) -> dict[str, bool]:
    if removed_channel is not None and removed_channel not in SOURCE_CHANNELS:
        raise ValueError(f"unknown source channel: {removed_channel}")
    return {
        f"include_{channel}": channel != removed_channel
        for channel in SOURCE_CHANNELS
    }


def fallback_option(
    nodes: dict[str, int | None], removed_channel: str | None
) -> int:
    ordered_nodes = (
        ("plain", "plain"),
        ("graph_call", "graph_base"),
        ("graph_call", "graph_context"),
        ("global_verifier", "global_verifier"),
        ("local_verifier", "local_verifier"),
        ("local_semantic", "local_semantic"),
        ("query_semantic", "query_semantic"),
    )
    for channel, key in ordered_nodes:
        if channel != removed_channel and nodes[key] is not None:
            return int(nodes[key])
    return 0


def validate_semantic_trace(
    bridge_path: Path,
    inference_hash: str,
    dataset: str,
    size: int,
) -> dict[int, dict[str, Any]]:
    bridge = load_jsonl(bridge_path)
    if set(bridge) != set(range(size)):
        raise ValueError(f"incomplete semantic trace for {dataset}")
    if any("gold" in record for record in bridge.values()):
        raise ValueError(f"label leaked into semantic trace for {dataset}")
    summary = load(bridge_path.with_suffix(".summary.json"))
    checks = {
        "dataset": summary.get("dataset") == dataset,
        "complete": summary.get("num_samples") == size,
        "workers": summary.get("workers") == WORKERS,
        "dataset_hash": summary.get("dataset_sha256") == inference_hash,
        "no_grad": summary.get("gradients_enabled") is False,
        "no_training": summary.get("training") is False,
        "no_gold": summary.get("uses_gold_labels") is False,
        "artifact_hash": summary.get("artifact_sha256") == sha256(bridge_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid semantic metadata for {dataset}: {failed}")
    return bridge


def decide(
    index: int,
    inference: list[dict[str, Any]],
    plain: dict[int, dict[str, Any]],
    graph: dict[int, dict[str, Any]],
    bridge: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    sample = inference[index]
    choices = [str(choice) for choice in sample["choices"]]
    nodes, plain_parse = observation_nodes(
        plain[index], graph[index], bridge[index], choices
    )
    variants: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        removed = None if variant == "full" else variant.removeprefix("without_")
        enabled = enabled_channels(removed)
        base_option_id = fallback_option(nodes, removed)
        scope_option_id, scope = scope_risk_projection(
            graph[index],
            sample,
            base_option_id,
            include_graph_call=enabled["include_graph_call"],
            include_global_verifier=enabled["include_global_verifier"],
            include_local_verifier=enabled["include_local_verifier"],
        )
        effective_plain_parse = (
            plain_parse
            if enabled["include_plain"]
            else {**plain_parse, "schema_recovered": False}
        )
        if not enabled["include_graph_call"]:
            # The graph-call family supplies the context/prior conflict route itself.
            # Without it, the remaining fusion families are not permitted to invent a
            # conflict update, so the ablation reduces to its active fallback source.
            option_id = base_option_id
            diagnostics = {
                "source_channel_ablation": "graph_call",
                "routing_disabled_without_graph_call": True,
                "fallback_option_id": base_option_id,
            }
        else:
            option_id, diagnostics = project_variant(
                nodes=nodes,
                plain_parse=effective_plain_parse,
                scope_option_id=scope_option_id,
                scope=scope,
                number_of_choices=len(choices),
                fallback_option_id=base_option_id,
                **enabled,
            )
        variants[variant] = {
            "option_id": option_id,
            "diagnostics": diagnostics,
            "scope_option_id": scope_option_id,
            "scope": scope,
            "removed_channel": removed,
        }
    return {
        "idx": index,
        "nodes": nodes,
        "plain_parse": plain_parse,
        "variants": variants,
    }


def accuracy(records: list[dict[str, Any]], variant: str) -> float:
    return 100 * sum(record["variants"][variant]["acc"] for record in records) / len(records)


def markdown(report: dict[str, Any]) -> str:
    order = [
        "faitheval",
        "timeqa",
        "musique_golden",
        "squad_golden",
        "musique_negative",
        "squad_negative",
    ]
    rows = {row["dataset"]: row for row in report["datasets"]}
    lines = [
        "# Source-Channel Ablation",
        "",
        "Every source trace in this report was regenerated sample by sample with the deployed model. Answer keys were physically separated before inference and joined only after all variant option IDs were fixed.",
        "",
        "Each row removes exactly one source family from graph fusion, source-family support counting, and any routing predicate directly produced by that family. `w/o graph_call` disables the conflict route because this family is the only producer of the context-versus-prior conflict observation; it falls back to the best remaining source observation.",
        "",
        "| Removed source channel | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC | Delta vs. full | Changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        values = [rows[dataset]["variants"][variant]["acc"] for dataset in order]
        aggregate = report["aggregate"]["variants"][variant]
        name = "Full MVCG-RAG" if variant == "full" else f"w/o {variant.removeprefix('without_')}"
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + f" | {aggregate['weighted_acc']:.2f} | {aggregate['effect_vs_full_pp']:+.2f} | {aggregate['changed_vs_full']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Tests",
            "",
            "| Removed source channel | Full better / ablation better | Effect (pp) | 95% bootstrap CI | McNemar p (Holm) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["paired_tests"]:
        name = f"w/o {row['variant'].removeprefix('without_')}"
        lines.append(
            f"| {name} | {row['full_correct_ablation_wrong']}/{row['full_wrong_ablation_correct']} | "
            f"{row['effect_pp']:+.2f} | [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}] | "
            f"{row['p_raw']:.4g} ({row['p_holm']:.4g}) |"
        )
    audit = report["trace_audit"]
    lines.extend(
        [
            "",
            "## Trace Audit",
            "",
            f"- Fresh per-sample decisions: {audit['samples']}",
            f"- Fresh retained model requests: {audit['requests']}",
            f"- Source failures: {audit['failures']}",
            "- Historical observation replay: no",
            "- Gold labels available during model inference or decision: no",
            f"- Official evaluator SHA-256: `{report['official_evaluator_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise ValueError(f"source-channel protocol requires --workers {WORKERS}")

    root = args.root.resolve()
    run_root = root / "artifacts/source_ablation"
    input_root = run_root / "inputs"
    source_root = run_root / "source" / args.model
    semantic_root = run_root / "semantic"
    decision_root = run_root / "decisions" / args.model
    dataset_rows = []
    all_correctness: dict[str, list[int]] = {variant: [] for variant in VARIANTS}
    total_requests = 0
    total_samples = 0
    total_failures = 0

    for dataset in args.datasets:
        inference_path = input_root / f"{dataset}.inference.json"
        answer_path = input_root / f"{dataset}.answer_key.json"
        bridge_path = semantic_root / f"{dataset}.bridge.jsonl"
        inference = load(inference_path)
        answer_key = load(answer_path)
        if len(inference) != len(answer_key):
            raise ValueError(f"input/answer key mismatch for {dataset}")
        inference_hash = sha256(inference_path)
        source = validate_fresh_sources(
            source_root=source_root,
            dataset=dataset,
            model=args.model,
            size=len(inference),
            inference_hash=inference_hash,
        )
        bridge = validate_semantic_trace(
            bridge_path, inference_hash, dataset, len(inference)
        )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            decisions = list(
                executor.map(
                    lambda index: decide(
                        index,
                        inference,
                        source["traces"]["plain"],
                        source["traces"]["graph"],
                        bridge,
                    ),
                    range(len(inference)),
                )
            )
        records = []
        for decision in decisions:
            index = int(decision["idx"])
            choices = [str(choice) for choice in inference[index]["choices"]]
            gold = answer_key[index]["answer"]
            scored = {
                variant: {
                    **score_option(value["option_id"], choices, gold),
                    **value,
                }
                for variant, value in decision["variants"].items()
            }
            records.append(
                {
                    "idx": index,
                    "id": answer_key[index]["id"],
                    "nodes": decision["nodes"],
                    "plain_parse": decision["plain_parse"],
                    "variants": scored,
                    "gold": gold,
                }
            )
        decision_root.mkdir(parents=True, exist_ok=True)
        decision_path = decision_root / f"{dataset}.jsonl"
        decision_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
            + "\n",
            encoding="utf-8",
        )
        metrics = {
            variant: {
                "acc": accuracy(records, variant),
                "changed_vs_full": sum(
                    record["variants"][variant]["option_id"]
                    != record["variants"]["full"]["option_id"]
                    for record in records
                ),
            }
            for variant in VARIANTS
        }
        dataset_rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "n": len(records),
                "variants": metrics,
                "source_hashes": {
                    "plain": sha256(source_root / f"{dataset}.plain.jsonl"),
                    "graph": sha256(source_root / f"{dataset}.graph.jsonl"),
                    "semantic": sha256(bridge_path),
                    "decisions": sha256(decision_path),
                },
                "fresh_requests": source["request_counts"],
            }
        )
        for variant in VARIANTS:
            all_correctness[variant].extend(
                int(record["variants"][variant]["acc"]) for record in records
            )
        total_requests += source["request_counts"]["total"]
        total_samples += len(records)
        total_failures += sum(
            int(source["summaries"][mode]["failed_samples"])
            for mode in ("plain", "graph")
        )

    full = np.asarray(all_correctness["full"], dtype=int)
    tests = []
    for variant in VARIANTS[1:]:
        other = np.asarray(all_correctness[variant], dtype=int)
        better, worse, p_raw = exact_mcnemar(full, other)
        ci_low, ci_high = paired_bootstrap_interval(
            full, other, args.bootstrap, stable_seed(args.seed, variant)
        )
        tests.append(
            {
                "variant": variant,
                "n": len(full),
                "effect_pp": float(100 * (full.mean() - other.mean())),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "full_correct_ablation_wrong": better,
                "full_wrong_ablation_correct": worse,
                "p_raw": p_raw,
            }
        )
    holm_adjust(tests)
    aggregate = {
        "variants": {
            variant: {
                "weighted_acc": float(100 * np.mean(all_correctness[variant])),
                "effect_vs_full_pp": float(
                    100
                    * (
                        np.mean(all_correctness[variant])
                        - np.mean(all_correctness["full"])
                    )
                ),
                "changed_vs_full": sum(
                    row["variants"][variant]["changed_vs_full"]
                    for row in dataset_rows
                ),
            }
            for variant in VARIANTS
        }
    }
    evaluator_path = root / "code/src/ugcp_v5/evaluate.py"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "UGCP-v5.1-QPCF",
        "analysis": "single-factor source-channel fusion ablation",
        "model": args.model,
        "source_channels": list(SOURCE_CHANNELS),
        "datasets": dataset_rows,
        "aggregate": aggregate,
        "paired_tests": tests,
        "fresh_model_inference": True,
        "historical_observation_replay": False,
        "one_sample_per_request": True,
        "workers": args.workers,
        "zero_shot": True,
        "training": False,
        "gold_used_for_model_inference": False,
        "gold_used_for_decision": False,
        "gold_used_for_evaluation_only": True,
        "official_evaluator_sha256": sha256(evaluator_path),
        "trace_audit": {
            "samples": total_samples,
            "requests": total_requests,
            "failures": total_failures,
            "complete": total_failures == 0,
        },
    }
    output = root / "artifacts/results/SOURCE_CHANNEL_ABLATION.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
