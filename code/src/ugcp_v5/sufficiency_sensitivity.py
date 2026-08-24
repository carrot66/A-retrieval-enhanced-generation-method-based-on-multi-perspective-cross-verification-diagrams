"""Evaluate evidence-sufficiency thresholds from a fresh label-isolated model run."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .projection import (
    FLOW_MIN_SOURCE_FAMILIES,
    IDK_MIN_SOURCE_FAMILIES,
    WORKERS,
    load_jsonl,
    observation_nodes,
    project_variant,
    score_option,
    scope_risk_projection,
)


DATA_FILES = {
    "faitheval": "faitheval_data.json",
    "musique_golden": "musique_golden.json",
    "squad_golden": "squad_golden.json",
    "timeqa": "timeqa_2022_nota.json",
    "musique_negative": "musique_negative.json",
    "squad_negative": "squad_negative.json",
}
DATASET_LABELS = {
    "faitheval": "FaithEval",
    "musique_golden": "MuSiQue-Golden",
    "squad_golden": "SQuAD-Golden",
    "timeqa": "RealtimeQA",
    "musique_negative": "MuSiQue-Negative",
    "squad_negative": "SQuAD-Negative",
}
THRESHOLDS = (2, 3, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_fresh_sources(
    *,
    source_root: Path,
    dataset: str,
    model: str,
    size: int,
    inference_hash: str,
) -> dict[str, Any]:
    expected = set(range(size))
    traces: dict[str, dict[int, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for mode in ("plain", "graph"):
        trace_path = source_root / f"{dataset}.{mode}.jsonl"
        summary_path = source_root / f"{dataset}.{mode}.summary.json"
        traces[mode] = load_jsonl(trace_path)
        summaries[mode] = load(summary_path)
        if set(traces[mode]) != expected:
            raise ValueError(f"incomplete fresh {mode} trace for {dataset}")
        summary = summaries[mode]
        checks = {
            "dataset": summary.get("dataset") == dataset,
            "mode": summary.get("mode") == mode,
            "model": str(summary.get("model", "")).lower() == model.lower(),
            "complete": summary.get("completed_samples") == size,
            "no_failures": summary.get("failed_samples") == 0,
            "workers": summary.get("workers") == WORKERS,
            "temperature": summary.get("temperature") == 0,
            "zero_shot": summary.get("zero_shot") is True,
            "decision_only": summary.get("decision_only") is True,
            "isolated": summary.get("physical_label_isolation_enforced") is True,
            "no_labels": summary.get("label_fields_available_to_inference") == [],
            "dataset_hash": summary.get("dataset_sha256") == inference_hash,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"invalid fresh {mode} metadata for {dataset}: {failed}"
            )

    plain_requests = 0
    graph_requests = 0
    verifier_requests = 0
    local_verifier_requests = 0
    for index in range(size):
        plain = traces["plain"][index]
        graph = traces["graph"][index]
        if "gold" in plain or "gold" in graph:
            raise ValueError(f"label leaked into source trace at {dataset}:{index}")
        if not isinstance(plain.get("request"), dict):
            raise ValueError(f"missing plain request at {dataset}:{index}")
        plain_requests += 1
        ugcp = graph.get("ugcp", {})
        if not isinstance(ugcp.get("graph_request"), dict):
            raise ValueError(f"missing graph request at {dataset}:{index}")
        graph_requests += 1
        for field in ("verifier", "local_verifier"):
            branch = ugcp.get(field, {})
            if branch.get("called") is True:
                if not isinstance(branch.get("request"), dict):
                    raise ValueError(
                        f"missing {field} request at {dataset}:{index}"
                    )
                if field == "verifier":
                    verifier_requests += 1
                else:
                    local_verifier_requests += 1
    return {
        "traces": traces,
        "summaries": summaries,
        "request_counts": {
            "plain": plain_requests,
            "graph": graph_requests,
            "global_verifier": verifier_requests,
            "local_verifier": local_verifier_requests,
            "total": (
                plain_requests
                + graph_requests
                + verifier_requests
                + local_verifier_requests
            ),
        },
    }


def accuracy(records: list[dict[str, Any]], family: str, threshold: int) -> float:
    key = str(threshold)
    return 100 * sum(
        float(record[family][key]["acc"]) for record in records
    ) / len(records)


def changed(records: list[dict[str, Any]], family: str, left: int, right: int) -> int:
    return sum(
        record[family][str(left)]["option_id"]
        != record[family][str(right)]["option_id"]
        for record in records
    )


def weighted_accuracy(rows: list[dict[str, Any]], threshold: int) -> float:
    correct = sum(
        row["n"] * row["graph_thresholds"][str(threshold)]["acc"] / 100
        for row in rows
    )
    return 100 * correct / sum(row["n"] for row in rows)


def markdown(report: dict[str, Any]) -> str:
    rows = report["datasets"]
    lines = [
        "# Evidence Sufficiency Threshold Sensitivity",
        "",
        "All observations in this analysis were regenerated sample by sample with the deployed model. Historical source traces were not used. Each API request contains exactly one sample; workers=12 controls only concurrent requests.",
        "",
        "## Graph Certificate Threshold",
        "",
        "The IDK protection threshold remains fixed at its formal value of 3, so this panel changes only the graph-certificate minimum support.",
        "",
        "| Minimum support | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_dataset = {row["dataset"]: row for row in rows}
    order = [
        "faitheval",
        "timeqa",
        "musique_golden",
        "squad_golden",
        "musique_negative",
        "squad_negative",
    ]
    for threshold in report["thresholds"]:
        key = str(threshold)
        values = [
            by_dataset[dataset]["graph_thresholds"][key]["acc"]
            for dataset in order
        ]
        lines.append(
            f"| {threshold} | "
            + " | ".join(f"{value:.2f}" for value in values)
            + f" | {report['aggregate']['graph_thresholds'][key]['weighted_acc']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## IDK Protection Threshold",
            "",
            "The graph-certificate threshold remains fixed at its formal value of 2. This panel is restricted to RealtimeQA because the other formal datasets have no explicit abstention option.",
            "",
            "| Minimum source families | RealtimeQA ACC | Changed vs. 3 |",
            "|---:|---:|---:|",
        ]
    )
    timeqa = by_dataset["timeqa"]
    for threshold in report["thresholds"]:
        key = str(threshold)
        row = timeqa["idk_thresholds"][key]
        lines.append(
            f"| {threshold} | {row['acc']:.2f} | {row['changed_vs_default']} |"
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
            "- Gold labels available to model inference: no",
            f"- Official evaluator SHA-256: `{report['official_evaluator_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(THRESHOLDS))
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise ValueError(f"sensitivity protocol requires --workers {WORKERS}")
    thresholds = tuple(args.thresholds)
    if thresholds != THRESHOLDS:
        raise ValueError(f"formal sensitivity thresholds must be {THRESHOLDS}")

    root = args.root.resolve()
    config = load(root / "code/configs/experiment.json")
    datasets = config["models"][args.model]["datasets"]
    sensitivity_root = root / "artifacts/sensitivity"
    input_root = sensitivity_root / "inputs"
    source_root = sensitivity_root / "source" / args.model
    semantic_root = sensitivity_root / "semantic"
    decision_root = sensitivity_root / "decisions" / args.model
    decision_root.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    total_requests = 0
    total_samples = 0
    total_failures = 0
    for dataset in datasets:
        inference_path = input_root / f"{dataset}.inference.json"
        answer_key_path = input_root / f"{dataset}.answer_key.json"
        inference = load(inference_path)
        answer_key = load(answer_key_path)
        if len(inference) != len(answer_key):
            raise ValueError(f"input and answer key size mismatch: {dataset}")
        if any({"answer", "answerKey"}.intersection(sample) for sample in inference):
            raise ValueError(f"label fields remain in inference input: {dataset}")
        for index, (sample, key) in enumerate(zip(inference, answer_key)):
            if int(key.get("idx", -1)) != index:
                raise ValueError(f"answer-key index mismatch: {dataset}:{index}")
            if str(sample.get("id")) != str(key.get("id")):
                raise ValueError(f"answer-key id mismatch: {dataset}:{index}")

        source = validate_fresh_sources(
            source_root=source_root,
            dataset=dataset,
            model=args.model,
            size=len(inference),
            inference_hash=sha256(inference_path),
        )
        bridge_path = semantic_root / f"{dataset}.bridge.jsonl"
        bridges = load_jsonl(bridge_path)
        if set(bridges) != set(range(len(inference))):
            raise ValueError(f"incomplete semantic trace for {dataset}")

        def decide(index: int) -> dict[str, Any]:
            sample = inference[index]
            choices = [str(choice) for choice in sample["choices"]]
            nodes, plain_parse = observation_nodes(
                source["traces"]["plain"][index],
                source["traces"]["graph"][index],
                bridges[index],
                choices,
            )
            scope_option_id, scope = scope_risk_projection(
                source["traces"]["graph"][index],
                sample,
                int(nodes["plain"]),
            )
            graph_decisions = {}
            idk_decisions = {}
            for threshold in thresholds:
                option_id, diagnostics = project_variant(
                    nodes=nodes,
                    plain_parse=plain_parse,
                    scope_option_id=scope_option_id,
                    scope=scope,
                    number_of_choices=len(choices),
                    minimum_support=threshold,
                    idk_minimum_source_families=IDK_MIN_SOURCE_FAMILIES,
                )
                graph_decisions[str(threshold)] = {
                    "option_id": option_id,
                    "diagnostics": diagnostics,
                }
                option_id, diagnostics = project_variant(
                    nodes=nodes,
                    plain_parse=plain_parse,
                    scope_option_id=scope_option_id,
                    scope=scope,
                    number_of_choices=len(choices),
                    minimum_support=FLOW_MIN_SOURCE_FAMILIES,
                    idk_minimum_source_families=threshold,
                )
                idk_decisions[str(threshold)] = {
                    "option_id": option_id,
                    "diagnostics": diagnostics,
                }
            return {
                "idx": index,
                "nodes": nodes,
                "plain_parse": plain_parse,
                "scope_option_id": scope_option_id,
                "scope": scope,
                "graph": graph_decisions,
                "idk": idk_decisions,
            }

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            decisions = list(executor.map(decide, range(len(inference))))

        records = []
        for decision in decisions:
            index = int(decision["idx"])
            choices = [str(choice) for choice in inference[index]["choices"]]
            gold = answer_key[index]["answer"]
            graph_scored = {
                key: {
                    **score_option(value["option_id"], choices, gold),
                    "diagnostics": value["diagnostics"],
                }
                for key, value in decision["graph"].items()
            }
            idk_scored = {
                key: {
                    **score_option(value["option_id"], choices, gold),
                    "diagnostics": value["diagnostics"],
                }
                for key, value in decision["idk"].items()
            }
            records.append(
                {
                    "idx": index,
                    "graph_thresholds": graph_scored,
                    "idk_thresholds": idk_scored,
                    "decision_inputs": {
                        "nodes": decision["nodes"],
                        "plain_parse": decision["plain_parse"],
                        "scope_option_id": decision["scope_option_id"],
                        "scope": decision["scope"],
                    },
                    "gold": gold,
                }
            )

        decision_path = decision_root / f"{dataset}.jsonl"
        decision_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
            + "\n",
            encoding="utf-8",
        )
        graph_metrics = {}
        idk_metrics = {}
        for threshold in thresholds:
            key = str(threshold)
            graph_metrics[key] = {
                "acc": accuracy(records, "graph_thresholds", threshold),
                "changed_vs_default": changed(
                    records,
                    "graph_thresholds",
                    FLOW_MIN_SOURCE_FAMILIES,
                    threshold,
                ),
            }
            idk_metrics[key] = {
                "acc": accuracy(records, "idk_thresholds", threshold),
                "changed_vs_default": changed(
                    records,
                    "idk_thresholds",
                    IDK_MIN_SOURCE_FAMILIES,
                    threshold,
                ),
            }
        requests = source["request_counts"]
        failures = sum(
            int(source["summaries"][mode]["failed_samples"])
            for mode in ("plain", "graph")
        )
        dataset_rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "n": len(records),
                "graph_thresholds": graph_metrics,
                "idk_thresholds": idk_metrics,
                "fresh_requests": requests,
                "source_hashes": {
                    "plain": sha256(source_root / f"{dataset}.plain.jsonl"),
                    "graph": sha256(source_root / f"{dataset}.graph.jsonl"),
                    "semantic": sha256(bridge_path),
                    "decisions": sha256(decision_path),
                },
            }
        )
        total_requests += requests["total"]
        total_samples += len(records)
        total_failures += failures

    aggregate = {
        "graph_thresholds": {
            str(threshold): {
                "weighted_acc": weighted_accuracy(dataset_rows, threshold),
                "changed_vs_default": sum(
                    row["graph_thresholds"][str(threshold)]["changed_vs_default"]
                    for row in dataset_rows
                ),
            }
            for threshold in thresholds
        }
    }
    evaluator_path = root / "code/src/ugcp_v5/evaluate.py"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": config["method"]["name"],
        "analysis": "evidence sufficiency threshold sensitivity",
        "model": args.model,
        "checkpoint": config["models"][args.model]["checkpoint"],
        "thresholds": list(thresholds),
        "default_graph_minimum_support": FLOW_MIN_SOURCE_FAMILIES,
        "default_idk_minimum_source_families": IDK_MIN_SOURCE_FAMILIES,
        "single_factor_design": True,
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
        "datasets": dataset_rows,
        "aggregate": aggregate,
        "trace_audit": {
            "samples": total_samples,
            "requests": total_requests,
            "failures": total_failures,
            "complete": total_samples
            == sum(len(load(input_root / f"{dataset}.inference.json")) for dataset in datasets)
            and total_failures == 0,
        },
    }
    output = root / "artifacts/results/SUFFICIENCY_SENSITIVITY.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
