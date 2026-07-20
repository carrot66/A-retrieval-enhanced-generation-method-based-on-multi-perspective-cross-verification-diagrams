"""Verify invariance to repeated observations within one provenance family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx

from .projection import (
    METHOD_NAME,
    SOURCE_NODE,
    add_source_family,
    flow_certificate,
    load_jsonl,
)


def build_family_graph(
    families: dict[str, dict[str, int | None]],
    *,
    replicas_per_observation: int = 0,
) -> nx.DiGraph:
    if replicas_per_observation < 0:
        raise ValueError("replicas_per_observation must be non-negative")
    graph = nx.DiGraph()
    graph.add_node(SOURCE_NODE)
    for family, observations in families.items():
        expanded: dict[str, int | None] = dict(observations)
        for role, option_id in observations.items():
            if option_id is None:
                continue
            for replica in range(replicas_per_observation):
                expanded[f"{role}__replica_{replica + 1}"] = int(option_id)
        add_source_family(graph, family, expanded)
    return graph


def certificate_decision(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "winner": certificate["winner"],
        "flow": certificate["flow"],
        "scores": certificate["scores"],
        "tie": certificate["tie"],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {METHOD_NAME} Provenance Replication Check",
        "",
        "Each active answer observation was repeated within its original source family. "
        "The source-to-family edge retained capacity one.",
        "",
        "| Model | Dataset | N | Reconstruction mismatches | Score changes | Winner changes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['samples']} | "
            f"{row['reconstruction_mismatches']} | {row['score_changes']} | "
            f"{row['winner_changes']} |"
        )
    lines.extend(
        [
            "",
            f"Status: **{report['status']}** over {report['samples']} samples with "
            f"{report['replicas_per_observation']} added replicas per active observation.",
            "",
            "This is an observation-level implementation check. It does not establish "
            "invariance to duplicated retrieved documents, paraphrased evidence, or newly "
            "generated model calls.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=8)
    args = parser.parse_args()
    if args.replicas < 1:
        raise ValueError("--replicas must be at least one")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    total_samples = 0
    total_reconstruction_mismatches = 0
    total_score_changes = 0
    total_winner_changes = 0
    expanded_graphs = 0

    for model in config["execution_order"]:
        for dataset in config["models"][model]["datasets"]:
            path = args.root / f"artifacts/projections/{model}/{dataset}.jsonl"
            records = load_jsonl(path)
            row = {
                "model": model,
                "dataset": dataset,
                "samples": len(records),
                "reconstruction_mismatches": 0,
                "score_changes": 0,
                "winner_changes": 0,
            }
            for record in records.values():
                full = record["decision"]["full"]
                stored = certificate_decision(full["certificate"])
                families = full["topology"]["families"]
                number_of_choices = len(stored["scores"])
                baseline_graph = build_family_graph(families)
                replicated_graph = build_family_graph(
                    families,
                    replicas_per_observation=args.replicas,
                )
                baseline = certificate_decision(
                    flow_certificate(baseline_graph, number_of_choices)
                )
                replicated = certificate_decision(
                    flow_certificate(replicated_graph, number_of_choices)
                )
                if baseline != stored:
                    row["reconstruction_mismatches"] += 1
                if replicated["scores"] != baseline["scores"]:
                    row["score_changes"] += 1
                if replicated["winner"] != baseline["winner"]:
                    row["winner_changes"] += 1
                if replicated_graph.number_of_nodes() > baseline_graph.number_of_nodes():
                    expanded_graphs += 1
            rows.append(row)
            total_samples += row["samples"]
            total_reconstruction_mismatches += row["reconstruction_mismatches"]
            total_score_changes += row["score_changes"]
            total_winner_changes += row["winner_changes"]

    passed = (
        total_reconstruction_mismatches == 0
        and total_score_changes == 0
        and total_winner_changes == 0
        and expanded_graphs == total_samples
    )
    report = {
        "method": METHOD_NAME,
        "status": "PASS" if passed else "FAIL",
        "scope": "within-source answer-observation replication",
        "gold_used_for_test": False,
        "replicas_per_observation": args.replicas,
        "pairs": len(rows),
        "samples": total_samples,
        "expanded_graphs": expanded_graphs,
        "reconstruction_mismatches": total_reconstruction_mismatches,
        "score_changes": total_score_changes,
        "winner_changes": total_winner_changes,
        "results": rows,
        "claim_boundary": (
            "This check does not test duplicate retrieval documents, paraphrases, or "
            "independently generated calls."
        ),
    }
    output = args.root / "artifacts/results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "PROVENANCE_REPLICATION.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "PROVENANCE_REPLICATION.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("provenance replication invariance failed")


if __name__ == "__main__":
    run()
