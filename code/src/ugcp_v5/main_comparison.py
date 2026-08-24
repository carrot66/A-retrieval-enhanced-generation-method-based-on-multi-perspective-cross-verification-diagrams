"""Build the paper-style main comparison table with provenance labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_LABELS = {
    "faitheval": "FaithEval",
    "timeqa": "RealtimeQA",
    "musique_negative": "MuSiQue",
    "squad_negative": "SQuAD",
}
METHOD_LABELS = {
    "without_rag": "w/o RAG",
    "rag": "RAG",
    "kre": "KRE",
    "cad": "CAD",
    "coiecd": "COIECD",
    "self_rag": "Self-RAG",
    "ours": "MVCG-RAG (Ours)",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_value(root: Path, model: str, dataset: str, method: str) -> float | None:
    path = root / "artifacts" / "baselines" / model / dataset / f"{method}.summary.json"
    if not path.is_file():
        return None
    return float(load_json(path)["metrics"]["acc"])


def ours_value(root: Path, model: str, dataset: str) -> float | None:
    path = root / "artifacts" / "projections" / model / f"{dataset}.summary.json"
    if not path.is_file():
        return None
    summary = load_json(path)
    metrics = summary.get("metrics", {})
    if "full" in metrics:
        return float(metrics["full"]["acc"])
    if "acc" in metrics:
        return float(metrics["acc"])
    return float(summary["results"]["acc"])


def add_row(
    rows: list[dict[str, Any]],
    method: str,
    model: str,
    values: dict[str, float | None],
    source: str,
    backbone: str,
) -> None:
    rows.append(
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "model": model,
            "backbone": backbone,
            "values": values,
            "source": source,
        }
    )


def build(config: dict[str, Any], root: Path) -> dict[str, Any]:
    comparison = config["main_comparison"]
    datasets = comparison["datasets"]
    model_keys = {
        "llama": "llama3.1-8b-instruct",
        "qwen": "qwen2.5-7b-instruct",
        "mistral": "mistral-7b-instruct",
    }
    rows: list[dict[str, Any]] = []
    for method in comparison["published_order"]:
        for model, published_values in comparison["published_baselines"].get(method, {}).items():
            local_model = next(
                (key for key, value in model_keys.items() if value == model), None
            )
            local_values = (
                {
                    dataset: local_value(root, local_model, dataset, method)
                    for dataset in datasets
                }
                if local_model is not None
                else {}
            )
            complete_local = bool(local_values) and all(
                value is not None for value in local_values.values()
            )
            add_row(
                rows,
                method,
                model,
                local_values if complete_local else {
                    dataset: published_values.get(dataset) for dataset in datasets
                },
                "local zero-shot rerun" if complete_local else "FaithfulRAG Table 1",
                comparison["backbones"].get(model, model),
            )

    for model in config["execution_order"]:
        values = {
            dataset: ours_value(root, model, dataset)
            for dataset in datasets
            if dataset in config["models"][model]["datasets"]
        }
        add_row(
            rows,
            "ours",
            model_keys[model],
            values,
            "local MVCG-RAG result",
            config["models"][model]["checkpoint"],
        )

    return {
        "method": config["method"]["name"],
        "datasets": datasets,
        "dataset_labels": {dataset: DATASET_LABELS[dataset] for dataset in datasets},
        "rows": rows,
        "provenance": {
            "published_baselines": "FaithfulRAG arXiv:2506.08938v2 Table 1",
            "local_reruns": "逐样本 zero-shot deployed-model outputs when present",
            "self_rag_scope": "published Llama2-7B result; not the deployed Llama3.1/Qwen/Mistral backbone",
            "cad_coiecd_scope": "published values unless token-level decoder outputs are explicitly supplied",
        },
    }


def write_report(report: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "MAIN_COMPARISON.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Main Comparison",
        "",
        "ACC (%) under the same TruthfulRAG official evaluator and the paper's four conflict datasets.",
        "",
        "| Method | Backbone | FaithEval | RealtimeQA | MuSiQue | SQuAD | Source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        values = [
            "-" if row["values"].get(dataset) is None else f"{row['values'][dataset]:.2f}"
            for dataset in report["datasets"]
        ]
        lines.append(
            f"| {row['method_label']} | {row['backbone']} | "
            f"{' | '.join(values)} | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "Published rows are reported values from FaithfulRAG Table 1, not silently relabeled local reruns.",
            "Local w/o RAG, RAG, and KRE rows replace the corresponding published row only when complete per-sample outputs exist.",
            "Self-RAG is reported with its original Llama2-7B backbone and is not a same-backbone comparison to MVCG-RAG.",
            "CAD and COIECD require token-level probability decoding; their published rows remain separate until an exact decoder run is available.",
        ]
    )
    (output_root / "MAIN_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = build(load_json(args.config), args.root)
    write_report(report, args.root / "artifacts" / "results")
    print(json.dumps({"rows": len(report["rows"]), "root": str(args.root)}, indent=2))


if __name__ == "__main__":
    main()
