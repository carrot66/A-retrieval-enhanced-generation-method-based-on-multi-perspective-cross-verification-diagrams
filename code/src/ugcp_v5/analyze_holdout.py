"""Analyze the frozen OpenBookQA holdout and robustness variants."""

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
    wilson_interval,
)
from .projection import METHOD_NAME, VARIANT_ORDER, load_jsonl, normalize


VARIANTS = ("base", "permuted", "distractor")
SOURCE_STEMS = {
    "base": "openbookqa_holdout",
    "permuted": "openbookqa_permuted",
    "distractor": "openbookqa_distractor",
}
FORBIDDEN_SOURCE_FIELDS = {"answer", "answerKey", "gold", "metrics", "variants"}


def correctness(records: list[dict[str, Any]], variant: str) -> np.ndarray:
    return np.asarray(
        [int(record["variants"][variant]["acc"]) for record in records],
        dtype=int,
    )


def predictions(records: list[dict[str, Any]], variant: str) -> list[str]:
    return [
        normalize(record["variants"][variant]["prediction"]) for record in records
    ]


def paired_test(
    first: np.ndarray,
    second: np.ndarray,
    *,
    name: str,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    first_only, second_only, p_value = exact_mcnemar(first, second)
    low, high = paired_bootstrap_interval(first, second, bootstrap, seed)
    return {
        "comparison": name,
        "n": len(first),
        "first_acc": 100 * float(first.mean()),
        "second_acc": 100 * float(second.mean()),
        "effect_pp_first_minus_second": 100 * float(first.mean() - second.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "p_raw": p_value,
    }


def intervention_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    full = correctness(records, "full")
    plain = correctness(records, "plain_context")
    full_predictions = predictions(records, "full")
    plain_predictions = predictions(records, "plain_context")
    changed = np.asarray(
        [first != second for first, second in zip(full_predictions, plain_predictions)],
        dtype=bool,
    )
    rescued = (full == 1) & (plain == 0)
    harmed = (full == 0) & (plain == 1)
    both_correct = (full == 1) & (plain == 1)
    both_wrong = (full == 0) & (plain == 0)
    return {
        "n": len(records),
        "plain_acc": 100 * float(plain.mean()),
        "full_acc": 100 * float(full.mean()),
        "full_minus_plain_pp": 100 * float(full.mean() - plain.mean()),
        "changed_predictions": int(changed.sum()),
        "changed_prediction_rate": 100 * float(changed.mean()),
        "rescued": int(rescued.sum()),
        "harmed": int(harmed.sum()),
        "net_rescues": int(rescued.sum() - harmed.sum()),
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
        "rescued_indices": np.flatnonzero(rescued).astype(int).tolist(),
        "harmed_indices": np.flatnonzero(harmed).astype(int).tolist(),
    }


def source_trace_audit(
    source_dir: Path,
    projection_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    total_requests = 0
    total_records = 0
    for name in VARIANTS:
        stem = SOURCE_STEMS[name]
        plain_path = source_dir / f"{stem}.plain.jsonl"
        graph_path = source_dir / f"{stem}.graph.jsonl"
        plain = load_jsonl(plain_path)
        graph = load_jsonl(graph_path)
        expected = set(range(len(projection_records[name])))
        if set(plain) != expected or set(graph) != expected:
            raise ValueError(f"{name} source indices do not match projections")

        missing_requests = 0
        missing_responses = 0
        leaked_fields: dict[int, list[str]] = {}
        conditional_pairs = 0
        requests = 0
        for index in sorted(expected):
            plain_record = plain[index]
            graph_record = graph[index]
            for record in (plain_record, graph_record):
                leaked = sorted(FORBIDDEN_SOURCE_FIELDS.intersection(record))
                if leaked:
                    leaked_fields.setdefault(index, []).extend(leaked)

            requests += 2
            if not isinstance(plain_record.get("request"), dict):
                missing_requests += 1
            if not str(plain_record.get("raw", "")).strip():
                missing_responses += 1

            ugcp = graph_record.get("ugcp", {})
            if not isinstance(ugcp.get("graph_request"), dict):
                missing_requests += 1
            if not str(ugcp.get("raw_graph", "")).strip():
                missing_responses += 1
            verifier_called = bool(ugcp.get("verifier", {}).get("called"))
            local_called = bool(ugcp.get("local_verifier", {}).get("called"))
            if verifier_called != local_called:
                raise ValueError(f"{name} verifier routing mismatch at index {index}")
            if verifier_called:
                conditional_pairs += 1
                for key in ("verifier", "local_verifier"):
                    requests += 1
                    branch = ugcp.get(key, {})
                    if not isinstance(branch.get("request"), dict):
                        missing_requests += 1
                    if not str(branch.get("raw", "")).strip():
                        missing_responses += 1

        if leaked_fields:
            raise ValueError(f"{name} decision-only sources contain labels: {leaked_fields}")
        if missing_requests or missing_responses:
            raise ValueError(
                f"{name} incomplete source trace: missing_requests={missing_requests}, "
                f"missing_responses={missing_responses}"
            )

        summaries = []
        for path in (plain_path, graph_path):
            summary_path = path.with_suffix(".summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not summary.get("decision_only"):
                raise ValueError(f"{summary_path} is not decision-only")
            if summary.get("failed_samples") != 0:
                raise ValueError(f"{summary_path} reports failed samples")
            if not summary.get("full_official_evaluation"):
                raise ValueError(f"{summary_path} is not a full evaluation")
            summaries.append(str(summary_path))

        rows.append(
            {
                "variant": name,
                "records": len(expected),
                "requests": requests,
                "conditional_verifier_pairs": conditional_pairs,
                "missing_requests": 0,
                "missing_responses": 0,
                "label_fields_in_source": 0,
                "summary_files": summaries,
            }
        )
        total_requests += requests
        total_records += len(expected)
    return {
        "complete": True,
        "records": total_records,
        "requests": total_requests,
        "missing_requests": 0,
        "missing_responses": 0,
        "label_fields_in_source": 0,
        "datasets": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {METHOD_NAME} Frozen Holdout Results ({report['model']})",
        "",
        "## Accuracy",
        "",
        "| Variant | N | Full ACC | 95% Wilson CI | Majority ACC | Full - majority |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["accuracy"]:
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['full_acc']:.2f} | "
            f"[{row['full_ci95_low']:.2f}, {row['full_ci95_high']:.2f}] | "
            f"{row['majority_acc']:.2f} | {row['full_minus_majority_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## End-to-End Robustness",
            "",
            "| Perturbation | Full prediction invariance | Majority invariance | Full ACC change |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["robustness"]:
        lines.append(
            f"| {row['variant']} | {row['full_prediction_invariance']:.2f} | "
            f"{row['majority_prediction_invariance']:.2f} | "
            f"{row['full_acc_change_pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Intervention Diagnostics",
            "",
            "| Variant | Plain ACC | Full ACC | Changed | Rescued | Harmed | Net rescues |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["interventions"]:
        lines.append(
            f"| {row['variant']} | {row['plain_acc']:.2f} | {row['full_acc']:.2f} | "
            f"{row['changed_predictions']} | {row['rescued']} | {row['harmed']} | "
            f"{row['net_rescues']:+d} |"
        )
    lines.extend(
        [
            "",
            "## Trace Completeness",
            "",
            "| Variant | Records | Requests | Conditional verifier pairs | Missing | Label fields |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["trace_audit"]["datasets"]:
        lines.append(
            f"| {row['variant']} | {row['records']} | {row['requests']} | "
            f"{row['conditional_verifier_pairs']} | "
            f"{row['missing_requests'] + row['missing_responses']} | "
            f"{row['label_fields_in_source']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "- The method and perturbation generators were frozen before model outputs were produced.",
            "- Source generation is decision-only; gold labels are joined during replay after option IDs are fixed.",
            "- McNemar and bootstrap claims are internal paired comparisons, not comparisons against published paper rows.",
            "- No model or dataset-specific threshold is introduced for this holdout.",
        ]
    )
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--permuted", type=Path, required=True)
    parser.add_argument("--distractor", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    paths = {
        "base": args.base,
        "permuted": args.permuted,
        "distractor": args.distractor,
    }
    records = {
        name: [item for _, item in sorted(load_jsonl(path).items())]
        for name, path in paths.items()
    }
    lengths = {len(items) for items in records.values()}
    if len(lengths) != 1:
        raise ValueError("holdout variants have different record counts")

    accuracy = []
    majority_tests = []
    for name in VARIANTS:
        full = correctness(records[name], "full")
        majority = correctness(records[name], "majority_same_budget")
        low, high = wilson_interval(int(full.sum()), len(full))
        accuracy.append(
            {
                "variant": name,
                "n": len(full),
                "full_acc": 100 * float(full.mean()),
                "full_correct": int(full.sum()),
                "full_ci95_low": low,
                "full_ci95_high": high,
                "majority_acc": 100 * float(majority.mean()),
                "full_minus_majority_pp": 100
                * float(full.mean() - majority.mean()),
            }
        )
        majority_tests.append(
            paired_test(
                full,
                majority,
                name=f"{name}: full vs majority_same_budget",
                bootstrap=args.bootstrap,
                seed=stable_seed(args.seed, name, "majority"),
            )
        )
    holm_adjust(majority_tests)

    base_full = correctness(records["base"], "full")
    base_full_predictions = predictions(records["base"], "full")
    base_majority_predictions = predictions(records["base"], "majority_same_budget")
    robustness = []
    robustness_tests = []
    for name in ("permuted", "distractor"):
        changed_full = correctness(records[name], "full")
        full_predictions = predictions(records[name], "full")
        majority_predictions = predictions(records[name], "majority_same_budget")
        robustness.append(
            {
                "variant": name,
                "n": len(changed_full),
                "full_prediction_invariance": 100
                * sum(a == b for a, b in zip(base_full_predictions, full_predictions))
                / len(changed_full),
                "majority_prediction_invariance": 100
                * sum(
                    a == b
                    for a, b in zip(base_majority_predictions, majority_predictions)
                )
                / len(changed_full),
                "full_acc_change_pp": 100
                * float(changed_full.mean() - base_full.mean()),
            }
        )
        robustness_tests.append(
            paired_test(
                changed_full,
                base_full,
                name=f"{name} full vs base full",
                bootstrap=args.bootstrap,
                seed=stable_seed(args.seed, name, "robustness"),
            )
        )
    holm_adjust(robustness_tests)

    interventions = []
    intervention_tests = []
    for name in VARIANTS:
        row = {"variant": name, **intervention_diagnostics(records[name])}
        interventions.append(row)
        intervention_tests.append(
            paired_test(
                correctness(records[name], "full"),
                correctness(records[name], "plain_context"),
                name=f"{name}: full vs plain_context",
                bootstrap=args.bootstrap,
                seed=stable_seed(args.seed, name, "plain_context"),
            )
        )
    holm_adjust(intervention_tests)

    ablation_tests = []
    for variant in VARIANT_ORDER:
        if variant == "full":
            continue
        ablation_tests.append(
            paired_test(
                base_full,
                correctness(records["base"], variant),
                name=f"base: full vs {variant}",
                bootstrap=args.bootstrap,
                seed=stable_seed(args.seed, "base", variant),
            )
        )
    holm_adjust(ablation_tests)

    report = {
        "method": METHOD_NAME,
        "model": args.model,
        "scope": "frozen post-method OpenBookQA official test_complete holdout",
        "training": False,
        "zero_shot": True,
        "workers": 12,
        "gold_used_for_source_generation": False,
        "bootstrap_repetitions": args.bootstrap,
        "accuracy": accuracy,
        "full_vs_majority_tests": majority_tests,
        "robustness": robustness,
        "robustness_tests": robustness_tests,
        "interventions": interventions,
        "intervention_tests": intervention_tests,
        "base_ablation_tests": ablation_tests,
        "trace_audit": source_trace_audit(args.source_dir, records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"accuracy": accuracy, "robustness": robustness}, indent=2))


if __name__ == "__main__":
    run()
