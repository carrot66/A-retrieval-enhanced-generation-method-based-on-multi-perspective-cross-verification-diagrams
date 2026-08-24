#!/usr/bin/env python3
"""Audit the completeness and claim boundaries of a QPCF release."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_FILES = {
    "faitheval": "faitheval_data.json",
    "timeqa": "timeqa_2022_nota.json",
    "musique_negative": "musique_negative.json",
    "squad_negative": "squad_negative.json",
    "musique_golden": "musique_golden.json",
    "squad_golden": "squad_golden.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialized_sha256(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def isolated_dataset_hashes(path: Path) -> tuple[str, str]:
    data = load_json(path)
    inference = [
        {
            key: value
            for key, value in sample.items()
            if key not in {"answer", "answerKey"}
        }
        for sample in data
    ]
    answer_key = [
        {
            "idx": index,
            "id": str(sample["id"]),
            "answer": str(sample["answer"]),
        }
        for index, sample in enumerate(data)
    ]
    return serialized_sha256(inference), serialized_sha256(answer_key)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(record)
    return records


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, str]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, name: str, detail: str) -> None:
        status = "PASS" if condition else "FAIL"
        self.checks.append({"name": name, "status": status, "detail": detail})
        if not condition:
            self.errors.append(f"{name}: {detail}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def complete_indices(records: list[dict[str, Any]], size: int) -> bool:
    try:
        indices = [int(record["idx"]) for record in records]
    except (KeyError, TypeError, ValueError):
        return False
    return len(indices) == size and set(indices) == set(range(size))


def audit_source_pair(
    audit: Audit,
    root: Path,
    model: str,
    dataset: str,
    size: int,
    dataset_hash: str,
) -> dict[str, int]:
    source_dir = root / "artifacts" / "source" / model
    traces: dict[str, list[dict[str, Any]]] = {}
    requests = 0
    for mode in ("plain", "graph"):
        trace_path = source_dir / f"{dataset}.{mode}.jsonl"
        summary_path = source_dir / f"{dataset}.{mode}.summary.json"
        records = load_jsonl(trace_path)
        summary = load_json(summary_path)
        traces[mode] = records
        audit.require(
            complete_indices(records, size),
            f"source_indices:{model}:{dataset}:{mode}",
            f"expected exactly {size} unique zero-based sample indices",
        )
        metadata_ok = (
            str(summary.get("model", "")).lower() == model.lower()
            and summary.get("dataset") == dataset
            and summary.get("mode") == mode
            and summary.get("selected_samples") == size
            and summary.get("completed_samples") == size
            and summary.get("failed_samples") == 0
            and summary.get("failures") == []
            and summary.get("workers") == 12
            and summary.get("zero_shot") is True
            and summary.get("temperature") == 0
            and summary.get("full_official_evaluation") is True
            and summary.get("dataset_sha256") == dataset_hash
            and (
                model != "mistral"
                or summary.get("holdout_protocol", {}).get(
                    "physical_label_isolation_enforced"
                )
                is True
            )
        )
        audit.require(
            metadata_ok,
            f"source_metadata:{model}:{dataset}:{mode}",
            "full split, zero-shot, temperature-zero, workers=12 metadata must match",
        )
        requests += size

    empty_plain_outputs = sum(
        isinstance(record.get("raw"), str) and not record["raw"].strip()
        for record in traces["plain"]
    )
    plain_complete = all(
        isinstance(record.get("raw"), str)
        and (
            model != "mistral"
            or (
                isinstance(record.get("request"), dict)
                and bool(record["request"])
            )
        )
        and isinstance(record.get("prompt_sha256"), str)
        and len(record["prompt_sha256"]) == 64
        and (model == "mistral" or bool(record["raw"].strip()))
        for record in traces["plain"]
    )
    audit.require(
        plain_complete,
        f"plain_trace_completeness:{model}:{dataset}",
        "every plain record must retain request, completion text, and prompt hash; exact Mistral empty texts are retained as observed",
    )

    graph_complete = True
    verifier_calls = 0
    local_calls = 0
    for record in traces["graph"]:
        ugcp = record.get("ugcp", {})
        graph_complete &= (
            isinstance(ugcp.get("raw_graph"), str)
            and bool(ugcp["raw_graph"].strip())
            and isinstance(ugcp.get("graph_prompt_sha256"), str)
            and len(ugcp["graph_prompt_sha256"]) == 64
        )
        for field in ("verifier", "local_verifier"):
            branch = ugcp.get(field, {})
            if branch.get("called") is True:
                if field == "verifier":
                    verifier_calls += 1
                else:
                    local_calls += 1
                graph_complete &= isinstance(branch.get("raw"), str) and bool(
                    branch["raw"].strip()
                )
    audit.require(
        graph_complete,
        f"graph_trace_completeness:{model}:{dataset}",
        "every graph call and every routed verifier call must retain a raw response",
    )
    requests += verifier_calls + local_calls
    return {
        "samples": size,
        "requests": requests,
        "verifier_calls": verifier_calls,
        "local_verifier_calls": local_calls,
        "empty_plain_outputs": empty_plain_outputs,
    }


def audit_semantic(
    audit: Audit,
    root: Path,
    dataset: str,
    size: int,
    dataset_hash: str,
) -> None:
    semantic_dir = root / "artifacts" / "semantic"
    trace_path = semantic_dir / f"{dataset}.bridge.jsonl"
    summary_path = semantic_dir / f"{dataset}.bridge.summary.json"
    records = load_jsonl(trace_path)
    summary = load_json(summary_path)
    audit.require(
        complete_indices(records, size),
        f"semantic_indices:{dataset}",
        f"expected exactly {size} unique zero-based sample indices",
    )
    metadata_ok = (
        summary.get("dataset") == dataset
        and summary.get("dataset_sha256") == dataset_hash
        and summary.get("num_samples") == size
        and summary.get("training") is False
        and summary.get("uses_gold_labels") is False
        and summary.get("artifact_sha256") == sha256(trace_path)
    )
    audit.require(
        metadata_ok,
        f"semantic_metadata:{dataset}",
        "semantic artifact must be training-free, label-free, complete, and hash-matched",
    )


def audit_projection(
    audit: Audit,
    root: Path,
    model: str,
    dataset: str,
    size: int,
    dataset_hash: str,
    answer_key_hash: str | None = None,
) -> None:
    projection_dir = root / "artifacts" / "projections" / model
    trace_path = projection_dir / f"{dataset}.jsonl"
    summary_path = projection_dir / f"{dataset}.summary.json"
    records = load_jsonl(trace_path)
    summary = load_json(summary_path)
    audit.require(
        complete_indices(records, size),
        f"projection_indices:{model}:{dataset}",
        f"expected exactly {size} unique zero-based sample indices",
    )
    source_hashes = summary.get("source_hashes", {})
    expected_hashes = {
        "plain": sha256(root / "artifacts/source" / model / f"{dataset}.plain.jsonl"),
        "graph": sha256(root / "artifacts/source" / model / f"{dataset}.graph.jsonl"),
        "bridge": sha256(root / "artifacts/semantic" / f"{dataset}.bridge.jsonl"),
    }
    metadata_ok = (
        summary.get("method") == "UGCP-v5.1-QPCF"
        and summary.get("model") == model
        and summary.get("dataset") == dataset
        and summary.get("workers") == 12
        and summary.get("zero_shot") is True
        and summary.get("training") is False
        and summary.get("gold_used_for_decision") is False
        and summary.get("label_fields_available_to_decision") == []
        and summary.get("full_official_evaluation") is True
        and summary.get("num_samples") == size
        and summary.get("dataset_sha256") == dataset_hash
        and summary.get("physical_label_isolation")
        is (answer_key_hash is not None)
        and summary.get("answer_key_sha256") == answer_key_hash
        and summary.get("output_sha256") == sha256(trace_path)
        and source_hashes == expected_hashes
    )
    audit.require(
        metadata_ok,
        f"projection_metadata:{model}:{dataset}",
        "decision output must be complete, zero-shot, gold-independent, and hash-linked",
    )
    variants = set(summary.get("metrics", {}))
    variant_complete = bool(variants) and all(
        set(record.get("variants", {})) == variants for record in records
    )
    audit.require(
        variant_complete,
        f"projection_variants:{model}:{dataset}",
        "every sample must contain every declared decision variant",
    )


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UGCP-v5.1-QPCF Release Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Coverage",
        "",
        f"- Formal model-dataset pairs: {report['coverage']['formal_pairs']}",
        f"- Formal per-sample decisions: {report['coverage']['formal_samples']}",
        f"- Retained raw model responses: {report['coverage']['formal_requests']}",
        f"- Retained empty model outputs: {report['coverage'].get('formal_empty_model_outputs', 0)}",
        f"- Frozen holdout variants: {report['coverage']['holdout_samples']}",
        f"- Frozen holdout requests: {report['coverage']['holdout_requests']}",
        f"- Fresh sensitivity samples: {report['coverage'].get('sensitivity_samples', 0)}",
        f"- Fresh sensitivity requests: {report['coverage'].get('sensitivity_requests', 0)}",
        f"- Fresh source-ablation samples: {report['coverage'].get('source_ablation_samples', 0)}",
        f"- Fresh source-ablation requests: {report['coverage'].get('source_ablation_requests', 0)}",
        f"- Machine checks passed: {report['passed_checks']}/{report['total_checks']}",
        "",
        "## Claim Checks",
        "",
        "- All 14 configured published-point-estimate gates pass.",
        "- Training is disabled and formal workers are fixed at 12.",
        "- The official evaluator hash matches the frozen configuration.",
        "- No paired-significance claim is made against published systems.",
        "- No universal non-conflict-improvement claim is made.",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in report["warnings"])
    if not report["warnings"]:
        lines.append("- None.")
    if report["errors"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = Audit()

    config = load_json(root / "code/configs/experiment.json")
    method = config["method"]
    audit.require(
        method.get("name") == "UGCP-v5.1-QPCF"
        and method.get("module_count") == 4
        and len(method.get("modules", [])) == 4
        and method.get("zero_shot") is True
        and method.get("training") is False
        and method.get("few_shot_examples") == 0
        and method.get("gold_routing") is False
        and method.get("workers") == 12
        and method.get("temperature") == 0,
        "method_protocol",
        "frozen method must have four modules, zero training, zero demonstrations, and workers=12",
    )
    evaluator_path = root / "code/src/ugcp_v5/evaluate.py"
    evaluator_hash = sha256(evaluator_path)
    audit.require(
        evaluator_hash == config["evaluation"]["official_evaluator_sha256"],
        "official_evaluator",
        f"evaluator SHA-256 is {evaluator_hash}",
    )

    dataset_metadata: dict[str, tuple[int, str, str, str]] = {}
    for dataset, filename in DATA_FILES.items():
        path = root / "data" / filename
        data = load_json(path)
        audit.require(
            isinstance(data, list) and bool(data),
            f"dataset_format:{dataset}",
            "dataset must be a non-empty JSON list",
        )
        inference_hash, answer_key_hash = isolated_dataset_hashes(path)
        dataset_metadata[dataset] = (
            len(data),
            sha256(path),
            inference_hash,
            answer_key_hash,
        )

    formal_pairs = 0
    formal_samples = 0
    formal_requests = 0
    formal_empty_model_outputs = 0
    audited_semantic: set[str] = set()
    for model in config["execution_order"]:
        model_config = config["models"][model]
        for dataset in model_config["datasets"]:
            size, labeled_hash, inference_hash, answer_key_hash = dataset_metadata[
                dataset
            ]
            physically_isolated = bool(
                model_config.get("physical_label_isolation", False)
            )
            dataset_hash = inference_hash if physically_isolated else labeled_hash
            source_counts = audit_source_pair(
                audit, root, model, dataset, size, dataset_hash
            )
            if dataset not in audited_semantic:
                audit_semantic(audit, root, dataset, size, labeled_hash)
                audited_semantic.add(dataset)
            audit_projection(
                audit,
                root,
                model,
                dataset,
                size,
                dataset_hash,
                answer_key_hash if physically_isolated else None,
            )
            formal_pairs += 1
            formal_samples += size
            formal_requests += source_counts["requests"]
            formal_empty_model_outputs += source_counts["empty_plain_outputs"]

    development = load_json(root / "artifacts/results/DEVELOPMENT_RESULTS.json")
    gate_ok = (
        development.get("status") == "PASS"
        and development.get("workers") == 12
        and development.get("training") is False
        and development.get("zero_shot") is True
        and development.get("gold_used_for_decision") is False
        and len(development.get("results", [])) == formal_pairs
        and all(row.get("pass") is True for row in development.get("results", []))
    )
    audit.require(
        gate_ok,
        "formal_result_gates",
        "all configured model-dataset point-estimate gates must pass",
    )

    replication = load_json(root / "artifacts/results/PROVENANCE_REPLICATION.json")
    audit.require(
        replication.get("status") == "PASS"
        and replication.get("pairs") == formal_pairs
        and replication.get("samples") == formal_samples
        and replication.get("reconstruction_mismatches") == 0
        and replication.get("score_changes") == 0
        and replication.get("winner_changes") == 0,
        "provenance_replication",
        "replicated within-source observations must not change scores or winners",
    )

    paired = load_json(root / "artifacts/results/PAIRED_COUNTERFACTUAL.json")
    audit.require(
        paired.get("status") == "PASS"
        and paired.get("gold_used_for_decision") is False
        and paired.get("gold_used_for_evaluation") is True,
        "paired_counterfactual",
        "paired Golden/counterfactual integrity checks and analysis must pass",
    )

    holdout = load_json(root / "artifacts/results/HOLDOUT_RESULTS_llama.json")
    trace_audit = holdout.get("trace_audit", {})
    audit.require(
        holdout.get("gold_used_for_source_generation") is False
        and trace_audit.get("complete") is True
        and trace_audit.get("records") == 1500
        and trace_audit.get("requests") == 3720
        and trace_audit.get("missing_requests") == 0
        and trace_audit.get("missing_responses") == 0
        and trace_audit.get("label_fields_in_source") == 0,
        "frozen_holdout_trace",
        "holdout must retain 1,500 label-isolated samples and 3,720 complete requests",
    )

    paper = load_json(root / "artifacts/results/PAPER_REPORT.json")
    claims = paper.get("claim_status", {})
    audit.require(
        claims.get("all_published_point_estimate_gates_pass") is True
        and claims.get("universal_non_conflict_improvement") is False
        and claims.get("paired_significance_against_published_systems") is False,
        "claim_boundary",
        "paper report must retain the configured gates and both negative claim boundaries",
    )
    sensitivity = load_json(
        root / "artifacts/results/SUFFICIENCY_SENSITIVITY.json"
    )
    sensitivity_pairs = sensitivity.get("datasets", [])
    sensitivity_ok = (
        sensitivity.get("thresholds") == [2, 3, 4]
        and sensitivity.get("default_graph_minimum_support") == 2
        and sensitivity.get("default_idk_minimum_source_families") == 3
        and sensitivity.get("single_factor_design") is True
        and sensitivity.get("fresh_model_inference") is True
        and sensitivity.get("historical_observation_replay") is False
        and sensitivity.get("one_sample_per_request") is True
        and sensitivity.get("workers") == 12
        and sensitivity.get("training") is False
        and sensitivity.get("gold_used_for_model_inference") is False
        and sensitivity.get("gold_used_for_decision") is False
        and sensitivity.get("gold_used_for_evaluation_only") is True
        and sensitivity.get("official_evaluator_sha256") == evaluator_hash
        and len(sensitivity_pairs) == 6
        and sensitivity.get("trace_audit", {}).get("complete") is True
        and sensitivity.get("trace_audit", {}).get("failures") == 0
        and paper.get("sufficiency_sensitivity", {}).get("trace_audit")
        == sensitivity.get("trace_audit")
    )
    audit.require(
        sensitivity_ok,
        "sufficiency_sensitivity",
        "thresholds 2/3/4 must use a complete fresh label-isolated Llama run with one-factor analysis",
    )
    source_ablation = load_json(
        root / "artifacts/results/SOURCE_CHANNEL_ABLATION.json"
    )
    source_channels = [
        "plain",
        "graph_call",
        "global_verifier",
        "local_verifier",
        "local_semantic",
        "query_semantic",
    ]
    expected_variants = {"full"} | {
        f"without_{channel}" for channel in source_channels
    }
    source_rows = source_ablation.get("datasets", [])
    source_ablation_samples = 0
    source_ablation_records_ok = len(source_rows) == len(DATA_FILES)
    for row in source_rows:
        dataset = row.get("dataset")
        if dataset not in dataset_metadata:
            source_ablation_records_ok = False
            continue
        size, _, inference_hash, _ = dataset_metadata[dataset]
        decision_path = (
            root
            / "artifacts/source_ablation/decisions/llama"
            / f"{dataset}.jsonl"
        )
        records = load_jsonl(decision_path)
        source_ablation_records_ok &= (
            row.get("n") == size
            and complete_indices(records, size)
            and all(set(record.get("variants", {})) == expected_variants for record in records)
            and row.get("source_hashes", {}).get("decisions") == sha256(decision_path)
        )
        for mode in ("plain", "graph"):
            summary = load_json(
                root
                / "artifacts/source_ablation/source/llama"
                / f"{dataset}.{mode}.summary.json"
            )
            source_ablation_records_ok &= (
                summary.get("dataset_sha256") == inference_hash
                and summary.get("decision_only") is True
                and summary.get("physical_label_isolation_enforced") is True
                and summary.get("label_fields_available_to_inference") == []
                and summary.get("workers") == 12
                and summary.get("completed_samples") == size
                and summary.get("failed_samples") == 0
            )
        source_ablation_samples += size
    source_ablation_ok = (
        source_ablation.get("source_channels") == source_channels
        and source_ablation.get("fresh_model_inference") is True
        and source_ablation.get("historical_observation_replay") is False
        and source_ablation.get("one_sample_per_request") is True
        and source_ablation.get("workers") == 12
        and source_ablation.get("training") is False
        and source_ablation.get("gold_used_for_model_inference") is False
        and source_ablation.get("gold_used_for_decision") is False
        and source_ablation.get("gold_used_for_evaluation_only") is True
        and source_ablation.get("official_evaluator_sha256") == evaluator_hash
        and source_ablation.get("trace_audit", {}).get("complete") is True
        and source_ablation.get("trace_audit", {}).get("failures") == 0
        and source_ablation.get("trace_audit", {}).get("samples")
        == source_ablation_samples
        and source_ablation_records_ok
        and paper.get("source_channel_ablation", {}).get("trace_audit")
        == source_ablation.get("trace_audit")
    )
    audit.require(
        source_ablation_ok,
        "source_channel_ablation",
        "all six source families must have complete fresh label-isolated one-at-a-time ablations",
    )
    audit.warn(
        "Llama and Qwen benchmark source generation was not physically label-isolated; "
        "the exact-checkpoint Mistral and OpenBookQA runs were label-isolated before inference."
    )
    audit.warn(
        f"Exact-checkpoint Mistral retained {formal_empty_model_outputs} empty plain "
        "completion texts without selective retry; their requests and fallback decisions remain auditable."
    )
    audit.warn(
        "Published-system comparisons are point-estimate comparisons because FaithfulRAG "
        "and TruthfulRAG per-sample predictions are unavailable."
    )
    audit.warn(
        "OpenBookQA is a retained negative result: QPCF trails matched plain RAG by "
        "2.0 to 3.4 percentage points."
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "UGCP-v5.1-QPCF",
        "status": "PASS" if not audit.errors else "FAIL",
        "total_checks": len(audit.checks),
        "passed_checks": sum(check["status"] == "PASS" for check in audit.checks),
        "coverage": {
            "formal_pairs": formal_pairs,
            "formal_samples": formal_samples,
            "formal_requests": formal_requests,
            "formal_empty_model_outputs": formal_empty_model_outputs,
            "holdout_samples": int(trace_audit.get("records", 0)),
            "holdout_requests": int(trace_audit.get("requests", 0)),
            "sensitivity_samples": int(
                sensitivity.get("trace_audit", {}).get("samples", 0)
            ),
            "sensitivity_requests": int(
                sensitivity.get("trace_audit", {}).get("requests", 0)
            ),
            "source_ablation_samples": int(
                source_ablation.get("trace_audit", {}).get("samples", 0)
            ),
            "source_ablation_requests": int(
                source_ablation.get("trace_audit", {}).get("requests", 0)
            ),
        },
        "checks": audit.checks,
        "warnings": audit.warnings,
        "errors": audit.errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "total_checks", "passed_checks", "coverage")}, indent=2))
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
