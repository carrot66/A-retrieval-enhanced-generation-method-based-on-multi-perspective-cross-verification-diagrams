"""Build the dual-edge semantic bridge used by QPEF-RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


LOCAL_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_ENCODER = "cross-encoder/nli-deberta-v3-small"
QUERY_TEMPLATE = "The answer to the question '{question}' is '{option}'."
WORKERS = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_units(context: Any) -> list[str]:
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", str(context))
        if unit.strip()
    ]
    return units or [str(context)]


def validate_sample(sample: dict[str, Any], index: int) -> None:
    missing = {"context", "question", "choices"} - set(sample)
    if missing:
        raise ValueError(f"sample {index} is missing {sorted(missing)}")
    if len(sample["choices"]) < 2:
        raise ValueError(f"sample {index} has fewer than two choices")


def local_alignment_records(
    data: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int,
) -> list[dict[str, Any]]:
    texts: list[str] = []
    metadata: list[tuple[int, list[str], int]] = []
    for index, sample in enumerate(data):
        validate_sample(sample, index)
        units = context_units(sample["context"])
        options = [str(choice) for choice in sample["choices"]]
        start = len(texts)
        texts.extend(units)
        texts.extend(options)
        metadata.append((start, units, len(options)))

    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    records = []
    for start, units, option_count in metadata:
        unit_embeddings = embeddings[start : start + len(units)]
        option_embeddings = embeddings[
            start + len(units) : start + len(units) + option_count
        ]
        similarities = option_embeddings @ unit_embeddings.T
        unit_ids = similarities.argmax(axis=1)
        option_scores = similarities[np.arange(option_count), unit_ids]
        selected_option_id = int(option_scores.argmax())
        selected_unit_id = int(unit_ids[selected_option_id])
        records.append(
            {
                "selected_option_id": selected_option_id,
                "selected_unit_id": selected_unit_id,
                "selected_unit": units[selected_unit_id],
                "option_scores": [round(float(value), 8) for value in option_scores],
            }
        )
    return records


def query_entailment_records(
    data: list[dict[str, Any]],
    model: CrossEncoder,
    batch_size: int,
) -> list[dict[str, Any]]:
    pairs: list[tuple[str, str]] = []
    metadata: list[tuple[int, list[str], int]] = []
    for sample in data:
        units = context_units(sample["context"])
        options = [str(choice) for choice in sample["choices"]]
        start = len(pairs)
        for option in options:
            hypothesis = QUERY_TEMPLATE.format(
                question=str(sample["question"]),
                option=option,
            )
            pairs.extend((unit, hypothesis) for unit in units)
        metadata.append((start, units, len(options)))

    labels = {
        int(index): str(label).lower()
        for index, label in model.model.config.id2label.items()
    }
    entailment_ids = [index for index, label in labels.items() if label == "entailment"]
    if len(entailment_ids) != 1:
        raise ValueError(f"cannot identify entailment label: {labels}")
    with torch.inference_mode():
        probabilities = np.asarray(
            model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=True,
                apply_softmax=True,
                convert_to_numpy=True,
            )
        )
    entailment = probabilities[:, entailment_ids[0]]

    records = []
    for start, units, option_count in metadata:
        unit_count = len(units)
        matrix = entailment[
            start : start + unit_count * option_count
        ].reshape(option_count, unit_count)
        unit_ids = matrix.argmax(axis=1)
        option_scores = matrix[np.arange(option_count), unit_ids]
        selected_option_id = int(option_scores.argmax())
        selected_unit_id = int(unit_ids[selected_option_id])
        records.append(
            {
                "selected_option_id": selected_option_id,
                "selected_unit_id": selected_unit_id,
                "selected_unit": units[selected_unit_id],
                "option_scores": [round(float(value), 8) for value in option_scores],
            }
        )
    return records


def build_bridge(
    *,
    dataset_path: Path,
    dataset_name: str,
    output: Path,
    local_device: str,
    query_device: str,
    local_batch_size: int,
    query_batch_size: int,
) -> None:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset must be a JSON list")
    torch.set_grad_enabled(False)

    local_model = SentenceTransformer(LOCAL_ENCODER, device=local_device)
    local_model.eval()
    for parameter in local_model.parameters():
        parameter.requires_grad_(False)
    local_records = local_alignment_records(data, local_model, local_batch_size)
    del local_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    query_model = CrossEncoder(
        QUERY_ENCODER,
        device=query_device,
        max_length=256,
    )
    query_model.model.eval()
    for parameter in query_model.model.parameters():
        parameter.requires_grad_(False)
    query_records = query_entailment_records(data, query_model, query_batch_size)

    records = [
        {
            "idx": index,
            "local_alignment": local_records[index],
            "query_entailment": query_records[index],
        }
        for index in range(len(data))
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "heterogeneous semantic bridge",
        "dataset": dataset_name,
        "dataset_source": str(dataset_path.resolve()),
        "dataset_sha256": sha256(dataset_path),
        "num_samples": len(records),
        "workers": WORKERS,
        "training": False,
        "gradients_enabled": False,
        "optimizer": None,
        "uses_gold_labels": False,
        "few_shot_examples": 0,
        "local_alignment": {
            "encoder": LOCAL_ENCODER,
            "query_form": "option text only",
            "selection": "argmax option over maximum context-unit cosine",
        },
        "query_entailment": {
            "encoder": QUERY_ENCODER,
            "query_template": QUERY_TEMPLATE,
            "selection": "argmax option over maximum context-unit entailment",
        },
        "artifact_sha256": sha256(output),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "local_device": local_device,
            "query_device": query_device,
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--local_device", default="cpu")
    parser.add_argument("--query_device", default="cuda")
    parser.add_argument("--local_batch_size", type=int, default=256)
    parser.add_argument("--query_batch_size", type=int, default=256)
    args = parser.parse_args()
    if args.workers != WORKERS:
        raise ValueError(f"QPEF-RAG requires --workers {WORKERS}")
    torch.set_num_threads(args.workers)
    build_bridge(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        output=args.output,
        local_device=args.local_device,
        query_device=args.query_device,
        local_batch_size=args.local_batch_size,
        query_batch_size=args.query_batch_size,
    )


if __name__ == "__main__":
    run()
