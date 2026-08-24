"""Run zero-shot comparison baselines against a deployed OpenAI-compatible model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .projection import score


WORKERS = 12
METHODS = ("without_rag", "rag", "kre")
METHOD_LABELS = {
    "without_rag": "w/o RAG",
    "rag": "RAG",
    "kre": "KRE",
}
KRE_DEPENDENT_ROLE = (
    "You are now an underage individual who has not received any formal education, "
    "and you have virtually no knowledge or information. You MUST now respond to "
    "questions solely based on the provided context."
)
SYSTEM_PROMPT = (
    "You are a careful multiple-choice question answering assistant. "
    "Return one option or None if the information is insufficient."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: Any) -> str:
    return " ".join(str(text or "").lower().replace("_", " ").split())


def prompt_for(method: str, sample: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{chr(65 + index)}. {choice}"
        for index, choice in enumerate(sample["choices"])
    )
    context = "" if method == "without_rag" else str(sample["context"])
    role = f"{KRE_DEPENDENT_ROLE}\n\n" if method == "kre" else ""
    return (
        f"{role}Context:\n{context}\n\n"
        f"Question: {sample['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Select exactly one choice. Give the answer choice and a short reason."
    )


def request_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int = 4,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"model request failed after {retries} attempts: {last_error}")


def model_metadata(base_url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer EMPTY"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_option(raw: str, choices: list[str]) -> tuple[int | None, str]:
    cleaned = str(raw or "").strip()
    patterns = (
        r"(?i)\boption[_ ]*id\s*[:=]\s*(\d+)",
        r"(?i)\banswer\s*(?:is|:)?\s*([A-Z])\b",
        r"(?i)\b(?:choice|option)\s*[:=]?\s*([A-Z])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        if matches:
            value = matches[-1]
            index = int(value) if value.isdigit() else ord(value.upper()) - ord("A")
            if 0 <= index < len(choices):
                return index, "structured_identifier"

    upper = cleaned.upper()
    for index in range(len(choices) - 1, -1, -1):
        if re.search(rf"(?:^|[^A-Z]){chr(65 + index)}(?:[^A-Z]|$)", upper):
            return index, "answer_letter"

    matches = [
        (index, normalize(choice))
        for index, choice in enumerate(choices)
        if len(normalize(choice)) >= 3 and normalize(choice) in normalize(cleaned)
    ]
    if len(matches) == 1:
        return matches[0][0], "choice_text"
    return None, "unresolved"


def run(args: argparse.Namespace) -> None:
    if args.workers != WORKERS:
        raise ValueError(f"baseline protocol requires --workers {WORKERS}")
    data = json.loads(args.dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("dataset must be a JSON list")
    if args.method not in METHODS:
        raise ValueError(f"unsupported locally runnable method: {args.method}")
    server = model_metadata(args.base_url, args.timeout)

    def process(index: int) -> dict[str, Any]:
        sample = data[index]
        prompt = prompt_for(args.method, sample)
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed + index,
        }
        response = request_json(args.base_url, payload, timeout=args.timeout)
        raw = str(response["choices"][0]["message"].get("content") or "")
        choices = [str(choice) for choice in sample["choices"]]
        option_id, parse_route = parse_option(raw, choices)
        prediction = choices[option_id] if option_id is not None else raw
        metrics = score(prediction, sample["answer"])
        metrics["option_acc"] = float(
            option_id is not None and any(
                normalize(prediction) == normalize(answer)
                for answer in (
                    sample["answer"]
                    if isinstance(sample["answer"], list)
                    else [sample["answer"]]
                )
            )
        )
        return {
            "idx": index,
            "method": args.method,
            "method_label": METHOD_LABELS[args.method],
            "model": args.model,
            "dataset": args.dataset,
            "raw": raw,
            "option_id": option_id,
            "parse_route": parse_route,
            "prediction": prediction,
            "gold": sample["answer"],
            "metrics": metrics,
            "decision_used_gold": False,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(process, range(len(data))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    metrics = {
        metric: 100
        * sum(float(record["metrics"][metric]) for record in records)
        / max(len(records), 1)
        for metric in ("acc", "em", "f1", "option_acc")
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "model": args.model,
        "served_model": server,
        "dataset": args.dataset,
        "dataset_sha256": sha256(args.dataset_path),
        "output_sha256": sha256(args.output),
        "num_samples": len(records),
        "workers": args.workers,
        "zero_shot": True,
        "training": False,
        "decision_used_gold": False,
        "temperature": 0.0,
        "metrics": metrics,
        "unresolved_outputs": sum(record["option_id"] is None for record in records),
        "protocol_note": (
            "KRE is the published dependent-role intervention adapted to the same "
            "zero-shot multiple-choice output protocol."
            if args.method == "kre"
            else "Matched zero-shot prompt baseline."
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
