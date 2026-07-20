#!/usr/bin/env python3
"""Probe whether an explicit-premise critic can prevent unsupported QPCF updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "source_generation"))
import source_runner

from ugcp_v5.projection import load_jsonl, normalize, score_option


PROMPT = """Act as a strict proof-completeness checker, not as a question-answering solver.

Use only propositions explicitly stated in CONTEXT. Treat all outside knowledge and
unstated common-sense bridges as unavailable. The context is sufficient only if every
premise needed to derive exactly one option is explicit. Topic overlap, plausibility,
analogy, or an option being compatible with the context is not a proof. Questions about
an example, consequence, instrument, material, location, or comparison require the
linking premise itself to be stated. If any required link is missing, return NONE.

First identify the relation asked by QUESTION. Then silently test every option. Return
exactly four lines:
SUFFICIENT=<YES or NO>
SUPPORTED_OPTION_ID=<integer or NONE>
QUOTE=<minimal exact context substring or NONE>
MISSING_PREMISE=<short description or NONE>

CONTEXT:
{context}

QUESTION:
{question}

OPTIONS:
{options}
"""

CONTRASTIVE_PROMPT = """Act as an impartial RAG answer adjudicator.

Compare the two anonymous candidate answers for the QUESTION. Neither candidate is the
default and their order is randomized. The CONTEXT defines the local evidence world:
when it explicitly conflicts with ordinary knowledge, follow the CONTEXT. When it gives
only a partial but non-conflicting premise, ordinary knowledge may supply routine bridges.
Reject topical associations that do not answer the exact relation. Select NONE only when
neither candidate can be justified.

Return exactly three lines:
WINNER=<A, B, or NONE>
EVIDENCE_MODE=<EXPLICIT_CONTEXT, CONTEXT_PLUS_KNOWLEDGE, or INSUFFICIENT>
RATIONALE=<one concise sentence>

CONTEXT:
{context}

QUESTION:
{question}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}
"""


DATA_FILES = {
    "faitheval": "faitheval_data.json",
    "musique_golden": "musique_golden.json",
    "musique_negative": "musique_negative.json",
    "squad_golden": "squad_golden.json",
    "squad_negative": "squad_negative.json",
    "timeqa": "timeqa_2022_nota.json",
    "openbookqa_holdout": "openbookqa_test.json",
    "openbookqa_permuted": "openbookqa_test_permuted.json",
    "openbookqa_distractor": "openbookqa_test_distractor.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_response(raw: str, number_of_choices: int) -> dict[str, Any]:
    sufficient_match = re.search(
        r"(?im)^\s*SUFFICIENT\s*=\s*(YES|NO)\s*$", raw
    )
    option_match = re.search(
        r"(?im)^\s*SUPPORTED_OPTION_ID\s*=\s*(\d+|NONE)\s*$", raw
    )
    quote_match = re.search(r"(?im)^\s*QUOTE\s*=\s*(.+?)\s*$", raw)
    missing_match = re.search(
        r"(?im)^\s*MISSING_PREMISE\s*=\s*(.+?)\s*$", raw
    )
    sufficient = bool(
        sufficient_match and sufficient_match.group(1).upper() == "YES"
    )
    option_id = None
    if option_match and option_match.group(1).upper() != "NONE":
        parsed = int(option_match.group(1))
        option_id = parsed if 0 <= parsed < number_of_choices else None
    schema_valid = bool(sufficient_match and option_match and quote_match and missing_match)
    if sufficient != (option_id is not None):
        schema_valid = False
        sufficient = False
        option_id = None
    return {
        "schema_valid": schema_valid,
        "sufficient": sufficient,
        "supported_option_id": option_id,
        "quote": quote_match.group(1).strip() if quote_match else "NONE",
        "missing_premise": (
            missing_match.group(1).strip() if missing_match else "UNPARSEABLE"
        ),
    }


def parse_contrastive_response(raw: str) -> dict[str, Any]:
    winner_match = re.search(
        r"(?im)^\s*WINNER\s*[:=]\s*(A|B|NONE)\s*$", raw
    )
    mode_match = re.search(
        r"(?im)^\s*EVIDENCE_MODE\s*[:=]\s*"
        r"(EXPLICIT_CONTEXT|CONTEXT_PLUS_KNOWLEDGE|INSUFFICIENT)\s*$",
        raw,
    )
    rationale_match = re.search(r"(?im)^\s*RATIONALE\s*[:=]\s*(.+?)\s*$", raw)
    return {
        "schema_valid": bool(winner_match and mode_match and rationale_match),
        "winner": winner_match.group(1).upper() if winner_match else "NONE",
        "evidence_mode": (
            mode_match.group(1).upper() if mode_match else "UNPARSEABLE"
        ),
        "rationale": (
            rationale_match.group(1).strip() if rationale_match else "UNPARSEABLE"
        ),
    }


def run_dataset(
    *,
    root: Path,
    dataset: str,
    model: str,
    base_url: str,
    prompt_format: str,
    workers: int,
    limit: int,
    seed: int,
    output: Path,
    mode: str,
) -> dict[str, Any]:
    data_path = root / "data" / DATA_FILES[dataset]
    data = json.loads(data_path.read_text(encoding="utf-8"))
    frozen_path = root / "artifacts" / "projections" / model / f"{dataset}.jsonl"
    frozen = load_jsonl(frozen_path)
    changed_indices = [
        index
        for index in range(len(data))
        if frozen[index]["variants"]["full"]["option_id"]
        != frozen[index]["variants"]["plain_context"]["option_id"]
    ]
    selected_indices = changed_indices[:limit] if limit else changed_indices

    def probe(index: int) -> dict[str, Any]:
        sample = {
            key: value
            for key, value in data[index].items()
            if key not in {"answer", "answerKey"}
        }
        choices = [str(choice) for choice in sample["choices"]]
        frozen_full = frozen[index]["variants"]["full"]["option_id"]
        frozen_plain = frozen[index]["variants"]["plain_context"]["option_id"]
        if mode == "proof":
            prompt = PROMPT.format(
                context=sample["context"],
                question=sample["question"],
                options=source_runner.format_options(choices),
            )
            full_is_a = None
        else:
            ordering_digest = hashlib.sha256(
                f"{dataset}:{index}".encode("utf-8")
            ).digest()
            full_is_a = ordering_digest[0] % 2 == 0
            candidate_a = choices[frozen_full if full_is_a else frozen_plain]
            candidate_b = choices[frozen_plain if full_is_a else frozen_full]
            prompt = CONTRASTIVE_PROMPT.format(
                context=sample["context"],
                question=sample["question"],
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )
        request = source_runner.completion_payload(
            model, prompt, prompt_format, seed + index, 220
        )
        raw = source_runner.request_completion(
            base_url, model, prompt, prompt_format, seed + index, 220
        )
        if mode == "proof":
            parsed = parse_response(raw, len(choices))
            accepted = bool(
                parsed["schema_valid"]
                and parsed["sufficient"]
                and parsed["supported_option_id"] == frozen_full
            )
            candidate_option_id = frozen_full if accepted else frozen_plain
        else:
            parsed = parse_contrastive_response(raw)
            winner = parsed["winner"]
            accepted = bool(
                parsed["schema_valid"]
                and ((winner == "A") == bool(full_is_a))
                and winner != "NONE"
            )
            candidate_option_id = (
                frozen_full if accepted else frozen_plain
            )
        return {
            "idx": index,
            "request": request,
            "prompt_sha256": text_sha256(prompt),
            "raw": raw,
            "critic": parsed,
            "mode": mode,
            "full_is_candidate_a": full_is_a,
            "frozen_full_option_id": frozen_full,
            "frozen_plain_option_id": frozen_plain,
            "critic_accepts_update": accepted,
            "candidate_option_id": candidate_option_id,
        }

    records: dict[int, dict[str, Any]] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(probe, index): index for index in selected_indices}
        for future in as_completed(futures):
            index = futures[future]
            try:
                records[index] = future.result()
            except Exception as error:
                failures.append({"idx": index, "error": repr(error)})
    if failures:
        raise RuntimeError(f"{dataset} proof probe failures: {failures}")

    scored = []
    for index in sorted(records):
        record = records[index]
        choices = [str(choice) for choice in data[index]["choices"]]
        gold = data[index]["answer"]
        record["scores"] = {
            "frozen_full": score_option(
                record["frozen_full_option_id"], choices, gold
            ),
            "plain": score_option(record["frozen_plain_option_id"], choices, gold),
            "candidate": score_option(record["candidate_option_id"], choices, gold),
        }
        scored.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in scored) + "\n",
        encoding="utf-8",
    )
    n = len(scored)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-hoc development probe",
        "mode": mode,
        "dataset": dataset,
        "model": model,
        "workers": workers,
        "zero_shot": True,
        "training": False,
        "gold_in_request": False,
        "selection": "first changed prediction indices; no gold-based selection",
        "changed_population": len(changed_indices),
        "probed": n,
        "requests": n,
        "failures": 0,
        "critic_accepts": sum(record["critic_accepts_update"] for record in scored),
        "metrics_on_probed_changed_subset": {
            variant: 100
            * sum(record["scores"][variant]["acc"] for record in scored)
            / n
            for variant in ("frozen_full", "plain", "candidate")
        },
        "candidate_rescues_vs_frozen": sum(
            record["scores"]["candidate"]["acc"] == 1
            and record["scores"]["frozen_full"]["acc"] == 0
            for record in scored
        ),
        "candidate_harms_vs_frozen": sum(
            record["scores"]["candidate"]["acc"] == 0
            and record["scores"]["frozen_full"]["acc"] == 1
            for record in scored
        ),
        "prompt_sha256": text_sha256(
            PROMPT if mode == "proof" else CONTRASTIVE_PROMPT
        ),
        "dataset_sha256": sha256(data_path),
        "frozen_projection_sha256": sha256(frozen_path),
        "output_sha256": sha256(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--prompt_format", default="llama")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("proof", "contrastive"), default="proof")
    args = parser.parse_args()
    if args.workers != 12:
        raise ValueError("formal development probes require workers=12")

    root = args.root.resolve()
    summaries = []
    for dataset in args.datasets:
        summaries.append(
            run_dataset(
                root=root,
                dataset=dataset,
                model=args.model,
                base_url=args.base_url,
                prompt_format=args.prompt_format,
                workers=args.workers,
                limit=args.limit,
                seed=args.seed,
                output=args.output_dir / f"{dataset}.jsonl",
                mode=args.mode,
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
