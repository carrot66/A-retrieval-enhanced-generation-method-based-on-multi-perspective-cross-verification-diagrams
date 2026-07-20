"""Unified Graph-Certified Projection with candidate-blind verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TRUTHFULRAG_ROOT = Path(
    os.environ.get(
        "TRUTHFULRAG_ROOT",
        "/root/user/cs_tcci_fengzihuan/TruthfulRAG-main/TruthfulRAG-main",
    )
)
DATA_ROOT = TRUTHFULRAG_ROOT / "datas"
DATA_PATHS = {
    "faitheval": DATA_ROOT / "faitheval_data.json",
    "musique_golden": DATA_ROOT / "musique_golden.json",
    "musique_negative": DATA_ROOT / "musique_negative.json",
    "squad_golden": DATA_ROOT / "squad_golden.json",
    "squad_negative": DATA_ROOT / "squad_negative.json",
    "timeqa": DATA_ROOT / "timeqa_2022_nota.json",
}

EVALUATE_PATH = TRUTHFULRAG_ROOT / "truthfulrag" / "evaluate.py"
EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "truthfulrag_official_evaluate", EVALUATE_PATH
)
if EVALUATE_SPEC is None or EVALUATE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load TruthfulRAG evaluator: {EVALUATE_PATH}")
TRUTHFULRAG_EVALUATE = importlib.util.module_from_spec(EVALUATE_SPEC)
EVALUATE_SPEC.loader.exec_module(TRUTHFULRAG_EVALUATE)

METHOD_NAME = "UGCP-v1"
IDK_NORMALS = {
    "i dont know",
    "i don t know",
    "i m not sure",
    "not sure",
    "none of above",
    "none of the above",
    "unknown",
    "not applicable",
}
VARIANT_ORDER = [
    "full",
    "global_verifier_only",
    "with_directional_gate",
    "without_exact_quote",
    "without_blind_recovery",
    "without_verifier",
    "without_abstention",
    "without_projection",
    "context_only",
]

GRAPH_PROMPT = """Build one query-conditioned evidence graph for the question below.

Operational rule: the CONTEXT defines a hypothetical evidence world. Follow what the
CONTEXT supports even when it conflicts with real-world knowledge. Do not repair,
reject, or reinterpret the CONTEXT using outside facts. If the CONTEXT is irrelevant
or insufficient and an explicit "I don't know" option exists, select that option.

Create these graph nodes in one pass and in this reasoning order:
1. Lock BASE as the option a normal context-aware solver initially selects.
2. Temporarily ignore BASE and all outside knowledge. Audit every option against the
   entire CONTEXT before selecting CONTEXT. Do not stop at the first topically related
   sentence. Prefer the author's explicit question-specific assertion or final conclusion
   over an earlier generic statement. Contrastive words such as however, instead,
   therefore, and in summary can reverse the intended edge direction.
3. Only after CONTEXT is fixed, create PRIOR from parametric knowledge while ignoring
   the CONTEXT. PRIOR must never alter CONTEXT.

For the CONTEXT path, copy one complete minimal clause for a direct path or exactly two
complete minimal clauses for a multi-hop path. The quote must express the distinguishing
content of the selected option rather than merely mention the question topic. Every quote
must be exact verbatim text from the CONTEXT. Use QUOTE_2=NONE for a direct path. Use NONE
when no path exists.

Return exactly these six lines and no explanation:
BASE_OPTION_ID=<integer or NONE>
CONTEXT_OPTION_ID=<integer or NONE>
PATH_TYPE=<DIRECT or MULTIHOP or NONE>
QUOTE_1=<exact verbatim clause or NONE>
QUOTE_2=<exact verbatim clause or NONE>
PRIOR_OPTION_ID=<integer or NONE>

CONTEXT:
{context}

QUESTION:
{question}

OPTIONS:
{options}
"""

BLIND_VERIFY_PROMPT = """Independently audit and recover an answer from an evidence graph.

The CONTEXT is interpreted as a hypothetical world, even if it conflicts with outside
knowledge. Do not correct it with common sense. You are intentionally not shown any
candidate answer selected by another solver. Audit the proposed evidence, search the full
CONTEXT, and independently recover the option supported by the shortest relation-complete
path. Copy your own exact clause from the CONTEXT; do not merely repeat a proposed quote
when it is incomplete. Return NONE if no relation-complete path exists. Perform reasoning
silently.

Topical overlap is not sufficient. If the CONTEXT does not contain the entity, event,
time, or relation needed to answer the QUESTION, return NONE even when it discusses a
similar subject.

Before deciding, audit every option against the entire CONTEXT. Do not stop at the first
topically related sentence. Prefer the author's explicit question-specific assertion or
final conclusion over an earlier generic statement. Contrastive words such as however,
instead, therefore, and in summary can reverse the intended relation. The proposed path
is only a retrieval hint and must not anchor your recovered option.

Return exactly:
RECOVERED_OPTION_ID=<integer or NONE>
ENTAILMENT=<DIRECT or MULTIHOP or NONE>
VERIFIED_QUOTE_1=<exact verbatim clause or NONE>
VERIFIED_QUOTE_2=<exact verbatim clause or NONE>

CONTEXT:
{context}

QUESTION:
{question}

OPTIONS:
{options}

PROPOSED EVIDENCE PATH:
{quotes}
"""

LOCAL_VERIFY_PROMPT = """Recover an option using only the proposed evidence path.

Interpret the evidence inside its hypothetical world even when it conflicts with outside
knowledge. You are intentionally not shown any candidate answer from another solver.
Do not use common sense to repair the evidence. Recover the option only when the quoted
path directly answers the question relation; otherwise return NONE. Perform reasoning
silently.

Topical overlap or a shared entity without the asked relation is not sufficient evidence.

Return exactly:
RECOVERED_OPTION_ID=<integer or NONE>
ENTAILMENT=<DIRECT or MULTIHOP or NONE>

QUESTION:
{question}

OPTIONS:
{options}

PROPOSED EVIDENCE PATH:
{quotes}
"""

PLAIN_PROMPT = """Answer the multiple-choice question using the CONTEXT as the evidence
world. Follow what the CONTEXT supports even if it conflicts with outside knowledge.
If the CONTEXT is irrelevant or insufficient and an explicit "I don't know" option
exists, select that option. Return exactly one line: OPTION_ID=<integer or NONE>.

CONTEXT:
{context}

QUESTION:
{question}

OPTIONS:
{options}
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


def normalize(text: Any) -> str:
    return TRUTHFULRAG_EVALUATE.normalize_answer(str(text or ""))


def canonical_verbatim(text: Any) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    return re.sub(r"\s+", " ", normalized).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score(prediction: str, gold: str | list[str]) -> dict[str, float]:
    exact_match = TRUTHFULRAG_EVALUATE.metric_max_over_ground_truths(
        TRUTHFULRAG_EVALUATE.exact_match_score, prediction, gold
    )
    accuracy = TRUTHFULRAG_EVALUATE.metric_max_over_ground_truths(
        TRUTHFULRAG_EVALUATE.acc_score, prediction, gold
    )
    f1 = TRUTHFULRAG_EVALUATE.metric_max_over_ground_truths(
        TRUTHFULRAG_EVALUATE.f1_score, prediction, gold
    )
    return {
        "em": float(exact_match),
        "acc": float(accuracy),
        "f1": float(f1),
    }


def option_id(prediction: str, choices: list[str]) -> int | None:
    normalized_prediction = normalize(prediction)
    for index, choice in enumerate(choices):
        if normalize(choice) == normalized_prediction:
            return index
    return None


def literal_option_anchor(option: str, context: str) -> bool:
    normalized_option = normalize(option)
    normalized_context = normalize(context)
    if not normalized_option:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_option)}(?![a-z0-9])",
            normalized_context,
        )
    )


def gold_option_ids(gold: str | list[str], choices: list[str]) -> set[int]:
    answers = gold if isinstance(gold, list) else [gold]
    normalized_answers = {normalize(answer) for answer in answers}
    return {
        index
        for index, choice in enumerate(choices)
        if normalize(choice) in normalized_answers
    }


def resolve_prompt_format(model: str, prompt_format: str) -> str:
    if prompt_format != "auto":
        return prompt_format
    model_name = model.lower()
    if "qwen" in model_name:
        return "qwen"
    if "mistral" in model_name:
        return "mistral"
    return "llama"


def wrap_prompt(prompt: str, prompt_format: str) -> str:
    if prompt_format == "llama":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if prompt_format == "qwen":
        return (
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    return f"<s>[INST] {prompt.strip()} [/INST]"


def endpoint_metadata(base_url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def request_completion(
    base_url: str,
    model: str,
    prompt: str,
    prompt_format: str,
    seed: int,
    max_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": wrap_prompt(prompt, prompt_format),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["text"])
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def format_options(choices: list[str]) -> str:
    return "\n".join(f"{index}: {choice}" for index, choice in enumerate(choices))


def parse_identifier(raw: str, field: str, number_of_choices: int) -> int | None:
    matches = re.findall(
        rf"^\s*{re.escape(field)}\s*=\s*(\d+|NONE)\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches or matches[-1].upper() == "NONE":
        return None
    value = int(matches[-1])
    return value if 0 <= value < number_of_choices else None


def parse_graph(raw: str, number_of_choices: int) -> dict[str, Any]:
    path_matches = re.findall(
        r"^\s*PATH_TYPE\s*=\s*(DIRECT|MULTIHOP|NONE)\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    path_type = path_matches[-1].upper() if path_matches else "NONE"
    quote_matches = re.findall(
        r"^\s*QUOTE_(\d+)\s*=\s*(.*?)\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    quotes_by_index: dict[int, str] = {}
    for quote_index, quote in quote_matches:
        cleaned = quote.strip().strip('"').strip("'")
        if cleaned and cleaned.upper() != "NONE":
            quotes_by_index[int(quote_index)] = cleaned
    return {
        "base_option_id": parse_identifier(raw, "BASE_OPTION_ID", number_of_choices),
        "prior_option_id": parse_identifier(raw, "PRIOR_OPTION_ID", number_of_choices),
        "context_option_id": parse_identifier(raw, "CONTEXT_OPTION_ID", number_of_choices),
        "path_type": path_type,
        "quotes": [quotes_by_index[index] for index in sorted(quotes_by_index)],
    }


def parse_blind_verifier(raw: str, number_of_choices: int) -> dict[str, Any]:
    entailment_matches = re.findall(
        r"^\s*ENTAILMENT\s*=\s*(DIRECT|MULTIHOP|NONE)\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    quote_matches = re.findall(
        r"^\s*VERIFIED_QUOTE_(\d+)\s*=\s*(.*?)\s*$",
        raw,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    quotes_by_index: dict[int, str] = {}
    for quote_index, quote in quote_matches:
        cleaned = quote.strip().strip('"').strip("'")
        if cleaned and cleaned.upper() != "NONE":
            quotes_by_index[int(quote_index)] = cleaned
    return {
        "recovered_option_id": parse_identifier(
            raw, "RECOVERED_OPTION_ID", number_of_choices
        ),
        "entailment": entailment_matches[-1].upper() if entailment_matches else "NONE",
        "quotes": [quotes_by_index[index] for index in sorted(quotes_by_index)],
    }


def parse_plain(raw: str, number_of_choices: int) -> int | None:
    return parse_identifier(raw, "OPTION_ID", number_of_choices)


def quote_diagnostics(
    context: str,
    path_type: str,
    quotes: list[str],
) -> dict[str, Any]:
    canonical_context = canonical_verbatim(context)
    rows = []
    for quote in quotes:
        token_count = len(re.findall(r"[A-Za-z0-9]+", quote))
        exact = bool(quote) and canonical_verbatim(quote) in canonical_context
        rows.append(
            {
                "quote": quote,
                "exact": exact,
                "token_count": token_count,
                "complete_clause": token_count >= 4,
            }
        )
    required_quotes = 2 if path_type == "MULTIHOP" else 1
    path_declared = path_type in {"DIRECT", "MULTIHOP"}
    compliant = (
        path_declared
        and len(rows) >= required_quotes
        and all(row["exact"] and row["complete_clause"] for row in rows)
    )
    return {
        "path_declared": path_declared,
        "required_quotes": required_quotes,
        "quotes": rows,
        "exact_quote_compliant": compliant,
    }


def fallback_option_id(
    base_option_id: int | None,
    context_option_id: int | None,
    prior_option_id: int | None,
    choices: list[str],
) -> int:
    for candidate in (base_option_id, context_option_id, prior_option_id):
        if candidate is not None:
            return candidate
    for index, choice in enumerate(choices):
        if normalize(choice) in IDK_NORMALS:
            return index
    return 0


def finalize_graph(
    graph: dict[str, Any],
    verifier: dict[str, Any],
    local_verifier: dict[str, Any],
    graph_quote_check: dict[str, Any],
    verifier_quote_check: dict[str, Any],
    choices: list[str],
) -> dict[str, Any]:
    base_option_id = fallback_option_id(
        graph.get("base_option_id"),
        graph.get("context_option_id"),
        graph.get("prior_option_id"),
        choices,
    )
    context_option_id = graph.get("context_option_id")
    prior_option_id = graph.get("prior_option_id")
    candidate_disagreement = (
        context_option_id is not None and base_option_id != context_option_id
    )
    generated_base_option_id = graph.get("generated_base_option_id")
    prior_anchor_option_id = (
        prior_option_id
        if prior_option_id is not None
        else generated_base_option_id
    )
    directional_conflict = (
        context_option_id is not None
        and prior_anchor_option_id is not None
        and context_option_id != prior_anchor_option_id
    )
    base_prior_agreement = (
        prior_anchor_option_id is not None
        and base_option_id == prior_anchor_option_id
    )
    directional_gate = directional_conflict and base_prior_agreement
    recovered_option_id = verifier.get("recovered_option_id")
    blind_recovery = (
        context_option_id is not None
        and recovered_option_id == context_option_id
        and verifier.get("entailment") in {"DIRECT", "MULTIHOP"}
    )
    local_recovered_option_id = local_verifier.get("recovered_option_id")
    local_blind_recovery = (
        context_option_id is not None
        and local_recovered_option_id == context_option_id
        and local_verifier.get("entailment") in {"DIRECT", "MULTIHOP"}
    )
    graph_exact_quote = bool(graph_quote_check.get("exact_quote_compliant"))
    verifier_exact_quote = bool(verifier_quote_check.get("exact_quote_compliant"))
    path_declared = bool(graph_quote_check.get("path_declared"))
    global_certificate = blind_recovery and verifier_exact_quote
    local_certificate = local_blind_recovery and graph_exact_quote
    dual_scope_certificate = global_certificate or local_certificate
    predicates = {
        "candidate_disagreement": candidate_disagreement,
        "directional_conflict": directional_conflict,
        "base_prior_agreement": base_prior_agreement,
        "directional_gate": directional_gate,
        "exact_quote": verifier_exact_quote,
        "graph_exact_quote": graph_exact_quote,
        "verifier_exact_quote": verifier_exact_quote,
        "blind_recovery": blind_recovery,
        "local_blind_recovery": local_blind_recovery,
        "global_certificate": global_certificate,
        "local_certificate": local_certificate,
        "dual_scope_certificate": dual_scope_certificate,
        "path_declared": path_declared,
    }
    idk_option_id = next(
        (
            index
            for index, choice in enumerate(choices)
            if normalize(choice) in IDK_NORMALS
        ),
        None,
    )
    abstention_consensus = (
        idk_option_id is not None
        and context_option_id is not None
        and context_option_id != idk_option_id
        and base_option_id == context_option_id
        and local_blind_recovery
        and bool(graph.get("context_option_literal_anchor"))
    )
    predicates["abstention_consensus"] = abstention_consensus
    if idk_option_id is not None:
        project_full = (
            context_option_id is not None
            and context_option_id != idk_option_id
            and (global_certificate or abstention_consensus)
        )
        project_global_only = (
            context_option_id is not None
            and context_option_id != idk_option_id
            and global_certificate
        )
    else:
        project_full = candidate_disagreement and dual_scope_certificate
        project_global_only = candidate_disagreement and global_certificate
    project_with_direction = all(
        predicates[key]
        for key in (
            "candidate_disagreement",
            "directional_gate",
        )
    ) and (
        global_certificate and local_certificate
        if idk_option_id is not None
        else dual_scope_certificate
    )
    project_without_exact = all(
        predicates[key]
        for key in (
            "candidate_disagreement",
            "path_declared",
        )
    ) and (blind_recovery or local_blind_recovery)
    project_without_blind = all(
        predicates[key]
        for key in (
            "candidate_disagreement",
            "graph_exact_quote",
        )
    )
    project_without_verifier = all(
        predicates[key]
        for key in ("candidate_disagreement", "path_declared")
    )
    project_without_abstention = candidate_disagreement and dual_scope_certificate

    def projected(condition: bool, use_abstention: bool = True) -> int:
        if condition and context_option_id is not None:
            return context_option_id
        if use_abstention and idk_option_id is not None:
            return idk_option_id
        return base_option_id

    variant_ids = {
        "full": projected(project_full),
        "global_verifier_only": projected(project_global_only),
        "with_directional_gate": projected(project_with_direction),
        "without_exact_quote": projected(project_without_exact),
        "without_blind_recovery": projected(project_without_blind),
        "without_verifier": projected(project_without_verifier),
        "without_abstention": projected(
            project_without_abstention, use_abstention=False
        ),
        "without_projection": base_option_id,
        "context_only": (
            context_option_id if context_option_id is not None else base_option_id
        ),
    }
    return {
        "base_option_id": base_option_id,
        "context_option_id": context_option_id,
        "prior_option_id": prior_option_id,
        "prior_anchor_option_id": prior_anchor_option_id,
        "prior_anchor_fallback_to_probe": (
            prior_option_id is None and generated_base_option_id is not None
        ),
        "recovered_option_id": recovered_option_id,
        "local_recovered_option_id": local_recovered_option_id,
        "idk_option_id": idk_option_id,
        "abstained": variant_ids["full"] == idk_option_id,
        "predicates": predicates,
        "full_certified": project_full,
        "variant_option_ids": variant_ids,
    }


def score_option(
    selected_option_id: int,
    choices: list[str],
    gold: str | list[str],
) -> dict[str, Any]:
    prediction = choices[selected_option_id]
    metrics = score(prediction, gold)
    metrics["option_acc"] = float(selected_option_id in gold_option_ids(gold, choices))
    return {
        "option_id": selected_option_id,
        "prediction": prediction,
        **metrics,
    }


def selected_indices(
    number_of_samples: int,
    num_samples: int,
    sample_offset: int,
    sample_stride: int,
) -> list[int]:
    indices = list(range(sample_offset, number_of_samples, sample_stride))
    return indices if num_samples < 0 else indices[:num_samples]


def load_completed(output: Path) -> dict[int, dict[str, Any]]:
    if not output.exists():
        return {}
    completed = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        completed[int(record["idx"])] = record
    return completed


def aggregate_graph(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    denominator = max(len(records), 1)
    for variant in VARIANT_ORDER:
        result[variant] = {
            metric: round(
                100
                * sum(record["variants"][variant][metric] for record in records)
                / denominator,
                4,
            )
            for metric in ("acc", "em", "f1", "option_acc")
        }
    result["full"].update(
        {
            "changed_predictions": sum(
                record["variants"]["full"]["option_id"]
                != record["ugcp"]["base_option_id"]
                for record in records
            ),
            "certified_projections": sum(
                bool(record["ugcp"]["full_certified"])
                and record["variants"]["full"]["option_id"]
                != record["ugcp"]["base_option_id"]
                for record in records
            ),
            "rescues": sum(
                record["variants"]["without_projection"]["acc"] == 0
                and record["variants"]["full"]["acc"] == 1
                for record in records
            ),
            "harms": sum(
                record["variants"]["without_projection"]["acc"] == 1
                and record["variants"]["full"]["acc"] == 0
                for record in records
            ),
        }
    )
    return result


def aggregate_plain(records: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = max(len(records), 1)
    return {
        metric: round(
            100 * sum(record["metrics"][metric] for record in records) / denominator,
            4,
        )
        for metric in ("acc", "em", "f1", "option_acc")
    }


def run() -> None:
    parser = argparse.ArgumentParser(description=METHOD_NAME)
    parser.add_argument("--dataset", choices=list(DATA_PATHS), required=True)
    parser.add_argument("--mode", choices=["graph", "plain"], default="graph")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base_input", type=Path)
    parser.add_argument("--base_url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--model", default="llama")
    parser.add_argument(
        "--prompt_format",
        choices=["auto", "llama", "qwen", "mistral"],
        default="auto",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    if args.workers != 12:
        raise ValueError("Formal UGCP runs require --workers 12")
    prompt_format = resolve_prompt_format(args.model, args.prompt_format)
    data_path = DATA_PATHS[args.dataset]
    data = json.loads(data_path.read_text(encoding="utf-8"))
    indices = selected_indices(
        len(data), args.num_samples, args.sample_offset, args.sample_stride
    )
    if args.no_resume and args.output.exists():
        args.output.unlink()
    completed = load_completed(args.output)
    completed = {index: record for index, record in completed.items() if index in indices}
    pending = [index for index in indices if index not in completed]
    base_records = load_completed(args.base_input) if args.base_input else {}
    if args.mode == "graph" and args.base_input:
        missing_base = sorted(set(indices) - set(base_records))
        if missing_base:
            raise ValueError(
                f"base_input is missing {len(missing_base)} selected indices; "
                f"first missing index={missing_base[0]}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    server_metadata = endpoint_metadata(args.base_url)
    source_path = Path(__file__).resolve()
    LOGGER.info(
        "start method=%s mode=%s model=%s dataset=%s selected=%d pending=%d workers=%d",
        METHOD_NAME,
        args.mode,
        args.model,
        args.dataset,
        len(indices),
        len(pending),
        args.workers,
    )
    LOGGER.info(
        "source_sha256=%s dataset_sha256=%s graph_prompt_sha256=%s blind_prompt_sha256=%s local_prompt_sha256=%s",
        sha256(source_path),
        sha256(data_path),
        text_sha256(GRAPH_PROMPT),
        text_sha256(BLIND_VERIFY_PROMPT),
        text_sha256(LOCAL_VERIFY_PROMPT),
    )
    LOGGER.info("server_metadata=%s", json.dumps(server_metadata, ensure_ascii=False))
    start_time = time.time()

    def evaluate_graph(index: int) -> dict[str, Any]:
        sample = data[index]
        question = str(sample["question"])
        context = str(sample["context"])
        choices = [str(choice) for choice in sample["choices"]]
        options = format_options(choices)
        graph_prompt = GRAPH_PROMPT.format(
            context=context,
            question=question,
            options=options,
        )
        raw_graph = request_completion(
            args.base_url,
            args.model,
            graph_prompt,
            prompt_format,
            args.seed + 2 * index,
            700,
        )
        graph = parse_graph(raw_graph, len(choices))
        generated_base_option_id = graph.get("base_option_id")
        if args.base_input:
            graph["base_option_id"] = int(base_records[index]["option_id"])
        graph["generated_base_option_id"] = generated_base_option_id
        context_option_id = graph.get("context_option_id")
        graph["context_option_literal_anchor"] = (
            context_option_id is not None
            and literal_option_anchor(choices[context_option_id], context)
        )
        quote_check = quote_diagnostics(
            context, graph["path_type"], graph["quotes"]
        )
        provisional_base = fallback_option_id(
            graph.get("base_option_id"),
            graph.get("context_option_id"),
            graph.get("prior_option_id"),
            choices,
        )
        verifier = {
            "recovered_option_id": None,
            "entailment": "NONE",
            "quotes": [],
            "called": False,
            "raw": "",
        }
        local_verifier = {
            "recovered_option_id": None,
            "entailment": "NONE",
            "quotes": [],
            "called": False,
            "raw": "",
        }
        has_idk_option = any(normalize(choice) in IDK_NORMALS for choice in choices)
        if (
            graph.get("context_option_id") is not None
            and (
                provisional_base != graph.get("context_option_id")
                or has_idk_option
            )
        ):
            quotes = "\n".join(
                f"{quote_index}: {quote}"
                for quote_index, quote in enumerate(graph["quotes"], start=1)
            ) or "NONE"
            verifier_prompt = BLIND_VERIFY_PROMPT.format(
                context=context,
                question=question,
                options=options,
                quotes=quotes,
            )
            raw_verifier = request_completion(
                args.base_url,
                args.model,
                verifier_prompt,
                prompt_format,
                args.seed + 2 * index + 1,
                300,
            )
            verifier = {
                **parse_blind_verifier(raw_verifier, len(choices)),
                "called": True,
                "raw": raw_verifier,
                "prompt_sha256": text_sha256(verifier_prompt),
            }
            local_prompt = LOCAL_VERIFY_PROMPT.format(
                question=question,
                options=options,
                quotes=quotes,
            )
            raw_local_verifier = request_completion(
                args.base_url,
                args.model,
                local_prompt,
                prompt_format,
                args.seed + 1_000_000 + index,
                180,
            )
            local_verifier = {
                **parse_blind_verifier(raw_local_verifier, len(choices)),
                "called": True,
                "raw": raw_local_verifier,
                "prompt_sha256": text_sha256(local_prompt),
            }
        verifier_quote_check = quote_diagnostics(
            context,
            verifier.get("entailment", "NONE"),
            verifier.get("quotes", []),
        )
        finalized = finalize_graph(
            graph,
            verifier,
            local_verifier,
            quote_check,
            verifier_quote_check,
            choices,
        )
        predictions = {
            variant: score_option(option_index, choices, sample["answer"])
            for variant, option_index in finalized["variant_option_ids"].items()
        }
        return {
            "idx": index,
            "gold": sample["answer"],
            "variants": predictions,
            "ugcp": {
                **finalized,
                "graph": graph,
                "quote_check": quote_check,
                "verifier_quote_check": verifier_quote_check,
                "verifier": verifier,
                "local_verifier": local_verifier,
                "edges": [
                    {
                        "source": "prior",
                        "target": "context",
                        "relation": "directed_conflict",
                        "active": finalized["predicates"]["directional_conflict"],
                    },
                    {
                        "source": "base_prior_agreement",
                        "target": "directional_gate",
                        "relation": "prior_trap_route",
                        "active": finalized["predicates"]["directional_gate"],
                    },
                    {
                        "source": "evidence_path",
                        "target": "context",
                        "relation": "verbatim_support",
                        "active": finalized["predicates"]["exact_quote"],
                    },
                    {
                        "source": "blind_recovery",
                        "target": "context",
                        "relation": "global_candidate_blind_entailment",
                        "active": finalized["predicates"]["global_certificate"],
                    },
                    {
                        "source": "local_blind_recovery",
                        "target": "context",
                        "relation": "local_candidate_blind_entailment",
                        "active": finalized["predicates"]["local_certificate"],
                    },
                    {
                        "source": "certificate",
                        "target": "final",
                        "relation": "conservative_projection",
                        "active": finalized["full_certified"],
                    },
                ],
                "raw_graph": raw_graph,
                "graph_prompt_sha256": text_sha256(graph_prompt),
            },
            "inference": {
                "model": args.model,
                "prompt_format": prompt_format,
                "graph_seed": args.seed + 2 * index,
                "verifier_seed": args.seed + 2 * index + 1,
                "local_verifier_seed": args.seed + 1_000_000 + index,
                "temperature": 0.0,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def evaluate_plain(index: int) -> dict[str, Any]:
        sample = data[index]
        question = str(sample["question"])
        context = str(sample["context"])
        choices = [str(choice) for choice in sample["choices"]]
        prompt = PLAIN_PROMPT.format(
            context=context,
            question=question,
            options=format_options(choices),
        )
        raw = request_completion(
            args.base_url,
            args.model,
            prompt,
            prompt_format,
            args.seed + index,
            64,
        )
        parsed_option_id = parse_plain(raw, len(choices))
        selected_option_id = fallback_option_id(
            parsed_option_id, None, None, choices
        )
        result = score_option(selected_option_id, choices, sample["answer"])
        return {
            "idx": index,
            "gold": sample["answer"],
            "prediction": result["prediction"],
            "option_id": result["option_id"],
            "metrics": {
                key: result[key] for key in ("acc", "em", "f1", "option_acc")
            },
            "raw": raw,
            "prompt_sha256": text_sha256(prompt),
            "inference": {
                "model": args.model,
                "prompt_format": prompt_format,
                "seed": args.seed + index,
                "temperature": 0.0,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    evaluate = evaluate_graph if args.mode == "graph" else evaluate_plain
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    output_mode = "a" if args.output.exists() else "w"
    with args.output.open(output_mode, encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate, index): index for index in pending}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    LOGGER.exception("sample_failed idx=%d error=%s", index, error)
                    failures.append({"idx": index, "error": repr(error)})
                    continue
                completed[index] = record
                with lock:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                number_done = len(completed)
                if number_done % 25 == 0 or number_done == len(indices):
                    if args.mode == "graph":
                        running = (
                            100
                            * sum(
                                item["variants"]["full"]["acc"]
                                for item in completed.values()
                            )
                            / number_done
                        )
                    else:
                        running = (
                            100
                            * sum(item["metrics"]["acc"] for item in completed.values())
                            / number_done
                        )
                    LOGGER.info(
                        "progress mode=%s dataset=%s completed=%d/%d ACC=%.2f",
                        args.mode,
                        args.dataset,
                        number_done,
                        len(indices),
                        running,
                    )

    records = [completed[index] for index in sorted(completed) if index in indices]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD_NAME if args.mode == "graph" else "Matched plain-context baseline",
        "mode": args.mode,
        "model": args.model,
        "prompt_format": prompt_format,
        "dataset": args.dataset,
        "dataset_source": str(data_path.resolve()),
        "dataset_sha256": sha256(data_path),
        "evaluation_protocol": str(EVALUATE_PATH.resolve()),
        "evaluation_sha256": sha256(EVALUATE_PATH),
        "source_sha256": sha256(source_path),
        "base_prediction_source": str(args.base_input.resolve()) if args.base_input else None,
        "base_prediction_sha256": sha256(args.base_input) if args.base_input else None,
        "graph_prompt_sha256": text_sha256(GRAPH_PROMPT),
        "blind_prompt_sha256": text_sha256(BLIND_VERIFY_PROMPT),
        "local_prompt_sha256": text_sha256(LOCAL_VERIFY_PROMPT),
        "plain_prompt_sha256": text_sha256(PLAIN_PROMPT),
        "zero_shot": True,
        "workers": args.workers,
        "seed": args.seed,
        "temperature": 0.0,
        "selected_samples": len(indices),
        "completed_samples": len(records),
        "failed_samples": len(failures),
        "failures": failures,
        "full_official_evaluation": (
            indices == list(range(len(data))) and len(records) == len(data)
        ),
        "wall_clock_s": round(time.time() - start_time, 1),
        "server_metadata": server_metadata,
        "results": (
            aggregate_graph(records) if args.mode == "graph" else aggregate_plain(records)
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures and not args.allow_partial:
        raise RuntimeError(
            f"{len(failures)} samples failed; rerun without --no_resume to complete them"
        )


if __name__ == "__main__":
    run()
