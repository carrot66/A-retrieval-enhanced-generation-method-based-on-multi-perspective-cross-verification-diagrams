"""Label-free UGCP-v5 projection over a provenance-capacitated evidence graph."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx


EVALUATE_PATH = Path(__file__).resolve().with_name("evaluate.py")
EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "truthfulrag_official_evaluate_v5", EVALUATE_PATH
)
if EVALUATE_SPEC is None or EVALUATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load TruthfulRAG evaluator: {EVALUATE_PATH}")
TRUTHFULRAG_EVALUATE = importlib.util.module_from_spec(EVALUATE_SPEC)
EVALUATE_SPEC.loader.exec_module(TRUTHFULRAG_EVALUATE)

METHOD_NAME = "UGCP-v5.1-QPCF"
WORKERS = 12
SOURCE_NODE = "source"
FLOW_MIN_SOURCE_FAMILIES = 2
IDK_MIN_SOURCE_FAMILIES = 3
SOURCE_CHANNELS = (
    "plain",
    "graph_call",
    "global_verifier",
    "local_verifier",
    "local_semantic",
    "query_semantic",
)
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
    "without_query_semantic",
    "without_local_semantic",
    "without_semantic_bridge",
    "without_provenance_capacity",
    "shared_backbone_capacity",
    "without_typed_topology",
    "without_directional_cut",
    "without_risk_projection",
    "without_idk_protection",
    "scope_only",
    "plain_context",
    "majority_same_budget",
    "provenance_grouped_vote",
]
PLAIN_IDENTIFIER_PATTERNS = [
    re.compile(r"(?i)(?:final\s*)?option\s*[_ -]?\s*id\s*[:=]\s*(\d+)"),
    re.compile(
        r"(?i)\b(?:answer|selection|selction)\s*(?:option\s*)?"
        r"(?:id\s*)?[:=]?\s*(\d+)\b"
    ),
    re.compile(r"(?i)\boption\s*[_ -]?\s*(\d+)\b"),
]


def normalize(text: Any) -> str:
    return TRUTHFULRAG_EVALUATE.normalize_answer(str(text or ""))


def score(prediction: str, gold: str | list[str]) -> dict[str, float]:
    return {
        "em": float(
            TRUTHFULRAG_EVALUATE.metric_max_over_ground_truths(
                TRUTHFULRAG_EVALUATE.exact_match_score, prediction, gold
            )
        ),
        "acc": float(
            TRUTHFULRAG_EVALUATE.metric_max_over_ground_truths(
                TRUTHFULRAG_EVALUATE.acc_score, prediction, gold
            )
        ),
        "f1": float(
            TRUTHFULRAG_EVALUATE.metric_max_over_ground_truths(
                TRUTHFULRAG_EVALUATE.f1_score, prediction, gold
            )
        ),
    }


def gold_option_ids(gold: str | list[str], choices: list[str]) -> set[int]:
    answers = gold if isinstance(gold, list) else [gold]
    normalized_answers = {normalize(answer) for answer in answers}
    return {
        index
        for index, choice in enumerate(choices)
        if normalize(choice) in normalized_answers
    }


def score_option(
    option_id: int,
    choices: list[str],
    gold: str | list[str],
) -> dict[str, Any]:
    prediction = choices[option_id]
    return {
        "option_id": option_id,
        "prediction": prediction,
        **score(prediction, gold),
        "option_acc": float(option_id in gold_option_ids(gold, choices)),
    }


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        index = int(record["idx"])
        if index in records:
            raise ValueError(f"duplicate index {index}: {path}")
        records[index] = record
    return records


def clean_serialization(raw: Any) -> str:
    return str(raw or "").replace("\\_", "_").replace("\\\\", "")


def parse_canonical_identifier(
    raw: Any,
    field: str,
    number_of_choices: int,
) -> int | None:
    matches = re.findall(
        rf"^\s*{re.escape(field)}\s*=\s*(\d+|NONE)\s*$",
        clean_serialization(raw),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not matches or matches[-1].upper() == "NONE":
        return None
    value = int(matches[-1])
    return value if 0 <= value < number_of_choices else None


def parse_graph_identifier(
    raw: Any,
    field: str,
    number_of_choices: int,
) -> int | None:
    field_pattern = re.escape(field).replace("_", r"[_ ]*")
    matches = re.findall(
        rf"(?im)(?:^|\n)\s*(?:[IVXLCDM]+\.|\d+[.)]|[-*])?\s*"
        rf"{field_pattern}\s*[:=]\s*(\d+|NONE)\b",
        clean_serialization(raw),
    )
    if not matches or matches[-1].upper() == "NONE":
        return None
    value = int(matches[-1])
    return value if 0 <= value < number_of_choices else None


def parse_plain_candidate(raw: Any, choices: list[str]) -> dict[str, Any]:
    number_of_choices = len(choices)
    canonical = parse_canonical_identifier(raw, "OPTION_ID", number_of_choices)
    if canonical is not None:
        return {
            "option_id": canonical,
            "route": "canonical_option_id",
            "schema_recovered": False,
        }

    cleaned = clean_serialization(raw)
    for pattern in PLAIN_IDENTIFIER_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            value = int(matches[-1])
            if 0 <= value < number_of_choices:
                return {
                    "option_id": value,
                    "route": "recovered_identifier",
                    "schema_recovered": True,
                }

    prefixed = [
        int(value)
        for value in re.findall(r"(?m)^\s*[.\[(]*\s*(\d+)\s*[:.)-]", cleaned)
        if 0 <= int(value) < number_of_choices
    ]
    if prefixed:
        return {
            "option_id": prefixed[-1],
            "route": "recovered_numbered_prefix",
            "schema_recovered": True,
        }

    normalized_raw = normalize(cleaned)
    text_matches = {
        index
        for index, choice in enumerate(choices)
        if len(normalize(choice)) >= 3 and normalize(choice) in normalized_raw
    }
    if len(text_matches) == 1:
        return {
            "option_id": next(iter(text_matches)),
            "route": "recovered_unique_choice_text",
            "schema_recovered": True,
        }
    return {"option_id": None, "route": "unresolved", "schema_recovered": False}


def find_idk_option(choices: list[str]) -> int | None:
    return next(
        (
            index
            for index, choice in enumerate(choices)
            if normalize(choice) in IDK_NORMALS
        ),
        None,
    )


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


def scope_risk_projection(
    source_record: dict[str, Any],
    sample: dict[str, Any],
    base_option_id: int,
    *,
    include_graph_call: bool = True,
    include_global_verifier: bool = True,
    include_local_verifier: bool = True,
) -> tuple[int, dict[str, Any]]:
    choices = [str(choice) for choice in sample["choices"]]
    context = str(sample["context"])
    ugcp = source_record["ugcp"]
    predicates = ugcp["predicates"]
    context_option_id = ugcp.get("context_option_id") if include_graph_call else None
    if context_option_id is not None:
        context_option_id = int(context_option_id)
        if not 0 <= context_option_id < len(choices):
            context_option_id = None
    idk_option_id = find_idk_option(choices)
    candidate_disagreement = (
        context_option_id is not None and context_option_id != base_option_id
    )
    option_literal_anchor_count = sum(
        literal_option_anchor(choice, context) for choice in choices
    )
    context_literal_anchor = (
        context_option_id is not None
        and literal_option_anchor(choices[context_option_id], context)
    )
    graph_exact_quote = bool(predicates.get("graph_exact_quote")) if include_graph_call else False
    local_blind_recovery = (
        bool(predicates.get("local_blind_recovery"))
        if include_local_verifier
        else False
    )
    global_certificate = (
        bool(predicates.get("global_certificate"))
        if include_global_verifier
        else False
    )
    local_certificate = (
        bool(predicates.get("local_certificate"))
        if include_local_verifier
        else False
    )
    abstention_consensus = (
        bool(predicates.get("abstention_consensus"))
        if include_local_verifier
        else False
    )
    anchored_direct = (
        candidate_disagreement
        and context_literal_anchor
        and (graph_exact_quote or local_blind_recovery)
    )
    global_semantic = (
        candidate_disagreement
        and option_literal_anchor_count == 0
        and global_certificate
    )

    if idk_option_id is not None:
        local_conflict_decision = (
            context_option_id
            if context_option_id is not None
            and context_option_id != idk_option_id
            and (global_certificate or abstention_consensus)
            else idk_option_id
        )
        selected_option_id = (
            local_conflict_decision if local_certificate else base_option_id
        )
    else:
        selected_option_id = (
            context_option_id
            if (anchored_direct or global_semantic)
            and context_option_id is not None
            else base_option_id
        )

    return selected_option_id, {
        "context_option_id": context_option_id,
        "idk_option_id": idk_option_id,
        "candidate_disagreement": candidate_disagreement,
        "option_literal_anchor_count": option_literal_anchor_count,
        "context_literal_anchor": context_literal_anchor,
        "graph_exact_quote": graph_exact_quote,
        "local_blind_recovery": local_blind_recovery,
        "global_certificate": global_certificate,
        "local_certificate": local_certificate,
        "abstention_consensus": abstention_consensus,
        "anchored_direct": anchored_direct,
        "global_semantic": global_semantic,
    }


def semantic_option_id(
    bridge_record: dict[str, Any],
    edge: str,
    number_of_choices: int,
) -> int:
    value = int(bridge_record[edge]["selected_option_id"])
    if not 0 <= value < number_of_choices:
        raise ValueError(f"invalid {edge} semantic option id: {value}")
    return value


def observation_nodes(
    plain_record: dict[str, Any],
    source_record: dict[str, Any],
    bridge_record: dict[str, Any],
    choices: list[str],
) -> tuple[dict[str, int | None], dict[str, Any]]:
    plain_parse = parse_plain_candidate(plain_record.get("raw", ""), choices)
    parsed_base = plain_parse["option_id"]
    base_option_id = (
        int(parsed_base)
        if parsed_base is not None
        else int(plain_record["option_id"])
    )
    if not 0 <= base_option_id < len(choices):
        raise ValueError(f"invalid base option id: {base_option_id}")
    ugcp = source_record["ugcp"]
    nodes = {
        "plain": base_option_id,
        "graph_base": parse_graph_identifier(
            ugcp.get("raw_graph", ""), "BASE_OPTION_ID", len(choices)
        ),
        "graph_context": parse_graph_identifier(
            ugcp.get("raw_graph", ""), "CONTEXT_OPTION_ID", len(choices)
        ),
        "global_verifier": parse_graph_identifier(
            ugcp.get("verifier", {}).get("raw", ""),
            "RECOVERED_OPTION_ID",
            len(choices),
        ),
        "local_verifier": parse_graph_identifier(
            ugcp.get("local_verifier", {}).get("raw", ""),
            "RECOVERED_OPTION_ID",
            len(choices),
        ),
        "local_semantic": semantic_option_id(
            bridge_record, "local_alignment", len(choices)
        ),
        "query_semantic": semantic_option_id(
            bridge_record, "query_entailment", len(choices)
        ),
    }
    return nodes, plain_parse


def add_source_family(
    graph: nx.DiGraph,
    family: str,
    observations: dict[str, int | None],
    *,
    source_node: str = SOURCE_NODE,
) -> None:
    active = {
        role: option_id
        for role, option_id in observations.items()
        if option_id is not None
    }
    if not active:
        return
    family_node = f"family:{family}"
    graph.add_edge(source_node, family_node, capacity=1)
    for role, option_id in active.items():
        observation_node = f"observation:{family}:{role}"
        option_node = f"option:{option_id}"
        graph.add_edge(family_node, observation_node, capacity=1)
        graph.add_edge(observation_node, option_node, capacity=1)


def build_evidence_graph(
    nodes: dict[str, int | None],
    *,
    conflict: bool,
    include_plain: bool = True,
    include_graph_call: bool = True,
    include_global_verifier: bool = True,
    include_local_verifier: bool = True,
    include_local_semantic: bool = True,
    include_query_semantic: bool = True,
    provenance_capacity: bool = True,
    shared_backbone_capacity: bool = False,
    directional_cut: bool = True,
) -> tuple[nx.DiGraph, dict[str, Any]]:
    graph = nx.DiGraph()
    graph.add_node(SOURCE_NODE)
    query_confirms_context = (
        include_query_semantic
        and include_graph_call
        and conflict
        and nodes["query_semantic"] == nodes["graph_context"]
    )
    cut_graph_base = directional_cut and query_confirms_context
    cut_plain = directional_cut and query_confirms_context

    graph_roles = {
        "base": (
            None
            if not include_graph_call or cut_graph_base
            else nodes["graph_base"]
        ),
        "context": nodes["graph_context"] if include_graph_call else None,
    }
    families = {
        "plain": {
            "plain": (
                None if not include_plain or cut_plain else nodes["plain"]
            )
        },
        "graph_call": graph_roles,
        "global_verifier": {
            "global": nodes["global_verifier"]
            if include_global_verifier
            else None
        },
        "local_verifier": {
            "local": nodes["local_verifier"] if include_local_verifier else None
        },
        "local_semantic": {
            "local": nodes["local_semantic"] if include_local_semantic else None
        },
        "query_semantic": {
            "query": nodes["query_semantic"] if include_query_semantic else None
        },
    }
    if provenance_capacity:
        llm_families = {
            "plain",
            "graph_call",
            "global_verifier",
            "local_verifier",
        }
        if shared_backbone_capacity:
            backbone_node = "backbone:llm"
            active_llm_family = any(
                option_id is not None
                for family, observations in families.items()
                if family in llm_families
                for option_id in observations.values()
            )
            if active_llm_family:
                graph.add_edge(SOURCE_NODE, backbone_node, capacity=1)
            for family, observations in families.items():
                add_source_family(
                    graph,
                    family,
                    observations,
                    source_node=(
                        backbone_node
                        if family in llm_families
                        else SOURCE_NODE
                    ),
                )
        else:
            for family, observations in families.items():
                add_source_family(graph, family, observations)
    else:
        for family, observations in families.items():
            for role, option_id in observations.items():
                add_source_family(
                    graph,
                    f"{family}:{role}",
                    {role: option_id},
                )
    return graph, {
        "query_confirms_context": query_confirms_context,
        "cut_graph_base": cut_graph_base,
        "cut_plain": cut_plain,
        "shared_backbone_capacity": shared_backbone_capacity,
        "families": families,
    }


def flow_certificate(
    graph: nx.DiGraph,
    number_of_choices: int,
    *,
    minimum_support: int = FLOW_MIN_SOURCE_FAMILIES,
) -> dict[str, Any]:
    if minimum_support < 1:
        raise ValueError("minimum_support must be positive")
    scores = {}
    for option_id in range(number_of_choices):
        option_node = f"option:{option_id}"
        scores[option_id] = (
            int(nx.maximum_flow_value(graph, SOURCE_NODE, option_node))
            if option_node in graph
            else 0
        )
    maximum = max(scores.values(), default=0)
    winners = [
        option_id
        for option_id, value in scores.items()
        if value == maximum and value >= minimum_support
    ]
    return {
        "winner": winners[0] if len(winners) == 1 else None,
        "flow": maximum if len(winners) == 1 else 0,
        "minimum_support": minimum_support,
        "scores": {str(key): value for key, value in scores.items()},
        "tie": len(winners) > 1,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }


def same_budget_majority(
    nodes: dict[str, int | None],
    base_option_id: int,
) -> int:
    counts: dict[int, int] = {}
    for option_id in nodes.values():
        if option_id is not None:
            counts[option_id] = counts.get(option_id, 0) + 1
    maximum = max(counts.values(), default=0)
    winners = [
        option_id
        for option_id, count in counts.items()
        if count == maximum and count >= 2
    ]
    return int(winners[0]) if len(winners) == 1 else base_option_id


def provenance_grouped_vote(
    nodes: dict[str, int | None],
    base_option_id: int,
) -> int:
    families = (
        (nodes["plain"],),
        (nodes["graph_base"], nodes["graph_context"]),
        (nodes["global_verifier"],),
        (nodes["local_verifier"],),
        (nodes["local_semantic"],),
        (nodes["query_semantic"],),
    )
    counts: dict[int, int] = {}
    for family in families:
        for option_id in {item for item in family if item is not None}:
            counts[option_id] = counts.get(option_id, 0) + 1
    maximum = max(counts.values(), default=0)
    winners = [
        option_id
        for option_id, count in counts.items()
        if count == maximum and count >= 2
    ]
    return int(winners[0]) if len(winners) == 1 else base_option_id


def source_family_support_count(
    nodes: dict[str, int | None],
    option_id: int,
    *,
    include_plain: bool = True,
    include_graph_call: bool = True,
    include_global_verifier: bool = True,
    include_local_verifier: bool = True,
    include_local_semantic: bool = True,
    include_query_semantic: bool = True,
) -> int:
    families = (
        ((nodes["plain"] if include_plain else None),),
        (
            nodes["graph_base"] if include_graph_call else None,
            nodes["graph_context"] if include_graph_call else None,
        ),
        ((nodes["global_verifier"] if include_global_verifier else None),),
        ((nodes["local_verifier"] if include_local_verifier else None),),
        ((nodes["local_semantic"] if include_local_semantic else None),),
        ((nodes["query_semantic"] if include_query_semantic else None),),
    )
    return sum(option_id in family for family in families)


def idk_sufficiency_projection(
    nodes: dict[str, int | None],
    idk_option_id: int,
    *,
    include_plain: bool = True,
    include_graph_call: bool = True,
    include_global_verifier: bool = True,
    include_local_verifier: bool = True,
    include_local_semantic: bool = True,
    include_query_semantic: bool = True,
    candidate_option_id: int | None = None,
    minimum_source_families: int = IDK_MIN_SOURCE_FAMILIES,
) -> tuple[int, dict[str, Any]]:
    if minimum_source_families < 1:
        raise ValueError("minimum_source_families must be positive")
    candidate = int(nodes["plain"] if candidate_option_id is None else candidate_option_id)
    support_count = source_family_support_count(
        nodes,
        candidate,
        include_plain=include_plain,
        include_graph_call=include_graph_call,
        include_global_verifier=include_global_verifier,
        include_local_verifier=include_local_verifier,
        include_local_semantic=include_local_semantic,
        include_query_semantic=include_query_semantic,
    )
    authorized = (
        candidate != idk_option_id
        and support_count >= minimum_source_families
    )
    return (candidate if authorized else idk_option_id), {
        "active": True,
        "candidate_option_id": candidate,
        "idk_option_id": idk_option_id,
        "source_family_support": support_count,
        "minimum_source_families": minimum_source_families,
        "authorized_non_idk": authorized,
    }


def project_variant(
    *,
    nodes: dict[str, int | None],
    plain_parse: dict[str, Any],
    scope_option_id: int,
    scope: dict[str, Any],
    number_of_choices: int,
    include_plain: bool = True,
    include_graph_call: bool = True,
    include_global_verifier: bool = True,
    include_local_verifier: bool = True,
    include_local_semantic: bool = True,
    include_query_semantic: bool = True,
    provenance_capacity: bool = True,
    shared_backbone_capacity: bool = False,
    directional_cut: bool = True,
    risk_projection: bool = True,
    idk_protection: bool = True,
    minimum_support: int = FLOW_MIN_SOURCE_FAMILIES,
    idk_minimum_source_families: int = IDK_MIN_SOURCE_FAMILIES,
    fallback_option_id: int | None = None,
) -> tuple[int, dict[str, Any]]:
    base_option_id = int(
        nodes["plain"] if fallback_option_id is None else fallback_option_id
    )
    conflict = bool(scope["candidate_disagreement"])
    graph_context_missing = (
        not include_graph_call or nodes["graph_context"] is None
    )
    active = (
        conflict
        or graph_context_missing
        or (include_plain and bool(plain_parse["schema_recovered"]))
    )
    graph, topology = build_evidence_graph(
        nodes,
        conflict=conflict,
        include_plain=include_plain,
        include_graph_call=include_graph_call,
        include_global_verifier=include_global_verifier,
        include_local_verifier=include_local_verifier,
        include_local_semantic=include_local_semantic,
        include_query_semantic=include_query_semantic,
        provenance_capacity=provenance_capacity,
        shared_backbone_capacity=shared_backbone_capacity,
        directional_cut=directional_cut,
    )
    certificate = flow_certificate(
        graph,
        number_of_choices,
        minimum_support=minimum_support,
    )
    winner = certificate["winner"]

    if risk_projection:
        authorized = (
            active
            and scope["idk_option_id"] is None
            and scope_option_id == base_option_id
            and winner is not None
        )
        option_id = int(winner) if authorized else scope_option_id
    else:
        authorized = active and winner is not None
        option_id = int(winner) if authorized else base_option_id
    idk_diagnostics = {"active": False}
    if (
        risk_projection
        and idk_protection
        and scope["idk_option_id"] is not None
    ):
        option_id, idk_diagnostics = idk_sufficiency_projection(
            nodes,
            int(scope["idk_option_id"]),
            include_plain=include_plain,
            include_graph_call=include_graph_call,
            include_global_verifier=include_global_verifier,
            include_local_verifier=include_local_verifier,
            include_local_semantic=include_local_semantic,
            include_query_semantic=include_query_semantic,
            candidate_option_id=base_option_id,
            minimum_source_families=idk_minimum_source_families,
        )
    return option_id, {
        "active": active,
        "conflict": conflict,
        "graph_context_missing": graph_context_missing,
        "authorized": authorized,
        "topology": topology,
        "certificate": certificate,
        "idk_sufficiency": idk_diagnostics,
    }


def select_variants(
    plain_record: dict[str, Any],
    source_record: dict[str, Any],
    bridge_record: dict[str, Any],
    sample: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    choices = [str(choice) for choice in sample["choices"]]
    nodes, plain_parse = observation_nodes(
        plain_record, source_record, bridge_record, choices
    )
    base_option_id = int(nodes["plain"])
    scope_option_id, scope = scope_risk_projection(
        source_record, sample, base_option_id
    )

    full_option_id, full_diagnostics = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
    )
    without_query, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        include_query_semantic=False,
    )
    without_local, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        include_local_semantic=False,
    )
    without_semantics, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        include_local_semantic=False,
        include_query_semantic=False,
    )
    without_provenance, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        provenance_capacity=False,
    )
    shared_backbone, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        shared_backbone_capacity=True,
    )
    without_topology, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        provenance_capacity=False,
        directional_cut=False,
    )
    without_cut, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        directional_cut=False,
    )
    without_risk, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        risk_projection=False,
    )
    without_idk, _ = project_variant(
        nodes=nodes,
        plain_parse=plain_parse,
        scope_option_id=scope_option_id,
        scope=scope,
        number_of_choices=len(choices),
        idk_protection=False,
    )
    variants = {
        "full": full_option_id,
        "without_query_semantic": without_query,
        "without_local_semantic": without_local,
        "without_semantic_bridge": without_semantics,
        "without_provenance_capacity": without_provenance,
        "shared_backbone_capacity": shared_backbone,
        "without_typed_topology": without_topology,
        "without_directional_cut": without_cut,
        "without_risk_projection": without_risk,
        "without_idk_protection": without_idk,
        "scope_only": scope_option_id,
        "plain_context": base_option_id,
        "majority_same_budget": same_budget_majority(nodes, base_option_id),
        "provenance_grouped_vote": provenance_grouped_vote(nodes, base_option_id),
    }
    if list(variants) != VARIANT_ORDER:
        raise RuntimeError("variant order drift")
    if any(not 0 <= option_id < len(choices) for option_id in variants.values()):
        raise ValueError("projection produced an invalid option id")
    return variants, {
        "base_option_id": base_option_id,
        "plain_parse": plain_parse,
        "scope_option_id": scope_option_id,
        "scope_certificate": scope,
        "nodes": nodes,
        "full": full_diagnostics,
    }
