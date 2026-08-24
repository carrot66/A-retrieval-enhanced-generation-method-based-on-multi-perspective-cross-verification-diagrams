from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from ugcp_v5 import projection
from ugcp_v5.provenance_robustness import build_family_graph


ROOT = Path(__file__).resolve().parents[1]


def source_record(
    *,
    graph_base: int | None,
    graph_context: int | None,
    global_option: int | None = None,
    local_option: int | None = None,
    predicates: dict | None = None,
) -> dict:
    value = lambda item: "NONE" if item is None else str(item)
    return {
        "ugcp": {
            "context_option_id": graph_context,
            "predicates": {
                "graph_exact_quote": False,
                "local_blind_recovery": False,
                "global_certificate": False,
                "local_certificate": False,
                "abstention_consensus": False,
                **(predicates or {}),
            },
            "raw_graph": (
                f"BASE_OPTION_ID={value(graph_base)}\n"
                f"CONTEXT_OPTION_ID={value(graph_context)}"
            ),
            "verifier": {
                "raw": f"RECOVERED_OPTION_ID={value(global_option)}"
            },
            "local_verifier": {
                "raw": f"RECOVERED_OPTION_ID={value(local_option)}"
            },
        }
    }


def bridge(local_option: int, query_option: int) -> dict:
    return {
        "local_alignment": {"selected_option_id": local_option},
        "query_entailment": {"selected_option_id": query_option},
    }


class FlowGraphTest(unittest.TestCase):
    def test_each_source_channel_can_be_removed_independently(self) -> None:
        nodes = {
            "plain": 1,
            "graph_base": 1,
            "graph_context": 1,
            "global_verifier": 1,
            "local_verifier": 1,
            "local_semantic": 1,
            "query_semantic": 1,
        }
        for channel in projection.SOURCE_CHANNELS:
            graph, topology = projection.build_evidence_graph(
                nodes,
                conflict=False,
                **{f"include_{channel}": False},
            )
            self.assertEqual(
                topology["families"][channel],
                {role: None for role in topology["families"][channel]},
            )
            self.assertEqual(
                projection.flow_certificate(graph, 2)["winner"],
                1,
            )

    def test_flow_certificate_respects_minimum_support(self) -> None:
        nodes = {
            "plain": 1,
            "graph_base": 1,
            "graph_context": 1,
            "global_verifier": 1,
            "local_verifier": None,
            "local_semantic": None,
            "query_semantic": None,
        }
        graph, _ = projection.build_evidence_graph(nodes, conflict=False)
        self.assertEqual(
            projection.flow_certificate(
                graph, 2, minimum_support=2
            )["winner"],
            1,
        )
        self.assertIsNone(
            projection.flow_certificate(
                graph, 2, minimum_support=4
            )["winner"]
        )

    def test_graph_roles_share_one_source_capacity(self) -> None:
        nodes = {
            "plain": 0,
            "graph_base": 0,
            "graph_context": 1,
            "global_verifier": None,
            "local_verifier": None,
            "local_semantic": None,
            "query_semantic": None,
        }
        graph, _ = projection.build_evidence_graph(
            nodes,
            conflict=False,
            include_local_semantic=False,
            include_query_semantic=False,
        )
        certificate = projection.flow_certificate(graph, 2)
        self.assertEqual(certificate["scores"], {"0": 2, "1": 1})

    def test_query_confirmation_cuts_prior_side(self) -> None:
        nodes = {
            "plain": 0,
            "graph_base": 0,
            "graph_context": 1,
            "global_verifier": 1,
            "local_verifier": 1,
            "local_semantic": 1,
            "query_semantic": 1,
        }
        graph, topology = projection.build_evidence_graph(nodes, conflict=True)
        certificate = projection.flow_certificate(graph, 2)
        self.assertTrue(topology["cut_graph_base"])
        self.assertTrue(topology["cut_plain"])
        self.assertEqual(certificate["winner"], 1)
        self.assertEqual(certificate["scores"], {"0": 0, "1": 5})

    def test_unconfirmed_disagreement_preserves_prior_side(self) -> None:
        nodes = {
            "plain": 0,
            "graph_base": 0,
            "graph_context": 1,
            "global_verifier": 1,
            "local_verifier": None,
            "local_semantic": 1,
            "query_semantic": 0,
        }
        graph, topology = projection.build_evidence_graph(nodes, conflict=True)
        certificate = projection.flow_certificate(graph, 2)
        self.assertFalse(topology["query_confirms_context"])
        self.assertFalse(topology["cut_graph_base"])
        self.assertFalse(topology["cut_plain"])
        self.assertEqual(certificate["scores"], {"0": 3, "1": 3})
        self.assertIsNone(certificate["winner"])

    def test_tied_flow_has_no_certificate(self) -> None:
        nodes = {
            "plain": 0,
            "graph_base": 0,
            "graph_context": 1,
            "global_verifier": 1,
            "local_verifier": None,
            "local_semantic": 0,
            "query_semantic": 1,
        }
        graph, _ = projection.build_evidence_graph(
            nodes,
            conflict=False,
        )
        certificate = projection.flow_certificate(graph, 2)
        self.assertIsNone(certificate["winner"])
        self.assertTrue(certificate["tie"])

    def test_repeated_observations_within_source_do_not_add_flow(self) -> None:
        families = {
            "graph_call": {"base": 0, "context": 1},
            "verifier": {"answer": 1},
        }
        baseline = projection.flow_certificate(
            build_family_graph(families), 2
        )
        replicated = projection.flow_certificate(
            build_family_graph(families, replicas_per_observation=20), 2
        )
        self.assertGreater(replicated["nodes"], baseline["nodes"])
        self.assertEqual(replicated["scores"], baseline["scores"])
        self.assertEqual(replicated["winner"], baseline["winner"])

    def test_shared_backbone_caps_all_llm_branches_together(self) -> None:
        nodes = {
            "plain": 1,
            "graph_base": 1,
            "graph_context": 1,
            "global_verifier": 1,
            "local_verifier": 1,
            "local_semantic": 1,
            "query_semantic": 1,
        }
        call_level, _ = projection.build_evidence_graph(
            nodes, conflict=False
        )
        backbone_level, topology = projection.build_evidence_graph(
            nodes,
            conflict=False,
            shared_backbone_capacity=True,
        )
        self.assertEqual(
            projection.flow_certificate(call_level, 2)["scores"]["1"], 6
        )
        self.assertEqual(
            projection.flow_certificate(backbone_level, 2)["scores"]["1"],
            3,
        )
        self.assertTrue(topology["shared_backbone_capacity"])

    def test_one_added_family_changes_each_score_by_at_most_one(self) -> None:
        baseline_families = {
            "plain": {"answer": 0},
            "verifier": {"answer": 1},
        }
        expanded_families = {
            **baseline_families,
            "new_source": {"answer": 0, "duplicate_field": 0},
        }
        baseline = projection.flow_certificate(
            build_family_graph(baseline_families), 2
        )
        expanded = projection.flow_certificate(
            build_family_graph(expanded_families), 2
        )
        for option_id in ("0", "1"):
            self.assertLessEqual(
                expanded["scores"][option_id]
                - baseline["scores"][option_id],
                1,
            )


class ProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "context": "The hypothetical evidence supports the second choice.",
            "question": "Which choice is supported?",
            "choices": ["first", "second"],
        }

    def test_conflict_projects_query_confirmed_flow(self) -> None:
        variants, diagnostics = projection.select_variants(
            {"raw": "OPTION_ID=0", "option_id": 0},
            source_record(
                graph_base=0,
                graph_context=1,
                global_option=1,
                local_option=1,
            ),
            bridge(1, 1),
            self.sample,
        )
        self.assertEqual(variants["full"], 1)
        self.assertTrue(diagnostics["full"]["topology"]["cut_plain"])

    def test_removed_graph_call_cannot_drive_scope_route(self) -> None:
        selected, diagnostics = projection.scope_risk_projection(
            source_record(
                graph_base=0,
                graph_context=1,
                global_option=1,
                local_option=1,
                predicates={"global_certificate": True},
            ),
            self.sample,
            0,
            include_graph_call=False,
        )
        self.assertEqual(selected, 0)
        self.assertIsNone(diagnostics["context_option_id"])
        self.assertFalse(diagnostics["candidate_disagreement"])

    def test_removed_verifiers_cannot_certify_scope_route(self) -> None:
        _, diagnostics = projection.scope_risk_projection(
            source_record(
                graph_base=0,
                graph_context=1,
                global_option=1,
                local_option=1,
                predicates={
                    "global_certificate": True,
                    "local_certificate": True,
                    "local_blind_recovery": True,
                    "abstention_consensus": True,
                },
            ),
            self.sample,
            0,
            include_global_verifier=False,
            include_local_verifier=False,
        )
        self.assertFalse(diagnostics["global_certificate"])
        self.assertFalse(diagnostics["local_certificate"])
        self.assertFalse(diagnostics["local_blind_recovery"])
        self.assertFalse(diagnostics["abstention_consensus"])

    def test_plain_idk_remains_abstention(self) -> None:
        sample = {
            **self.sample,
            "choices": ["supported answer", "I don't know"],
        }
        variants, diagnostics = projection.select_variants(
            {"raw": "OPTION_ID=1", "option_id": 1},
            source_record(
                graph_base=1,
                graph_context=0,
                global_option=0,
                local_option=0,
            ),
            bridge(0, 0),
            sample,
        )
        self.assertEqual(
            variants["full"], diagnostics["scope_certificate"]["idk_option_id"]
        )

    def test_idk_protection_is_inactive_without_idk_option(self) -> None:
        variants, _ = projection.select_variants(
            {"raw": "OPTION_ID=0", "option_id": 0},
            source_record(
                graph_base=0,
                graph_context=1,
                global_option=1,
                local_option=1,
            ),
            bridge(1, 1),
            self.sample,
        )
        self.assertEqual(variants["full"], variants["without_idk_protection"])

    def test_idk_sufficiency_accepts_three_source_families(self) -> None:
        sample = {
            **self.sample,
            "choices": ["supported answer", "I don't know"],
        }
        variants, diagnostics = projection.select_variants(
            {"raw": "OPTION_ID=0", "option_id": 0},
            source_record(
                graph_base=0,
                graph_context=0,
                global_option=0,
                local_option=None,
            ),
            bridge(1, 1),
            sample,
        )
        self.assertEqual(variants["full"], 0)
        self.assertEqual(
            diagnostics["full"]["idk_sufficiency"]["source_family_support"],
            3,
        )

    def test_idk_sufficiency_abstains_below_three_families(self) -> None:
        sample = {
            **self.sample,
            "choices": ["unsupported answer", "I don't know"],
        }
        variants, diagnostics = projection.select_variants(
            {"raw": "OPTION_ID=0", "option_id": 0},
            source_record(
                graph_base=0,
                graph_context=0,
                global_option=None,
                local_option=None,
            ),
            bridge(1, 1),
            sample,
        )
        self.assertEqual(variants["full"], 1)
        self.assertEqual(variants["without_idk_protection"], 0)
        self.assertFalse(
            diagnostics["full"]["idk_sufficiency"]["authorized_non_idk"]
        )

    def test_idk_sufficiency_threshold_is_parameterized(self) -> None:
        nodes = {
            "plain": 0,
            "graph_base": 0,
            "graph_context": 0,
            "global_verifier": None,
            "local_verifier": None,
            "local_semantic": 1,
            "query_semantic": 1,
        }
        accepted, accepted_diagnostics = projection.idk_sufficiency_projection(
            nodes,
            1,
            minimum_source_families=2,
        )
        rejected, rejected_diagnostics = projection.idk_sufficiency_projection(
            nodes,
            1,
            minimum_source_families=3,
        )
        self.assertEqual(accepted, 0)
        self.assertTrue(accepted_diagnostics["authorized_non_idk"])
        self.assertEqual(rejected, 1)
        self.assertFalse(rejected_diagnostics["authorized_non_idk"])

    def test_decision_is_invariant_to_gold(self) -> None:
        plain = {"raw": "OPTION_ID=0", "option_id": 0}
        graph = source_record(
            graph_base=0,
            graph_context=1,
            global_option=1,
            local_option=1,
        )
        semantic = bridge(1, 1)
        with_gold = {**self.sample, "answer": "first"}
        changed_gold = {**self.sample, "answer": "second"}
        self.assertEqual(
            projection.select_variants(plain, graph, semantic, with_gold),
            projection.select_variants(plain, graph, semantic, changed_gold),
        )

    def test_option_permutation_is_equivariant(self) -> None:
        plain = {"raw": "OPTION_ID=0", "option_id": 0}
        graph = source_record(
            graph_base=0,
            graph_context=1,
            global_option=1,
            local_option=1,
        )
        semantic = bridge(1, 1)
        original, _ = projection.select_variants(
            plain, graph, semantic, self.sample
        )

        permuted_sample = {
            **self.sample,
            "choices": list(reversed(self.sample["choices"])),
        }
        permuted_plain = {"raw": "OPTION_ID=1", "option_id": 1}
        permuted_graph = source_record(
            graph_base=1,
            graph_context=0,
            global_option=0,
            local_option=0,
        )
        permuted_semantic = bridge(0, 0)
        permuted, _ = projection.select_variants(
            permuted_plain,
            permuted_graph,
            permuted_semantic,
            permuted_sample,
        )
        self.assertEqual(permuted["full"], 1 - original["full"])

    def test_majority_baseline_uses_same_observations(self) -> None:
        variants, _ = projection.select_variants(
            {"raw": "OPTION_ID=0", "option_id": 0},
            source_record(
                graph_base=0,
                graph_context=1,
                global_option=None,
                local_option=None,
            ),
            bridge(0, 1),
            self.sample,
        )
        self.assertEqual(variants["majority_same_budget"], 0)
        self.assertEqual(variants["full"], 1)

    def test_provenance_ablation_is_single_factor(self) -> None:
        variants, _ = projection.select_variants(
            {"raw": "OPTION_ID=0", "option_id": 0},
            source_record(
                graph_base=0,
                graph_context=1,
                global_option=1,
                local_option=1,
            ),
            bridge(1, 1),
            self.sample,
        )
        self.assertIn("without_provenance_capacity", variants)
        self.assertIn("shared_backbone_capacity", variants)
        self.assertIn("without_directional_cut", variants)
        self.assertIn("without_typed_topology", variants)

    def test_grouped_vote_counts_one_graph_call_per_option(self) -> None:
        nodes = {
            "plain": 0,
            "graph_base": 0,
            "graph_context": 1,
            "global_verifier": None,
            "local_verifier": None,
            "local_semantic": 1,
            "query_semantic": 1,
        }
        self.assertEqual(projection.provenance_grouped_vote(nodes, 0), 1)

    def test_scope_only_is_reported_without_graph_update(self) -> None:
        variants, diagnostics = projection.select_variants(
            {"raw": "OPTION_ID=0", "option_id": 0},
            source_record(
                graph_base=0,
                graph_context=1,
                global_option=1,
                local_option=1,
                predicates={"global_certificate": True},
            ),
            bridge(1, 1),
            self.sample,
        )
        self.assertEqual(variants["scope_only"], diagnostics["scope_option_id"])


class ProtocolTest(unittest.TestCase):
    def test_official_evaluator_is_unchanged(self) -> None:
        digest = hashlib.sha256(
            (ROOT / "src/ugcp_v5/evaluate.py").read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest,
            "ad2eafe2502f4f2a38855ab38b1a69f59165726e6586f79181e191dc5d506ebb",
        )

    def test_method_has_no_model_or_dataset_branch(self) -> None:
        source = (ROOT / "src/ugcp_v5/projection.py").read_text(encoding="utf-8")
        for forbidden in ("faitheval", "musique", "squad", "qwen", "llama", "mistral"):
            self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__":
    unittest.main()
