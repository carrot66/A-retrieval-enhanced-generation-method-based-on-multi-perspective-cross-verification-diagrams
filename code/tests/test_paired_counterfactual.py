from __future__ import annotations

import unittest

from ugcp_v5.paired_counterfactual import summarize_variant, validate_pair


class PairedCounterfactualTest(unittest.TestCase):
    def setUp(self) -> None:
        self.golden = [
            {
                "question": "Where is the site?",
                "context": "The site is in France.",
                "choices": ["France", "Spain"],
                "answer": "France",
            },
            {
                "question": "Where is the second site?",
                "context": "The site is in France.",
                "choices": ["France", "Spain"],
                "answer": "France",
            },
        ]
        self.counterfactual = [
            {
                **sample,
                "context": "The site is in Spain.",
                "answer": "Spain",
            }
            for sample in self.golden
        ]

    def test_pair_integrity_requires_answer_changing_contexts(self) -> None:
        integrity = validate_pair(self.golden, self.counterfactual)
        self.assertTrue(integrity["valid"])
        self.assertEqual(integrity["disjoint_answers"], 2)

    def test_joint_accuracy_requires_both_predictions(self) -> None:
        metrics, joint = summarize_variant(
            [0, 0],
            [1, 0],
            self.golden,
            self.counterfactual,
        )
        self.assertEqual(metrics["golden_option_acc"], 100.0)
        self.assertEqual(metrics["counterfactual_option_acc"], 50.0)
        self.assertEqual(metrics["joint_option_acc"], 50.0)
        self.assertEqual(metrics["stale_given_golden_correct"], 50.0)
        self.assertEqual(joint.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
