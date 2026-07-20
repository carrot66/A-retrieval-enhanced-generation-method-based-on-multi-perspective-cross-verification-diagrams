from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "code/data_preparation/prepare_openbookqa.py"
SPEC = importlib.util.spec_from_file_location("prepare_openbookqa", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)

ROBUSTNESS_SCRIPT = ROOT / "code/data_preparation/prepare_robustness_variants.py"
ROBUSTNESS_SPEC = importlib.util.spec_from_file_location(
    "prepare_robustness_variants", ROBUSTNESS_SCRIPT
)
if ROBUSTNESS_SPEC is None or ROBUSTNESS_SPEC.loader is None:
    raise RuntimeError(f"cannot load {ROBUSTNESS_SCRIPT}")
ROBUSTNESS = importlib.util.module_from_spec(ROBUSTNESS_SPEC)
ROBUSTNESS_SPEC.loader.exec_module(ROBUSTNESS)

ISOLATION_SCRIPT = ROOT / "code/data_preparation/isolate_holdout_labels.py"
ISOLATION_SPEC = importlib.util.spec_from_file_location(
    "isolate_holdout_labels", ISOLATION_SCRIPT
)
if ISOLATION_SPEC is None or ISOLATION_SPEC.loader is None:
    raise RuntimeError(f"cannot load {ISOLATION_SCRIPT}")
ISOLATION = importlib.util.module_from_spec(ISOLATION_SPEC)
ISOLATION_SPEC.loader.exec_module(ISOLATION)

sys.path.insert(0, str(ROOT / "code/src"))
from ugcp_v5 import analyze_holdout


class OpenBookQAPreparationTests(unittest.TestCase):
    def test_exact_field_mapping(self) -> None:
        raw = {
            "id": "sample-1",
            "question": {
                "stem": "Which option?",
                "choices": [
                    {"text": "first", "label": "A"},
                    {"text": "second", "label": "B"},
                ],
            },
            "fact1": "A frozen supporting fact.",
            "answerKey": "B",
        }
        converted = PREPARE.convert_record(raw, 0)
        self.assertEqual(converted["question"], "Which option?")
        self.assertEqual(converted["choices"], ["first", "second"])
        self.assertEqual(converted["answer"], "second")
        self.assertEqual(converted["context"], "A frozen supporting fact.")

    def test_invalid_answer_key_is_rejected(self) -> None:
        raw = {
            "id": "sample-1",
            "question": {
                "stem": "Which option?",
                "choices": [
                    {"text": "first", "label": "A"},
                    {"text": "second", "label": "B"},
                ],
            },
            "fact1": "A fact.",
            "answerKey": "C",
        }
        with self.assertRaises(ValueError):
            PREPARE.convert_record(raw, 0)

    def test_option_permutation_is_nonzero_and_answer_independent(self) -> None:
        sample = {
            "id": "fixed-id",
            "answer": "second",
            "answerKey": "B",
            "choices": ["first", "second", "third", "fourth"],
        }
        changed_answer = {**sample, "answer": "third", "answerKey": "C"}
        first = ROBUSTNESS.permute_options(sample)
        second = ROBUSTNESS.permute_options(changed_answer)
        self.assertNotEqual(first["choices"], sample["choices"])
        self.assertEqual(first["choices"], second["choices"])
        self.assertFalse(
            first["perturbation"]["uses_answer_to_choose_perturbation"]
        )

    def test_distractor_keeps_question_choices_and_answer(self) -> None:
        sample = {
            "id": "base",
            "question": "Question",
            "answer": "first",
            "choices": ["first", "second"],
            "context": "Relevant fact.",
        }
        distractor = {"id": "other", "context": "Different fact."}
        result = ROBUSTNESS.append_distractor(sample, distractor)
        self.assertEqual(result["question"], sample["question"])
        self.assertEqual(result["choices"], sample["choices"])
        self.assertEqual(result["answer"], sample["answer"])
        self.assertIn("Different fact.", result["context"])

    def test_holdout_labels_are_physically_separated(self) -> None:
        sample = {
            "id": "sample-1",
            "question": "Which option?",
            "context": "A frozen fact.",
            "choices": ["first", "second"],
            "answer": "second",
            "answerKey": "B",
        }
        inference, answer_key = ISOLATION.split_record(sample, 0)
        self.assertFalse(ISOLATION.LABEL_FIELDS.intersection(inference))
        self.assertEqual(answer_key["answer"], "second")
        self.assertEqual(answer_key["idx"], 0)

    def test_intervention_diagnostics_count_rescues_and_harms(self) -> None:
        outcomes = (
            (1, 0, "full-a", "plain-a"),
            (0, 1, "full-b", "plain-b"),
            (1, 1, "same-c", "same-c"),
            (0, 0, "same-d", "same-d"),
        )
        records = [
            {
                "variants": {
                    "full": {"acc": full, "prediction": full_prediction},
                    "plain_context": {
                        "acc": plain,
                        "prediction": plain_prediction,
                    },
                }
            }
            for full, plain, full_prediction, plain_prediction in outcomes
        ]
        diagnostics = analyze_holdout.intervention_diagnostics(records)
        self.assertEqual(diagnostics["changed_predictions"], 2)
        self.assertEqual(diagnostics["rescued"], 1)
        self.assertEqual(diagnostics["harmed"], 1)
        self.assertEqual(diagnostics["net_rescues"], 0)
        self.assertEqual(diagnostics["both_correct"], 1)
        self.assertEqual(diagnostics["both_wrong"], 1)


if __name__ == "__main__":
    unittest.main()
