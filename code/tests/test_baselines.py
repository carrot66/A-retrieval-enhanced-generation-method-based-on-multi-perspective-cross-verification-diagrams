from __future__ import annotations

import unittest

from ugcp_v5.baseline_runner import parse_option, prompt_for


class BaselinePromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "context": "The context supports the second choice.",
            "question": "Which choice is supported?",
            "choices": ["first", "second"],
        }

    def test_without_rag_excludes_context(self) -> None:
        prompt = prompt_for("without_rag", self.sample)
        self.assertNotIn(self.sample["context"], prompt)
        self.assertIn(self.sample["question"], prompt)

    def test_rag_includes_context(self) -> None:
        self.assertIn(self.sample["context"], prompt_for("rag", self.sample))

    def test_kre_uses_dependent_role(self) -> None:
        prompt = prompt_for("kre", self.sample)
        self.assertIn("solely based on the provided context", prompt)

    def test_parser_handles_identifier_letter_and_text(self) -> None:
        choices = self.sample["choices"]
        self.assertEqual(parse_option("OPTION_ID=1", choices)[0], 1)
        self.assertEqual(parse_option("The answer is B: second", choices)[0], 1)
        self.assertEqual(parse_option("second", choices)[0], 1)
        self.assertIsNone(parse_option("None", choices)[0])


if __name__ == "__main__":
    unittest.main()
