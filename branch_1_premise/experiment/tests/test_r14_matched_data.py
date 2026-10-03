import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from gen_r14_matched_data import (  # noqa: E402
    ARMS,
    budget_ratio,
    instruction_id,
    select_balanced,
)


class R14MatchedDataTests(unittest.TestCase):
    def test_instruction_id_is_full_stable_sha256(self):
        value = instruction_id("same request")
        self.assertEqual(len(value), 64)
        self.assertEqual(value, instruction_id("same request"))
        self.assertNotEqual(value, instruction_id("Same request"))

    def test_selection_is_deterministic_and_exact_size(self):
        ids = [instruction_id(f"request {i}") for i in range(8)]
        lengths = {
            arm: {row_id: 100 + arm_i + row_i for row_i, row_id in enumerate(ids)}
            for arm_i, arm in enumerate(ARMS)
        }
        first, totals_first = select_balanced(ids, lengths, 5)
        second, totals_second = select_balanced(list(reversed(ids)), lengths, 5)
        self.assertEqual(first, second)
        self.assertEqual(totals_first, totals_second)
        self.assertEqual(len(first), 5)

    def test_budget_ratio(self):
        self.assertAlmostEqual(
            budget_ratio({"arm1": 100, "arm2": 100, "arm4": 101, "arm5": 100}),
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
