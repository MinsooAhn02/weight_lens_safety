import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from check_narrowness import (  # noqa: E402
    ArmMetrics,
    candidate_pass,
    fixed_sample,
    validate_anchors,
)


def metrics(arm, distinct, vocab, length_sd, overlap):
    return ArmMetrics(
        arm=arm,
        source_rows=411,
        sample_rows=400,
        distinct_2=distinct,
        vocabulary_size=vocab,
        response_length_mean=100.0,
        response_length_sd=length_sd,
        pairwise_unigram_overlap=overlap,
    )


class NarrownessGateTests(unittest.TestCase):
    def test_anchor_validation_requires_multi_metric_direction(self):
        narrow = metrics("arm1", 0.50, 8_000, 30.0, 0.10)
        broad = metrics("arm2", 0.55, 8_500, 35.0, 0.08)
        passed, gaps = validate_anchors(narrow, broad)
        self.assertTrue(passed)
        self.assertTrue(all(value > 0 for value in gaps.values()))

    def test_candidate_gate_rejects_one_metric_spike(self):
        positions = {
            "distinct_2": 10.0,
            "vocabulary_size": 0.0,
            "response_length_sd": 0.0,
            "pairwise_unigram_overlap": 0.0,
        }
        passed, votes, _ = candidate_pass(positions, "broader_than_arm1")
        self.assertFalse(passed)
        self.assertEqual(votes, 1)

    def test_fixed_sample_is_order_independent(self):
        rows = [
            {"instruction": f"request {i}", "response": f"answer {i}", "arm": "arm1"}
            for i in range(410)
        ]
        forward = fixed_sample(rows)
        backward = fixed_sample(list(reversed(rows)))
        self.assertEqual(forward, backward)
        self.assertEqual(len(forward), 400)


if __name__ == "__main__":
    unittest.main()
