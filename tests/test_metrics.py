"""Tests for experiment summary statistics."""

import unittest

from src.metrics import summarize


class MetricsTests(unittest.TestCase):
    def test_summary_contains_mean_and_confidence_interval(self) -> None:
        summary = summarize([1.0, 2.0, 3.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        lower, upper = summary["confidence_interval_95"]
        self.assertLess(lower, 2.0)
        self.assertGreater(upper, 2.0)


if __name__ == "__main__":
    unittest.main()