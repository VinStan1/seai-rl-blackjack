"""Tests for human-readable sweep analysis helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.analyze import (
    _best_by_algorithm,
    _paired_comparison,
    _score_per_training_second,
    build_report,
    load_summary,
    resolve_summary_path,
)


def configuration(identifier: str, algorithm: str, reward: float):
    return {
        "configuration_id": identifier,
        "algorithm": algorithm,
        "training_episodes": 100_000,
        "parameters": {"epsilon": 0.1, "gamma": 1.0},
        "completed_seeds": 2,
        "mean_reward": {
            "count": 2,
            "mean": reward,
            "standard_deviation": 0.01,
            "confidence_interval_95": [reward - 0.01, reward + 0.01],
        },
        "win_rate": {
            "count": 2,
            "mean": 0.43,
            "standard_deviation": 0.01,
            "confidence_interval_95": [0.42, 0.44],
        },
        "training_seconds": {
            "count": 2,
            "mean": 5.0,
            "standard_deviation": 0.1,
            "confidence_interval_95": [4.9, 5.1],
        },
    }


class AnalyzeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mc = configuration("mc_0", "monte_carlo", -0.07)
        self.q = configuration("q_0", "q_learning", -0.05)
        self.runs = [
            {
                "status": "completed",
                "configuration_id": "mc_0",
                "seed": seed,
                "evaluation": {"mean_reward": reward},
            }
            for seed, reward in [(0, -0.08), (1, -0.06)]
        ] + [
            {
                "status": "completed",
                "configuration_id": "q_0",
                "seed": seed,
                "evaluation": {"mean_reward": reward},
            }
            for seed, reward in [(0, -0.05), (1, -0.05)]
        ]
        self.summary = {
            "experiment": "test sweep",
            "status": "completed",
            "total_runs": 4,
            "completed_runs": 4,
            "failed_runs": 0,
            "configurations": [self.mc, self.q],
            "runs": self.runs,
            "baseline": {
                "name": "stick_on_17",
                "episodes": 100,
                "mean_reward": -0.08,
                "win_rate": 0.42,
            },
        }

    def test_selects_best_configuration_per_algorithm(self) -> None:
        better_mc = configuration("mc_1", "monte_carlo", -0.04)

        best = _best_by_algorithm([self.mc, better_mc, self.q])

        self.assertEqual(best["monte_carlo"]["configuration_id"], "mc_1")
        self.assertEqual(best["q_learning"]["configuration_id"], "q_0")

    def test_paired_comparison_matches_runs_by_seed(self) -> None:
        comparison = _paired_comparison(self.q, self.mc, self.runs)

        self.assertIsNotNone(comparison)
        self.assertEqual(comparison["count"], 2)
        self.assertAlmostEqual(comparison["mean_difference"], 0.02)

    def test_score_per_training_second_is_averaged_across_seeds(self) -> None:
        for run, seconds in zip(self.runs[:2], [2.0, 4.0], strict=True):
            run["training"] = {"seconds": seconds}

        score = _score_per_training_second("mc_0", self.runs)

        self.assertEqual(score["count"], 2)
        self.assertAlmostEqual(score["mean"], (-0.08 / 2.0 - 0.06 / 4.0) / 2.0)

    def test_report_contains_result_and_interpretation_sections(self) -> None:
        report = build_report(self.summary, Path("summary.json"))

        self.assertIn("Q-learning", report)
        self.assertIn("Paired comparison", report)
        self.assertIn("Interpretation limits", report)
        self.assertIn("configuration_performance.png", report)
        self.assertIn("sample_efficiency.png", report)
        self.assertIn("training_time.png", report)
        self.assertIn("performance_vs_training_time.png", report)
        self.assertIn("evaluation mean reward is divided", report)
        self.assertIn("stick-on-17", report)
        self.assertIn("Sutton", report)

    def test_final_report_does_not_claim_to_reestimate_sensitivity(self) -> None:
        self.summary["experiment_metadata"] = {"phase": "final_validation"}

        report = build_report(self.summary, Path("summary.json"))

        self.assertIn("Final configuration scope", report)
        self.assertIn("separate pilot sweep", report)
        self.assertNotIn("![Hyperparameter sensitivity]", report)

    def test_latest_summary_is_resolved_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old" / "summary.json"
            new = root / "new" / "summary.json"
            old.parent.mkdir()
            new.parent.mkdir()
            old.write_text(json.dumps(self.summary), encoding="utf-8")
            new.write_text(json.dumps(self.summary), encoding="utf-8")
            old.touch()
            new.touch()
            old_time = old.stat().st_mtime - 10
            os.utime(old, (old_time, old_time))

            selected = resolve_summary_path("latest", root)
            loaded = load_summary(selected)

        self.assertEqual(selected.name, "summary.json")
        self.assertEqual(selected.parent.name, "new")
        self.assertEqual(loaded["experiment"], "test sweep")


if __name__ == "__main__":
    unittest.main()
