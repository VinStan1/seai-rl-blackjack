"""Tests for human-readable sweep analysis helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.agents import QLearningAgent
from src.agents.double_dqn import DoubleDQNAgent
from src.analyze import (
    _best_by_algorithm,
    _composition_true_count_group,
    _paired_comparison,
    build_report,
    generate_analysis,
    load_summary,
    resolve_summary_path,
)
from src.environments.factory import make_blackjack_environment


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

    def test_report_contains_result_and_interpretation_sections(self) -> None:
        report = build_report(self.summary, Path("summary.json"))

        self.assertIn("Q-learning", report)
        self.assertIn("Paired comparison", report)
        self.assertIn("Interpretation limits", report)
        self.assertIn("configuration_performance.png", report)
        self.assertIn("sample_efficiency.png", report)
        self.assertIn("training_time.png", report)
        self.assertIn("performance_vs_training_time_100000.png", report)
        self.assertIn("preferred region is the upper-left", report)
        self.assertIn("stick-on-17", report)
        self.assertIn("Sutton", report)

    def test_final_report_does_not_claim_to_reestimate_sensitivity(self) -> None:
        self.summary["experiment_metadata"] = {"phase": "final_validation"}

        report = build_report(self.summary, Path("summary.json"))

        self.assertIn("Final configuration scope", report)
        self.assertIn("separate pilot sweep", report)
        self.assertNotIn("![Hyperparameter sensitivity]", report)

    def test_generates_best_policy_heatmap_for_each_available_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_directory = Path(temporary_directory)
            q_model_path = result_directory / "models" / "q_model.json"
            q_agent = QLearningAgent(seed=1)
            q_agent.q_values[(12, 2, False)] = [0.2, 0.7]
            q_agent.save(q_model_path)
            mc_model_path = result_directory / "models" / "mc_model.json"
            mc_agent = QLearningAgent(seed=2)
            mc_agent.q_values[(18, 10, False)] = [0.8, 0.1]
            mc_agent.save(mc_model_path)

            summary = json.loads(json.dumps(self.summary))
            summary["environment"] = {"variant": "finite_hidden"}
            summary["runs"] = [
                {
                    "status": "completed",
                    "configuration_id": "mc_0",
                    "seed": 2,
                    "evaluation": {"mean_reward": -0.07},
                    "model": str(mc_model_path),
                },
                {
                    "status": "completed",
                    "configuration_id": "q_0",
                    "seed": 1,
                    "evaluation": {"mean_reward": -0.05},
                    "model": str(q_model_path),
                }
            ]
            summary_path = result_directory / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            generated = generate_analysis(summary_path, result_directory / "analysis")
            heatmap_path = result_directory / "analysis" / "best_policy_heatmap.png"
            report = (result_directory / "analysis" / "analysis.md").read_text(
                encoding="utf-8"
            )

            self.assertIn(heatmap_path, generated)
            self.assertTrue(heatmap_path.is_file())
        self.assertIn("Best policies by algorithm", report)
        self.assertIn("highest-reward configuration of one tabular algorithm", report)

    def test_generates_projected_double_dqn_policy_heatmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_directory = Path(temporary_directory)
            model_path = result_directory / "models" / "double_dqn.pt"
            DoubleDQNAgent(seed=7).save(model_path)
            dqn_configuration = configuration("dqn_0", "double_dqn", -0.05)
            summary = json.loads(json.dumps(self.summary))
            summary["environment"] = {
                "variant": "finite_composition",
                "decks": 6,
                "penetration": 0.75,
                "natural": False,
                "sab": True,
            }
            summary["configurations"] = [dqn_configuration]
            summary["runs"] = [
                {
                    "status": "completed",
                    "configuration_id": "dqn_0",
                    "seed": 7,
                    "evaluation": {"episodes": 20, "mean_reward": -0.05},
                    "model": str(model_path),
                }
            ]
            summary_path = result_directory / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            generated = generate_analysis(summary_path, result_directory / "analysis")
            heatmap_path = result_directory / "analysis" / "best_policy_heatmap.png"
            report = (result_directory / "analysis" / "analysis.md").read_text(
                encoding="utf-8"
            )

            self.assertIn(heatmap_path, generated)
            self.assertTrue(heatmap_path.is_file())
            self.assertTrue(
                (result_directory / "analysis" / "best_policy_coverage_heatmap.png").is_file()
            )
            self.assertTrue(
                (result_directory / "analysis" / "best_policy_true_count_double_dqn.png").is_file()
            )
        self.assertIn("Projected Double DQN policy", report)
        self.assertIn("compressed projections", report)
        self.assertIn("fraction of contributing states or decisions", report)

    def test_generates_projected_tabular_composition_heatmaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_directory = Path(temporary_directory)
            model_path = result_directory / "models" / "q_learning.json"
            agent = QLearningAgent(seed=9)
            environment_config = {
                "variant": "finite_composition",
                "decks": 6,
                "penetration": 0.75,
                "natural": False,
                "sab": True,
            }
            with make_blackjack_environment(environment_config) as environment:
                state, _ = environment.reset(seed=91_000_000)
            agent.q_values[state] = [0.8, 0.1]
            agent.save(model_path)
            q_configuration = configuration("q_0", "q_learning", -0.05)
            summary = json.loads(json.dumps(self.summary))
            summary["environment"] = environment_config
            summary["configurations"] = [q_configuration]
            summary["runs"] = [
                {
                    "status": "completed",
                    "configuration_id": "q_0",
                    "seed": 9,
                    "evaluation": {"episodes": 20, "mean_reward": -0.05},
                    "model": str(model_path),
                }
            ]
            summary_path = result_directory / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            generated = generate_analysis(summary_path, result_directory / "analysis")
            report = (result_directory / "analysis" / "analysis.md").read_text(
                encoding="utf-8"
            )

        generated_names = {path.name for path in generated}
        self.assertIn("best_policy_heatmap.png", generated_names)
        self.assertIn("best_policy_coverage_heatmap.png", generated_names)
        self.assertIn("best_policy_true_count_q_learning.png", generated_names)
        self.assertIn("Projected finite-composition policies", report)
        self.assertIn("learned exact-composition Q-table", report)
        self.assertIn("negative, neutral, and positive Hi-Lo true", report)

    def test_groups_exact_compositions_by_hi_lo_true_count(self) -> None:
        neutral = (16, 10, False, 24, 24, 24, 24, 24, 24, 24, 24, 24, 96)
        positive = (16, 10, False, 24, 8, 24, 24, 24, 24, 24, 24, 24, 96)
        negative = (16, 10, False, 24, 24, 24, 24, 24, 24, 24, 24, 24, 80)

        self.assertEqual(_composition_true_count_group(neutral), "neutral")
        self.assertEqual(_composition_true_count_group(positive), "positive")
        self.assertEqual(_composition_true_count_group(negative), "negative")

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
