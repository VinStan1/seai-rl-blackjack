"""Tests for sweep configuration validation and grid expansion."""

import json
import tempfile
import unittest
from pathlib import Path

from src.baselines import StickOnSeventeenPolicy
from src.sweep import (
    _evaluate,
    _evaluate_baseline,
    _format_duration,
    _progress_line,
    expand_configurations,
    load_config,
    validate_config,
)


class SweepConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "training": {"episodes": 10, "seeds": [0, 1]},
            "evaluation": {"episodes": 5, "seed": 100},
            "algorithms": [
                {
                    "name": "monte_carlo",
                    "parameters": {"epsilon": [0.1, 0.2], "gamma": 1.0},
                },
                {
                    "name": "sarsa",
                    "parameters": {"epsilon": 0.1, "alpha": [0.05, 0.1]},
                },
            ],
        }

    def test_grid_expands_cartesian_product_per_algorithm(self) -> None:
        validate_config(self.config)
        configurations = expand_configurations(self.config)

        self.assertEqual(len(configurations), 4)
        self.assertEqual(configurations[0]["parameters"]["epsilon"], 0.1)
        self.assertEqual(configurations[-1]["parameters"]["alpha"], 0.1)
        self.assertEqual(configurations[0]["training_episodes"], 10)

    def test_training_episode_budgets_are_another_grid_dimension(self) -> None:
        self.config["training"]["episodes"] = [20, 50, 100, 200]

        validate_config(self.config)
        configurations = expand_configurations(self.config)

        self.assertEqual(len(configurations), 16)
        self.assertEqual(
            {
                item["training_episodes"]
                for item in configurations
                if item["algorithm"] == "monte_carlo"
            },
            {20, 50, 100, 200},
        )

    def test_config_extends_and_deep_merges_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "base",
                        "environment": {"natural": False, "sab": True},
                        "training": {"episodes": 10, "seeds": [1]},
                    }
                ),
                encoding="utf-8",
            )
            child = root / "child.json"
            child.write_text(
                json.dumps(
                    {
                        "extends": "base.json",
                        "experiment_name": "finite",
                        "environment": {"variant": "finite_hidden"},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(child)

        self.assertEqual(config["experiment_name"], "finite")
        self.assertEqual(config["environment"]["variant"], "finite_hidden")
        self.assertTrue(config["environment"]["sab"])
        self.assertEqual(config["training"]["episodes"], 10)

    def test_rejects_parameters_not_supported_by_an_algorithm(self) -> None:
        self.config["algorithms"][0]["parameters"]["alpha"] = [0.1]

        with self.assertRaisesRegex(ValueError, "unsupported parameters"):
            validate_config(self.config)

    def test_accepts_explicit_epsilon_schedule_parameters(self) -> None:
        self.config["algorithms"] = [
            {
                "name": "q_learning",
                "parameters": {
                    "epsilon_start": [1.0],
                    "epsilon_end": [0.05],
                    "epsilon_decay_fraction": [0.8],
                    "alpha": [0.005, 0.01],
                    "gamma": [1.0],
                },
            }
        ]

        validate_config(self.config)
        configurations = expand_configurations(self.config)

        self.assertEqual(len(configurations), 2)
        self.assertEqual(configurations[0]["parameters"]["epsilon_start"], 1.0)

    def test_progress_line_reports_percentage_elapsed_time_and_eta(self) -> None:
        progress = _progress_line(completed=5, total=10, elapsed=100.0)

        self.assertIn("5/10 (50.00%)", progress)
        self.assertIn("elapsed 01:40", progress)
        self.assertIn("ETA 01:40", progress)
        self.assertEqual(_format_duration(3_661), "1:01:01")

    def test_literature_baseline_uses_requested_evaluation_protocol(self) -> None:
        baseline = _evaluate_baseline(
            {"natural": False, "sab": True}, episodes=20, seed=500
        )

        self.assertEqual(baseline["name"], "stick_on_17")
        self.assertEqual(baseline["episodes"], 20)
        self.assertEqual(baseline["evaluation_seed"], 500)
        self.assertIn("Sutton", baseline["reference"]["authors"])

    def test_literature_baseline_sticks_on_seventeen(self) -> None:
        policy = StickOnSeventeenPolicy()

        self.assertEqual(policy.select_action((16, 10, False)), 1)
        self.assertEqual(policy.select_action((17, 10, False)), 0)
        self.assertEqual(policy.select_action((21, 1, True)), 0)

    def test_evaluation_supports_all_finite_observation_variants(self) -> None:
        policy = StickOnSeventeenPolicy()
        for variant in (
            "finite_hidden",
            "finite_hi_lo",
            "finite_composition",
        ):
            with self.subTest(variant=variant):
                result = _evaluate(
                    policy,
                    {
                        "variant": variant,
                        "decks": 6,
                        "penetration": 0.75,
                        "natural": False,
                        "sab": True,
                    },
                    episodes=20,
                    seed=600,
                )

                self.assertEqual(result["episodes"], 20)
                self.assertAlmostEqual(
                    result["win_rate"]
                    + result["draw_rate"]
                    + result["loss_rate"],
                    1.0,
                )


if __name__ == "__main__":
    unittest.main()
