"""Tests for selecting and configuring the final experiment."""

import unittest
from pathlib import Path

from src.final_experiment import build_final_config, select_best_configurations


def result(
    identifier: str,
    algorithm: str,
    reward: float,
    training_episodes: int = 100_000,
    **parameters,
):
    return {
        "configuration_id": identifier,
        "algorithm": algorithm,
        "training_episodes": training_episodes,
        "parameters": parameters,
        "mean_reward": {"mean": reward},
    }


class FinalExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = {
            "configurations": [
                result("mc_bad", "monte_carlo", -0.08, epsilon=0.1, gamma=1.0),
                result("mc_best", "monte_carlo", -0.06, epsilon=0.2, gamma=1.0),
                result(
                    "sarsa_best",
                    "sarsa",
                    -0.05,
                    epsilon=0.2,
                    alpha=0.05,
                    gamma=1.0,
                ),
                result(
                    "q_lucky_short",
                    "q_learning",
                    -0.01,
                    training_episodes=20_000,
                    epsilon=0.2,
                    alpha=0.1,
                    gamma=1.0,
                ),
                result(
                    "q_best",
                    "q_learning",
                    -0.04,
                    epsilon=0.1,
                    alpha=0.05,
                    gamma=1.0,
                ),
            ]
        }

    def test_selects_one_best_configuration_per_required_algorithm(self) -> None:
        selected = select_best_configurations(self.summary)

        self.assertEqual(
            [item["configuration_id"] for item in selected],
            ["mc_best", "sarsa_best", "q_best"],
        )

    def test_builds_scalar_final_config_with_fresh_budget_and_seeds(self) -> None:
        config = build_final_config(
            self.summary,
            Path("pilot/summary.json"),
            episodes=1_000_000,
            evaluation_episodes=1_000_000,
            evaluation_seed=10_000_000,
            seeds=[100, 101],
            workers=2,
            output_dir=Path("results/final"),
            experiment_name="final",
        )

        self.assertEqual(config["training"]["episodes"], 1_000_000)
        self.assertEqual(config["evaluation"]["episodes"], 1_000_000)
        self.assertEqual(config["training"]["seeds"], [100, 101])
        self.assertEqual(len(config["algorithms"]), 3)
        self.assertEqual(config["algorithms"][0]["parameters"]["epsilon"], 0.2)
        self.assertEqual(config["metadata"]["phase"], "final_validation")

    def test_requires_all_three_algorithms(self) -> None:
        self.summary["configurations"] = [
            item
            for item in self.summary["configurations"]
            if item["algorithm"] != "q_learning"
        ]

        with self.assertRaisesRegex(ValueError, "q_learning"):
            select_best_configurations(self.summary)

    def test_can_retrain_selected_settings_in_finite_hidden_environment(self) -> None:
        config = build_final_config(
            self.summary,
            Path("pilot/summary.json"),
            episodes=500_000,
            evaluation_episodes=100_000,
            evaluation_seed=20_000_000,
            seeds=[200, 201],
            workers=2,
            output_dir=Path("results/final"),
            experiment_name="hidden_transfer",
            environment_override={
                "variant": "finite_hidden",
                "decks": 6,
                "penetration": 0.75,
            },
        )

        self.assertEqual(config["environment"]["variant"], "finite_hidden")
        self.assertEqual(config["environment"]["decks"], 6)
        self.assertEqual(config["environment"]["penetration"], 0.75)
        self.assertEqual(
            config["metadata"]["validation_environment"], config["environment"]
        )


if __name__ == "__main__":
    unittest.main()
