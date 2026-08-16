"""Tests for sweep configuration validation and grid expansion."""

import unittest

from src.sweep import (
    _format_duration,
    _progress_line,
    expand_configurations,
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

    def test_rejects_parameters_not_supported_by_an_algorithm(self) -> None:
        self.config["algorithms"][0]["parameters"]["alpha"] = [0.1]

        with self.assertRaisesRegex(ValueError, "unsupported parameters"):
            validate_config(self.config)

    def test_progress_line_reports_percentage_elapsed_time_and_eta(self) -> None:
        progress = _progress_line(completed=5, total=10, elapsed=100.0)

        self.assertIn("5/10 (50.00%)", progress)
        self.assertIn("elapsed 01:40", progress)
        self.assertIn("ETA 01:40", progress)
        self.assertEqual(_format_duration(3_661), "1:01:01")


if __name__ == "__main__":
    unittest.main()
