"""Retrain the best configuration per algorithm for a final comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analyze import (
    _algorithm_name,
    _paired_comparison,
    generate_analysis,
    load_summary,
    resolve_summary_path,
)

REQUIRED_ALGORITHMS = ("monte_carlo", "sarsa", "q_learning")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="latest",
        help="pilot summary.json path, or 'latest' for the newest pilot sweep",
    )
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--evaluation-episodes", type=int, default=1_000_000)
    parser.add_argument("--evaluation-seed", type=int, default=10_000_000)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(100, 110)),
        help="fresh training seeds; defaults to 100 through 109",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/final"))
    parser.add_argument(
        "--experiment-name",
        default="blackjack_final_million",
    )
    parser.add_argument(
        "--environment-variant",
        choices=("standard", "finite_hidden", "finite_hi_lo", "finite_composition"),
        default=None,
        help="optionally retrain the selected settings in another environment",
    )
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--penetration", type=float, default=0.75)
    return parser.parse_args()


def select_best_configurations(
    pilot_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select exactly one highest-mean-reward configuration per algorithm."""
    configurations = pilot_summary["configurations"]
    selected: list[dict[str, Any]] = []
    for algorithm in REQUIRED_ALGORITHMS:
        matches = [
            item for item in configurations if item["algorithm"] == algorithm
        ]
        if not matches:
            raise ValueError(f"pilot summary has no completed {algorithm} configuration")
        available_budgets = [
            int(item["training_episodes"])
            for item in matches
            if item.get("training_episodes") is not None
        ]
        if available_budgets:
            largest_budget = max(available_budgets)
            matches = [
                item
                for item in matches
                if int(item["training_episodes"]) == largest_budget
            ]
        selected.append(
            max(matches, key=lambda item: float(item["mean_reward"]["mean"]))
        )
    return selected


def build_final_config(
    pilot_summary: dict[str, Any],
    pilot_summary_path: Path,
    *,
    episodes: int,
    evaluation_episodes: int,
    evaluation_seed: int,
    seeds: list[int],
    workers: int,
    output_dir: Path,
    experiment_name: str,
    environment_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a scalar, best-per-algorithm sweep configuration."""
    if episodes < 1 or evaluation_episodes < 1:
        raise ValueError("training and evaluation episodes must be positive")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")

    source_config_path = pilot_summary_path.parent / "config.json"
    source_config: dict[str, Any] = {}
    if source_config_path.is_file():
        loaded = json.loads(source_config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            source_config = loaded

    source_environment = source_config.get(
        "environment", {"natural": False, "sab": True}
    )
    if not isinstance(source_environment, dict):
        raise ValueError("source environment must be an object")
    environment = {**source_environment, **(environment_override or {})}

    selected = select_best_configurations(pilot_summary)
    return {
        "experiment_name": experiment_name,
        "output_dir": str(output_dir),
        "workers": workers,
        "metadata": {
            "phase": "final_validation",
            "selection_source": str(pilot_summary_path),
            "selection_metric": "mean_evaluation_reward",
            "validation_environment": environment,
            "selected_pilot_configurations": [
                {
                    "algorithm": item["algorithm"],
                    "configuration_id": item["configuration_id"],
                    "parameters": item["parameters"],
                    "pilot_training_episodes": item.get("training_episodes"),
                    "pilot_mean_reward": item["mean_reward"]["mean"],
                }
                for item in selected
            ],
        },
        "environment": environment,
        "training": {"episodes": episodes, "seeds": seeds},
        "evaluation": {
            "episodes": evaluation_episodes,
            "seed": evaluation_seed,
        },
        "algorithms": [
            {
                "name": item["algorithm"],
                "parameters": item["parameters"],
            }
            for item in selected
        ],
    }


def write_final_selection(summary_path: Path) -> Path:
    """Write a machine-readable decision artifact for the final experiment."""
    summary = load_summary(summary_path)
    ordered = sorted(
        summary["configurations"],
        key=lambda item: float(item["mean_reward"]["mean"]),
        reverse=True,
    )
    winner = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    comparison = (
        _paired_comparison(winner, runner_up, summary["runs"])
        if runner_up is not None
        else None
    )
    conclusive = False
    if comparison is not None:
        lower, upper = comparison["confidence_interval_95"]
        conclusive = lower > 0 or upper < 0

    models = [
        run["model"]
        for run in summary["runs"]
        if run.get("status") == "completed"
        and run["configuration_id"] == winner["configuration_id"]
    ]
    decision = {
        "selected_algorithm": winner["algorithm"],
        "selected_algorithm_display_name": _algorithm_name(winner["algorithm"]),
        "selected_configuration_id": winner["configuration_id"],
        "parameters": winner["parameters"],
        "mean_reward": winner["mean_reward"],
        "win_rate": winner["win_rate"],
        "training_seconds": winner["training_seconds"],
        "models": models,
        "runner_up_comparison": comparison,
        "difference_is_conclusive_at_95_percent": conclusive,
        "recommendation": (
            "The selected algorithm has the highest observed reward and its paired "
            "95% interval versus the runner-up excludes zero."
            if conclusive
            else "The selected algorithm has the highest observed reward, but the "
            "paired 95% interval versus the runner-up includes zero; treat the "
            "winner as provisional rather than a proven superiority claim."
        ),
        "source_summary": str(summary_path),
    }
    output_path = summary_path.parent / "final_selection.json"
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    from src.sweep import run_sweep

    arguments = parse_arguments()
    pilot_summary_path = resolve_summary_path(arguments.summary)
    pilot_summary = load_summary(pilot_summary_path)
    environment_override = None
    if arguments.environment_variant is not None:
        environment_override = {"variant": arguments.environment_variant}
        if arguments.environment_variant.startswith("finite_"):
            environment_override.update(
                {"decks": arguments.decks, "penetration": arguments.penetration}
            )
    final_config = build_final_config(
        pilot_summary,
        pilot_summary_path,
        episodes=arguments.episodes,
        evaluation_episodes=arguments.evaluation_episodes,
        evaluation_seed=arguments.evaluation_seed,
        seeds=arguments.seeds,
        workers=arguments.workers,
        output_dir=arguments.output_dir,
        experiment_name=arguments.experiment_name,
        environment_override=environment_override,
    )

    print(f"pilot_summary={pilot_summary_path}")
    for specification in final_config["algorithms"]:
        print(
            f"selected={specification['name']} "
            f"parameters={json.dumps(specification['parameters'], sort_keys=True)}"
        )

    final_summary_path = run_sweep(
        final_config,
        config_path=pilot_summary_path,
        workers_override=arguments.workers,
    )
    generated = generate_analysis(
        final_summary_path, final_summary_path.parent / "analysis"
    )
    selection_path = write_final_selection(final_summary_path)
    print(f"final_selection={selection_path}")
    for path in generated:
        print(f"generated={path}")


if __name__ == "__main__":
    main()
