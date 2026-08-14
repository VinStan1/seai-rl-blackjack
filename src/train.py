"""Train first-visit Monte Carlo agents on Gymnasium Blackjack-v1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agents.monte_carlo import MonteCarloAgent
from src.environments.blackjack import BlackjackEnvironment
from src.metrics import summarize


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=Path("results/models"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, float | int | str]] = []

    for seed in arguments.seeds:
        agent = MonteCarloAgent(
            epsilon=arguments.epsilon,
            gamma=arguments.gamma,
            seed=seed,
        )
        started_at = time.perf_counter()
        with BlackjackEnvironment(sab=True) as environment:
            rewards = agent.train(environment, arguments.episodes)
        training_seconds = time.perf_counter() - started_at
        model_path = arguments.output_dir / f"monte_carlo_seed_{seed}.json"
        run = {
            "seed": seed,
            "episodes": arguments.episodes,
            "mean_training_reward": sum(rewards) / len(rewards),
            "training_seconds": training_seconds,
            "model": str(model_path),
        }
        agent.save(
            model_path,
            metadata={
                **run,
                "environment": "Blackjack-v1",
                "sab": True,
                "natural": False,
            },
        )
        runs.append(run)
        print(
            f"seed={seed} reward={run['mean_training_reward']:.4f} "
            f"seconds={training_seconds:.2f} model={model_path}"
        )

    summary = {
        "algorithm": "first_visit_monte_carlo_control",
        "environment": "Blackjack-v1",
        "runs": runs,
        "mean_training_reward": summarize(
            [float(run["mean_training_reward"]) for run in runs]
        ),
        "training_seconds": summarize(
            [float(run["training_seconds"]) for run in runs]
        ),
    }
    summary_path = arguments.output_dir.parent / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()