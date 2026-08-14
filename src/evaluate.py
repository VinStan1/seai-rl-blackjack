"""Evaluate saved Monte Carlo policies over independent Gymnasium episodes."""

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
    parser.add_argument("--model-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    return parser.parse_args()


def evaluate_model(
    model_path: Path,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float | int | str]:
    agent = MonteCarloAgent.load(model_path)
    rewards: list[float] = []
    action_count = 0
    elapsed_ns = 0

    with BlackjackEnvironment(sab=True) as environment:
        for episode_index in range(episodes):
            started_at = time.perf_counter_ns()
            episode = agent.generate_episode(
                environment,
                seed=seed + episode_index,
                explore=False,
            )
            elapsed_ns += time.perf_counter_ns() - started_at
            rewards.append(sum(step[2] for step in episode))
            action_count += len(episode)

    return {
        "model": str(model_path),
        "training_seed": agent.seed,
        "episodes": episodes,
        "mean_reward": sum(rewards) / episodes,
        "win_rate": sum(reward > 0 for reward in rewards) / episodes,
        "draw_rate": sum(reward == 0 for reward in rewards) / episodes,
        "loss_rate": sum(reward < 0 for reward in rewards) / episodes,
        "mean_inference_microseconds_per_action": elapsed_ns / action_count / 1_000,
    }


def main() -> None:
    arguments = parse_arguments()
    if arguments.episodes < 1:
        raise ValueError("episodes must be positive")

    model_paths = sorted(arguments.model_dir.glob("monte_carlo_seed_*.json"))
    if not model_paths:
        raise FileNotFoundError(
            f"no Monte Carlo models found in {arguments.model_dir}; run training first"
        )

    runs = [
        evaluate_model(
            model_path,
            episodes=arguments.episodes,
            seed=arguments.seed,
        )
        for model_path in model_paths
    ]
    report = {
        "algorithm": "first_visit_monte_carlo_control",
        "environment": "Blackjack-v1",
        "evaluation_seed": arguments.seed,
        "runs": runs,
        "mean_reward": summarize([float(run["mean_reward"]) for run in runs]),
        "win_rate": summarize([float(run["win_rate"]) for run in runs]),
        "mean_inference_microseconds_per_action": summarize(
            [float(run["mean_inference_microseconds_per_action"]) for run in runs]
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report={arguments.output}")


if __name__ == "__main__":
    main()