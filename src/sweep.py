"""Run reproducible Blackjack hyperparameter sweeps from a JSON configuration."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agents import MonteCarloAgent, QLearningAgent, SarsaAgent
from src.baselines import StickOnSeventeenPolicy
from src.environments.factory import make_blackjack_environment
from src.metrics import summarize

AGENTS = {
    "monte_carlo": MonteCarloAgent,
    "sarsa": SarsaAgent,
    "q_learning": QLearningAgent,
}
AGENT_NAMES = {*AGENTS, "double_dqn"}
ALLOWED_PARAMETERS = {
    "monte_carlo": {
        "epsilon",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_fraction",
        "gamma",
    },
    "sarsa": {
        "epsilon",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_fraction",
        "alpha",
        "gamma",
    },
    "q_learning": {
        "epsilon",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_fraction",
        "alpha",
        "gamma",
    },
    "double_dqn": {
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_fraction",
        "learning_rate",
        "gamma",
        "batch_size",
        "replay_capacity",
        "learning_starts",
        "train_frequency",
        "target_update_interval",
        "hidden_size",
        "gradient_clip",
        "decks",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="override the worker count in the configuration",
    )
    return parser.parse_args()


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a JSON sweep config, recursively resolving an optional ``extends``."""
    resolved_path = path.resolve()
    visited = set() if seen is None else seen
    if resolved_path in visited:
        raise ValueError(f"cyclic sweep configuration inheritance at {path}")
    visited.add(resolved_path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("the top-level JSON value must be an object")
    parent_name = loaded.pop("extends", None)
    if parent_name is None:
        return loaded
    if not isinstance(parent_name, str) or not parent_name:
        raise ValueError("extends must be a non-empty path string")
    parent = load_config(path.parent / parent_name, visited)
    return _merge_config(parent, loaded)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _format_duration(seconds: float) -> str:
    """Format a duration for compact progress output."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _progress_line(completed: int, total: int, elapsed: float) -> str:
    """Build an aggregate progress bar with elapsed time and an ETA."""
    ratio = completed / total
    width = 30
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    if completed:
        remaining = elapsed / completed * (total - completed)
        eta = _format_duration(remaining)
    else:
        eta = "estimating"
    return (
        f"[{bar}] {completed}/{total} ({ratio:6.2%}) "
        f"elapsed {_format_duration(elapsed)} ETA {eta}"
    )


def _display_progress(
    completed: int,
    total: int,
    started: float,
    *,
    interactive: bool,
) -> None:
    line = _progress_line(completed, total, time.perf_counter() - started)
    if interactive:
        print(f"\r{line:<100}", end="", flush=True)
    else:
        print(line, flush=True)


def validate_config(config: dict[str, Any]) -> None:
    """Fail early when a sweep configuration is incomplete or inconsistent."""
    training = config.get("training")
    evaluation = config.get("evaluation")
    algorithms = config.get("algorithms")
    environment = config.get("environment", {"natural": False, "sab": True})
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("algorithms must be a non-empty array")
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object")

    episode_budgets = training.get("episodes")
    if isinstance(episode_budgets, list):
        if not episode_budgets:
            raise ValueError("training.episodes must not be empty")
        for budget in episode_budgets:
            _positive_integer(budget, "training.episodes value")
        if len(episode_budgets) != len(set(episode_budgets)):
            raise ValueError("training.episodes must not contain duplicates")
    else:
        _positive_integer(episode_budgets, "training.episodes")
    _positive_integer(evaluation.get("episodes"), "evaluation.episodes")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("training.seeds must be a non-empty array of integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("training.seeds must not contain duplicates")

    for specification in algorithms:
        if not isinstance(specification, dict):
            raise ValueError("each algorithms entry must be an object")
        name = specification.get("name")
        if name not in AGENT_NAMES:
            raise ValueError(
                f"unknown algorithm {name!r}; expected one of {sorted(AGENT_NAMES)}"
            )
        if (
            name == "double_dqn"
            and environment.get("variant") != "finite_composition"
        ):
            raise ValueError(
                "double_dqn requires the finite_composition environment"
            )
        parameters = specification.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"parameters for {name} must be an object")
        unknown = set(parameters) - ALLOWED_PARAMETERS[name]
        if unknown:
            raise ValueError(f"unsupported parameters for {name}: {sorted(unknown)}")
        for parameter, values in parameters.items():
            if isinstance(values, list) and not values:
                raise ValueError(f"parameter grid {name}.{parameter} must not be empty")


def expand_configurations(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand every algorithm's parameter grid into concrete configurations."""
    expanded: list[dict[str, Any]] = []
    raw_budgets = config["training"]["episodes"]
    episode_budgets = raw_budgets if isinstance(raw_budgets, list) else [raw_budgets]
    for specification in config["algorithms"]:
        name = specification["name"]
        grid = specification.get("parameters", {})
        keys = list(grid)
        value_sets = [
            value if isinstance(value, list) else [value] for value in grid.values()
        ]
        combinations = itertools.product(*value_sets) if keys else [()]
        for values in combinations:
            parameters = dict(zip(keys, values, strict=True))
            for episodes in episode_budgets:
                expanded.append(
                    {
                        "configuration_id": f"{name}_{len(expanded):03d}",
                        "algorithm": name,
                        "parameters": parameters,
                        "training_episodes": episodes,
                    }
                )
    return expanded


def _make_agent(name: str, parameters: dict[str, Any], seed: int) -> Any:
    if name == "double_dqn":
        from src.agents.double_dqn import DoubleDQNAgent

        return DoubleDQNAgent(seed=seed, **parameters)
    return AGENTS[name](seed=seed, **parameters)


def _evaluate(
    agent: Any,
    environment_config: dict[str, Any],
    episodes: int,
    seed: int,
) -> dict[str, float | int]:
    rewards: list[float] = []
    action_count = 0
    unseen_state_action_count = 0
    selection_ns = 0
    q_values = getattr(agent, "q_values", None)
    with make_blackjack_environment(environment_config) as environment:
        for episode_index in range(episodes):
            state, _ = environment.reset(seed=seed if episode_index == 0 else None)
            episode_reward = 0.0
            while True:
                if q_values is not None and state not in q_values:
                    unseen_state_action_count += 1
                started = time.perf_counter_ns()
                action = agent.select_action(state, explore=False)
                selection_ns += time.perf_counter_ns() - started
                action_count += 1
                state, reward, terminated, truncated, _ = environment.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(episode_reward)

    result = {
        "episodes": episodes,
        "actions": action_count,
        "mean_actions_per_episode": action_count / episodes,
        "mean_reward": sum(rewards) / episodes,
        "win_rate": sum(reward > 0 for reward in rewards) / episodes,
        "draw_rate": sum(reward == 0 for reward in rewards) / episodes,
        "loss_rate": sum(reward < 0 for reward in rewards) / episodes,
        "mean_inference_microseconds_per_action": selection_ns / action_count / 1_000,
    }
    if q_values is not None:
        result.update(
            {
                "q_table_states": len(q_values),
                "unseen_state_actions": unseen_state_action_count,
                "unseen_state_action_rate": (
                    unseen_state_action_count / action_count
                ),
            }
        )
    return result


def _evaluate_baseline(
    environment_config: dict[str, Any], episodes: int, seed: int
) -> dict[str, Any]:
    """Evaluate the literature baseline on the same seeded evaluation episodes."""
    policy = StickOnSeventeenPolicy()
    return {
        "name": policy.name,
        "description": policy.description,
        "reference": policy.reference,
        "evaluation_seed": seed,
        **_evaluate(policy, environment_config, episodes, seed),
    }


def _execute_run(job: dict[str, Any]) -> dict[str, Any]:
    configuration = job["configuration"]
    seed = job["seed"]
    run_id = f"{configuration['configuration_id']}_seed_{seed}"
    output_dir = Path(job["output_dir"])
    report_path = output_dir / "runs" / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    agent = _make_agent(
        configuration["algorithm"], configuration["parameters"], seed
    )
    model_extension = getattr(agent, "model_extension", ".json")
    model_path = output_dir / "models" / f"{run_id}{model_extension}"
    with make_blackjack_environment(job["environment"]) as environment:
        training_started = time.perf_counter()
        rewards = agent.train(environment, configuration["training_episodes"])
        training_seconds = time.perf_counter() - training_started

    evaluation = _evaluate(
        agent,
        job["environment"],
        job["evaluation_episodes"],
        job["evaluation_seed"],
    )
    training = {
        "episodes": configuration["training_episodes"],
        "actions": agent.last_training_action_count,
        "mean_actions_per_episode": (
            agent.last_training_action_count
            / configuration["training_episodes"]
        ),
        "mean_reward": sum(rewards) / len(rewards),
        "seconds": training_seconds,
    }
    q_values = getattr(agent, "q_values", None)
    if q_values is not None:
        training["q_table_states"] = len(q_values)
    diagnostics = getattr(agent, "training_diagnostics", None)
    if diagnostics is not None:
        training.update(diagnostics())

    report = {
        "run_id": run_id,
        "status": "completed",
        **configuration,
        "seed": seed,
        "training": training,
        "evaluation": evaluation,
        "model": str(model_path),
        "total_seconds": time.perf_counter() - started,
    }
    agent.save(model_path, metadata=report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _aggregate(
    configurations: list[dict[str, Any]], runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for configuration in configurations:
        matching = [
            run
            for run in runs
            if run.get("status") == "completed"
            and run["configuration_id"] == configuration["configuration_id"]
        ]
        if not matching:
            continue
        aggregates.append(
            {
                **configuration,
                "completed_seeds": len(matching),
                "mean_reward": summarize(
                    [float(run["evaluation"]["mean_reward"]) for run in matching]
                ),
                "win_rate": summarize(
                    [float(run["evaluation"]["win_rate"]) for run in matching]
                ),
                "training_seconds": summarize(
                    [float(run["training"]["seconds"]) for run in matching]
                ),
            }
        )
    aggregates.sort(key=lambda item: item["mean_reward"]["mean"], reverse=True)
    return aggregates


def _write_summary(
    path: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    configurations: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    total_runs: int,
    workers: int,
    started_at: str,
    baseline: dict[str, Any],
) -> None:
    completed = sum(run.get("status") == "completed" for run in runs)
    failed = sum(run.get("status") == "failed" for run in runs)
    if completed + failed < total_runs:
        status = "running"
    elif failed:
        status = "completed_with_failures"
    else:
        status = "completed"
    payload = {
        "experiment": config.get("experiment_name", "blackjack_sweep"),
        "experiment_metadata": config.get("metadata", {}),
        "environment": config.get(
            "environment", {"natural": False, "sab": True}
        ),
        "config_path": str(config_path),
        "started_at_utc": started_at,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "workers": workers,
        "status": status,
        "total_runs": total_runs,
        "completed_runs": completed,
        "failed_runs": failed,
        "baseline": baseline,
        "configurations": _aggregate(configurations, runs),
        "runs": sorted(runs, key=lambda run: run["run_id"]),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_sweep(
    config: dict[str, Any],
    *,
    config_path: Path,
    workers_override: int | None = None,
) -> Path:
    """Execute a validated sweep and return its generated summary path."""
    validate_config(config)

    # Constructor validation catches invalid ranges before worker processes start.
    configurations = expand_configurations(config)
    for configuration in configurations:
        _make_agent(
            configuration["algorithm"], configuration["parameters"], seed=0
        )

    workers = (
        workers_override
        if workers_override is not None
        else config.get("workers", 1)
    )
    workers = _positive_integer(workers, "workers")
    seeds = config["training"]["seeds"]
    total_runs = len(configurations) * len(seeds)
    environment = config.get("environment", {"natural": False, "sab": True})
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object")
    evaluation_episodes = config["evaluation"]["episodes"]
    evaluation_seed = config["evaluation"].get("seed", 10_000)
    baseline = _evaluate_baseline(
        environment,
        evaluation_episodes,
        evaluation_seed,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    experiment_name = config.get("experiment_name", "blackjack_sweep")
    output_root = Path(config.get("output_dir", "results/sweeps"))
    output_dir = output_root / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    summary_path = output_dir / "summary.json"
    started_at = datetime.now(UTC).isoformat()

    jobs = [
        {
            "configuration": configuration,
            "seed": seed,
            "environment": environment,
            "evaluation_episodes": evaluation_episodes,
            "evaluation_seed": evaluation_seed,
            "output_dir": str(output_dir),
        }
        for configuration in configurations
        for seed in seeds
    ]
    runs: list[dict[str, Any]] = []
    _write_summary(
        summary_path,
        config_path=config_path,
        config=config,
        configurations=configurations,
        runs=runs,
        total_runs=total_runs,
        workers=workers,
        started_at=started_at,
        baseline=baseline,
    )
    print(f"experiment={output_dir} configurations={len(configurations)} runs={total_runs}")
    sweep_started = time.perf_counter()
    interactive = sys.stdout.isatty()
    _display_progress(0, total_runs, sweep_started, interactive=interactive)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_execute_run, job): job for job in jobs}
        pending = set(futures)
        while pending:
            completed_futures, pending = wait(
                pending,
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            if not completed_futures:
                if interactive:
                    _display_progress(
                        len(runs), total_runs, sweep_started, interactive=True
                    )
                continue

            if interactive:
                print()
            for future in completed_futures:
                job = futures[future]
                configuration = job["configuration"]
                run_id = f"{configuration['configuration_id']}_seed_{job['seed']}"
                try:
                    run = future.result()
                    print(
                        f"completed {run_id} "
                        f"reward={run['evaluation']['mean_reward']:.4f}"
                    )
                except Exception as error:  # keep a long sweep running
                    run = {
                        "run_id": run_id,
                        "status": "failed",
                        **configuration,
                        "seed": job["seed"],
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": "".join(traceback.format_exception(error)),
                    }
                    print(f"failed {run_id}: {run['error']}")
                    failed_report = output_dir / "runs" / f"{run_id}.json"
                    failed_report.parent.mkdir(parents=True, exist_ok=True)
                    failed_report.write_text(
                        json.dumps(run, indent=2), encoding="utf-8"
                    )
                runs.append(run)
                _write_summary(
                    summary_path,
                    config_path=config_path,
                    config=config,
                    configurations=configurations,
                    runs=runs,
                    total_runs=total_runs,
                    workers=workers,
                    started_at=started_at,
                    baseline=baseline,
                )
            _display_progress(
                len(runs), total_runs, sweep_started, interactive=interactive
            )

    if interactive:
        print()
    print(f"summary={summary_path}")
    if any(run["status"] == "failed" for run in runs):
        raise RuntimeError(f"one or more sweep runs failed; see {summary_path}")
    return summary_path


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    run_sweep(
        config,
        config_path=arguments.config,
        workers_override=arguments.workers,
    )


if __name__ == "__main__":
    main()
