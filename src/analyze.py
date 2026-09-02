"""Turn a Blackjack sweep summary into a readable report and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ALGORITHM_NAMES = {
    "monte_carlo": "Monte Carlo",
    "sarsa": "SARSA",
    "q_learning": "Q-learning",
    "double_dqn": "Double DQN",
}
COLORS = {
    "monte_carlo": "#4C78A8",
    "sarsa": "#F58518",
    "q_learning": "#54A24B",
    "double_dqn": "#B279A2",
}
BASELINE_COLOR = "#B22222"
POLICY_PROJECTION_EPISODES = 10_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="latest",
        help="summary.json path, or 'latest' for the newest result",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to an analysis directory beside summary.json",
    )
    return parser.parse_args()


def resolve_summary_path(value: str, search_root: Path = Path("results/sweeps")) -> Path:
    """Resolve an explicit summary path or find the newest sweep summary."""
    if value != "latest":
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"summary not found: {path}")
        return path

    candidates = list(search_root.glob("*/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"no summary.json files found under {search_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("summary must contain a top-level JSON object")
    configurations = summary.get("configurations")
    runs = summary.get("runs")
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("summary contains no completed configurations")
    if not isinstance(runs, list):
        raise ValueError("summary.runs must be an array")
    episodes_by_configuration = {
        run["configuration_id"]: int(run["training"]["episodes"])
        for run in runs
        if run.get("status") == "completed"
        and isinstance(run.get("training"), dict)
        and "episodes" in run["training"]
    }
    for configuration in configurations:
        if "training_episodes" not in configuration:
            episodes = episodes_by_configuration.get(
                configuration.get("configuration_id")
            )
            if episodes is not None:
                configuration["training_episodes"] = episodes
    return summary


def _algorithm_name(name: str) -> str:
    return ALGORITHM_NAMES.get(name, name.replace("_", " ").title())


def _parameters_text(parameters: dict[str, Any]) -> str:
    parts: list[str] = []
    schedule_keys = {"epsilon_start", "epsilon_end", "epsilon_decay_fraction"}
    if "epsilon_start" in parameters and "epsilon_end" in parameters:
        fraction = float(parameters.get("epsilon_decay_fraction", 0.8))
        parts.append(
            f"epsilon={parameters['epsilon_start']}->{parameters['epsilon_end']} "
            f"linear ({fraction:.0%})"
        )
    elif "epsilon" in parameters:
        parts.append(f"epsilon={parameters['epsilon']}")
    for key in ("alpha", "gamma"):
        if key in parameters:
            parts.append(f"{key}={parameters[key]}")
    remaining = set(parameters) - schedule_keys - {"epsilon", "alpha", "gamma"}
    parts.extend(f"{key}={parameters[key]}" for key in sorted(remaining))
    return ", ".join(parts)


def _interval(configuration: dict[str, Any], metric: str) -> tuple[float, float]:
    lower, upper = configuration[metric]["confidence_interval_95"]
    return float(lower), float(upper)


def _best_by_algorithm(
    configurations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for configuration in configurations:
        algorithm = configuration["algorithm"]
        if (
            algorithm not in best
            or configuration["mean_reward"]["mean"]
            > best[algorithm]["mean_reward"]["mean"]
        ):
            best[algorithm] = configuration
    return best


def _paired_comparison(
    first: dict[str, Any],
    second: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compare two configurations through per-seed reward differences."""
    rewards: dict[str, dict[int, float]] = {}
    for configuration in (first, second):
        rewards[configuration["configuration_id"]] = {
            int(run["seed"]): float(run["evaluation"]["mean_reward"])
            for run in runs
            if run.get("status") == "completed"
            and run["configuration_id"] == configuration["configuration_id"]
        }

    first_rewards = rewards[first["configuration_id"]]
    second_rewards = rewards[second["configuration_id"]]
    common_seeds = sorted(set(first_rewards) & set(second_rewards))
    if not common_seeds:
        return None
    differences = [
        first_rewards[seed] - second_rewards[seed] for seed in common_seeds
    ]
    difference_mean = mean(differences)
    deviation = stdev(differences) if len(differences) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(len(differences))
    return {
        "first": first["algorithm"],
        "second": second["algorithm"],
        "count": len(differences),
        "mean_difference": difference_mean,
        "confidence_interval_95": [
            difference_mean - margin,
            difference_mean + margin,
        ],
    }


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _baseline_reward(baseline: dict[str, Any] | None) -> float | None:
    if not baseline or "mean_reward" not in baseline:
        return None
    return float(baseline["mean_reward"])


def _plot_baseline(axis: Any, baseline: dict[str, Any] | None) -> None:
    reward = _baseline_reward(baseline)
    if reward is None:
        return
    axis.axhline(
        reward,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=f"Stick-on-17 baseline ({reward:.4f})",
    )


def plot_configuration_performance(
    configurations: list[dict[str, Any]],
    output_path: Path,
    baseline: dict[str, Any] | None = None,
) -> None:
    """Plot reward points and intervals without zero-anchored bars."""
    plt = _pyplot()
    algorithms = [
        name
        for name in ALGORITHM_NAMES
        if any(item["algorithm"] == name for item in configurations)
    ]
    budgets = sorted({int(item["training_episodes"]) for item in configurations})
    palette = plt.get_cmap("viridis")
    budget_colors = {
        budget: palette(index / max(1, len(budgets) - 1))
        for index, budget in enumerate(budgets)
    }
    figure, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(5.5 * len(algorithms), 6),
        sharey=True,
        squeeze=False,
    )

    for axis, algorithm in zip(axes[0], algorithms, strict=True):
        matches = [item for item in configurations if item["algorithm"] == algorithm]
        parameter_sets = sorted(
            {json.dumps(item["parameters"], sort_keys=True) for item in matches}
        )
        labels = [
            _parameters_text(json.loads(parameters)) for parameters in parameter_sets
        ]
        offsets = {
            budget: (index - (len(budgets) - 1) / 2) * 0.14
            for index, budget in enumerate(budgets)
        }
        for budget in budgets:
            for position, serialized in enumerate(parameter_sets):
                item = next(
                    (
                        candidate
                        for candidate in matches
                        if candidate["training_episodes"] == budget
                        and json.dumps(candidate["parameters"], sort_keys=True)
                        == serialized
                    ),
                    None,
                )
                if item is None:
                    continue
                value = float(item["mean_reward"]["mean"])
                lower, upper = _interval(item, "mean_reward")
                axis.errorbar(
                    position + offsets[budget],
                    value,
                    yerr=[[value - lower], [upper - value]],
                    fmt="o",
                    color=budget_colors[budget],
                    capsize=3,
                    markersize=6,
                    label=f"{budget:,} episodes" if position == 0 else None,
                )
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
        axis.set_title(_algorithm_name(algorithm))
        axis.set_xlabel("Agent hyperparameters")
        axis.grid(axis="y", alpha=0.25)
        _plot_baseline(axis, baseline)

    axes[0][0].set_ylabel("Mean evaluation reward (higher is better)")
    axes[0][-1].legend(title="Training budget / reference", fontsize=8)
    figure.suptitle("Configuration performance with 95% confidence intervals")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_sample_efficiency(
    configurations: list[dict[str, Any]],
    output_path: Path,
    baseline: dict[str, Any] | None = None,
) -> None:
    """Plot evaluation reward as training experience increases."""
    plt = _pyplot()
    algorithms = [
        name
        for name in ALGORITHM_NAMES
        if any(item["algorithm"] == name for item in configurations)
    ]
    figure, axes = plt.subplots(
        1, len(algorithms), figsize=(5.5 * len(algorithms), 5), sharey=True
    )
    if len(algorithms) == 1:
        axes = [axes]

    for axis, algorithm in zip(axes, algorithms, strict=True):
        matches = [item for item in configurations if item["algorithm"] == algorithm]
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in matches:
            key = json.dumps(item["parameters"], sort_keys=True)
            groups.setdefault(key, []).append(item)
        for serialized, group in sorted(groups.items()):
            group.sort(key=lambda item: int(item["training_episodes"]))
            episodes = [int(item["training_episodes"]) for item in group]
            rewards = [float(item["mean_reward"]["mean"]) for item in group]
            intervals = [_interval(item, "mean_reward") for item in group]
            errors = [
                [
                    value - lower
                    for value, (lower, _) in zip(rewards, intervals, strict=True)
                ],
                [
                    upper - value
                    for value, (_, upper) in zip(rewards, intervals, strict=True)
                ],
            ]
            axis.errorbar(
                episodes,
                rewards,
                yerr=errors,
                marker="o",
                linewidth=1.5,
                capsize=2,
                label=_parameters_text(json.loads(serialized)),
            )
        axis.set_xscale("log")
        budgets = sorted({int(item["training_episodes"]) for item in matches})
        axis.set_xticks(budgets, [f"{budget // 1000}k" for budget in budgets])
        axis.set_title(_algorithm_name(algorithm))
        axis.set_xlabel("Training episodes")
        axis.grid(alpha=0.25)
        _plot_baseline(axis, baseline)
        axis.legend(fontsize=7)

    axes[0].set_ylabel("Mean evaluation reward (higher is better)")
    figure.suptitle("Sample efficiency: performance versus training experience")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_time(
    configurations: list[dict[str, Any]], output_path: Path
) -> None:
    """Plot training-time scaling and configuration-level observations."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 6))
    for algorithm in ALGORITHM_NAMES:
        matches = [item for item in configurations if item["algorithm"] == algorithm]
        if not matches:
            continue
        budgets = sorted({int(item["training_episodes"]) for item in matches})
        color = COLORS.get(algorithm, "#777777")
        axis.scatter(
            [int(item["training_episodes"]) for item in matches],
            [float(item["training_seconds"]["mean"]) for item in matches],
            color=color,
            alpha=0.28,
            s=35,
        )
        budget_means = [
            mean(
                float(item["training_seconds"]["mean"])
                for item in matches
                if int(item["training_episodes"]) == budget
            )
            for budget in budgets
        ]
        axis.plot(
            budgets,
            budget_means,
            color=color,
            marker="o",
            linewidth=2.3,
            label=_algorithm_name(algorithm),
        )

    all_budgets = sorted(
        {int(item["training_episodes"]) for item in configurations}
    )
    axis.set_xscale("log")
    axis.set_xticks(
        all_budgets, [f"{budget // 1000}k" for budget in all_budgets]
    )
    axis.set_xlabel("Training episodes")
    axis.set_ylabel("Mean training seconds per run")
    axis.set_title("Training-time scaling")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_performance_vs_training_time(
    configurations: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot reward against training time in a separate chart per budget."""
    plt = _pyplot()
    budgets = sorted({int(item["training_episodes"]) for item in configurations})
    output_paths: list[Path] = []
    for budget in budgets:
        matches = [
            item
            for item in configurations
            if int(item["training_episodes"]) == budget
        ]
        best_ids = {
            max(
                (item for item in matches if item["algorithm"] == algorithm),
                key=lambda item: float(item["mean_reward"]["mean"]),
            )["configuration_id"]
            for algorithm in ALGORITHM_NAMES
            if any(item["algorithm"] == algorithm for item in matches)
        }

        figure, axis = plt.subplots(figsize=(9, 6))
        for algorithm in ALGORITHM_NAMES:
            algorithm_matches = [
                item for item in matches if item["algorithm"] == algorithm
            ]
            if not algorithm_matches:
                continue
            color = COLORS.get(algorithm, "#777777")
            for item in algorithm_matches:
                reward = float(item["mean_reward"]["mean"])
                reward_lower, reward_upper = _interval(item, "mean_reward")
                seconds = float(item["training_seconds"]["mean"])
                seconds_lower, seconds_upper = _interval(item, "training_seconds")
                is_best = item["configuration_id"] in best_ids
                axis.errorbar(
                    seconds,
                    reward,
                    xerr=[[seconds - seconds_lower], [seconds_upper - seconds]],
                    yerr=[[reward - reward_lower], [reward_upper - reward]],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    alpha=0.9 if is_best else 0.45,
                    capsize=2,
                    markersize=9 if is_best else 6,
                    markeredgecolor="black" if is_best else color,
                    markeredgewidth=1.5 if is_best else 0.5,
                    zorder=3 if is_best else 2,
                )
            axis.scatter([], [], color=color, label=_algorithm_name(algorithm))

        axis.scatter(
            [],
            [],
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Best reward per algorithm",
        )
        axis.text(
            0.02,
            0.98,
            "Preferred direction: upper-left\n(higher reward, less time)",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
        )
        axis.set_xlabel("Mean training seconds per run (lower is better)")
        axis.set_ylabel("Mean evaluation reward (higher is better)")
        axis.set_title(f"Performance versus training time — {budget:,} episodes")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
        figure.tight_layout()

        output_path = output_dir / f"performance_vs_training_time_{budget}.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def _representative_best_run(
    configuration: dict[str, Any], runs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Select the run closest to the best configuration's aggregate reward."""
    matches = [
        run
        for run in runs
        if run.get("status") == "completed"
        and run.get("configuration_id") == configuration["configuration_id"]
        and isinstance(run.get("model"), str)
        and isinstance(run.get("evaluation"), dict)
        and "mean_reward" in run["evaluation"]
    ]
    if not matches:
        return None
    aggregate_reward = float(configuration["mean_reward"]["mean"])
    return min(
        matches,
        key=lambda run: (
            abs(float(run["evaluation"]["mean_reward"]) - aggregate_reward),
            int(run.get("seed", 0)),
            str(run.get("run_id", "")),
        ),
    )


def _load_tabular_q_values(model_path: Path) -> dict[tuple[int | bool, ...], list[float]]:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    table = payload.get("q_table")
    if not isinstance(table, list):
        raise ValueError("model does not contain a tabular q_table")

    q_values: dict[tuple[int | bool, ...], list[float]] = {}
    for entry in table:
        if not isinstance(entry, dict):
            continue
        state = entry.get("state")
        values = entry.get("values")
        if not isinstance(state, list) or len(state) != 3:
            continue
        if not isinstance(values, list) or len(values) < 2:
            continue
        q_values[(int(state[0]), int(state[1]), bool(state[2]))] = [
            float(value) for value in values
        ]
    return q_values


def _project_composition_actions(
    model_path: Path,
    algorithm: str,
    environment_config: dict[str, Any],
    *,
    episodes: int,
    seed: int = 91_000_000,
) -> dict[str, Any]:
    """Sample a composition-aware greedy policy and retain useful projections."""
    from src.agents import MonteCarloAgent, QLearningAgent, SarsaAgent
    from src.environments.factory import make_blackjack_environment

    if algorithm == "double_dqn":
        from src.agents.double_dqn import DoubleDQNAgent

        agent = DoubleDQNAgent.load(model_path)
    else:
        loaders = {
            "monte_carlo": MonteCarloAgent,
            "sarsa": SarsaAgent,
            "q_learning": QLearningAgent,
        }
        agent = loaders[algorithm].load(model_path)
    action_counts: dict[tuple[int | bool, ...], list[float]] = {}
    count_action_counts: dict[
        str, dict[tuple[int | bool, ...], list[float]]
    ] = {label: {} for label in ("negative", "neutral", "positive")}
    q_values = getattr(agent, "q_values", None)
    if q_values is not None:
        for state, values in q_values.items():
            if len(state) != 13 or len(values) < 2:
                continue
            action = values.index(max(values))
            visible_state = (int(state[0]), int(state[1]), bool(state[2]))
            counts = action_counts.setdefault(visible_state, [0.0, 0.0])
            counts[action] += 1.0
            count_group = _composition_true_count_group(state)
            grouped_counts = count_action_counts[count_group].setdefault(
                visible_state, [0.0, 0.0]
            )
            grouped_counts[action] += 1.0
        return {
            "action_counts": action_counts,
            "count_action_counts": count_action_counts,
            "episodes": 0,
            "source": "learned_states",
            "sample_count": sum(sum(values) for values in action_counts.values()),
        }

    with make_blackjack_environment(environment_config) as environment:
        for episode_index in range(episodes):
            state, _ = environment.reset(seed=seed if episode_index == 0 else None)
            while True:
                action = agent.select_action(state, explore=False)
                visible_state = (int(state[0]), int(state[1]), bool(state[2]))
                counts = action_counts.setdefault(visible_state, [0.0, 0.0])
                counts[action] += 1.0
                count_group = _composition_true_count_group(state)
                grouped_counts = count_action_counts[count_group].setdefault(
                    visible_state, [0.0, 0.0]
                )
                grouped_counts[action] += 1.0
                state, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
    return {
        "action_counts": action_counts,
        "count_action_counts": count_action_counts,
        "episodes": episodes,
        "source": "greedy_replay",
        "sample_count": sum(sum(values) for values in action_counts.values()),
    }


def _composition_true_count_group(state: tuple[int | bool, ...]) -> str:
    """Compress exact remaining-card counts into a familiar Hi-Lo count band."""
    if len(state) != 13:
        raise ValueError("finite-composition state must contain 13 values")
    remaining = [int(value) for value in state[3:]]
    hi_lo_weights = (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1)
    remaining_count = sum(
        weight * count for weight, count in zip(hi_lo_weights, remaining, strict=True)
    )
    running_count = -remaining_count
    decks_remaining = max(sum(remaining) / 52.0, 0.25)
    true_count = running_count / decks_remaining
    if true_count < -1.0:
        return "negative"
    if true_count > 1.0:
        return "positive"
    return "neutral"


def _composition_policies(
    summary: dict[str, Any], environment: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Load and replay one representative best model per available algorithm."""
    policies: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for algorithm in ("monte_carlo", "sarsa", "q_learning", "double_dqn"):
        matches = [
            item for item in summary["configurations"] if item["algorithm"] == algorithm
        ]
        if not matches:
            continue
        best_configuration = max(
            matches, key=lambda item: float(item["mean_reward"]["mean"])
        )
        run = _representative_best_run(best_configuration, summary["runs"])
        if run is None:
            continue
        model_path = Path(run["model"])
        if not model_path.is_file():
            continue
        evaluation = run.get("evaluation", {})
        projection_episodes = min(
            POLICY_PROJECTION_EPISODES,
            max(1, int(evaluation.get("episodes", POLICY_PROJECTION_EPISODES))),
        )
        projection = _project_composition_actions(
            model_path,
            algorithm,
            environment,
            episodes=projection_episodes,
        )
        if projection["action_counts"]:
            policies.append((best_configuration, run, projection))
    return policies


def _heatmap_matrix(
    action_counts: dict[tuple[int | bool, ...], list[float]],
    *,
    metric: str,
    usable_ace: bool,
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for player_total in range(4, 22):
        row = []
        for dealer_upcard in range(1, 11):
            values = action_counts.get((player_total, dealer_upcard, usable_ace))
            if values is None or sum(values) == 0:
                row.append(float("nan"))
            elif metric == "hit_frequency":
                row.append(values[1] / sum(values))
            elif metric == "coverage":
                row.append(math.log10(1.0 + sum(values)))
            else:
                raise ValueError(f"unknown policy heatmap metric: {metric}")
        matrix.append(row)
    return matrix


def _format_policy_axis(axis: Any, *, show_ylabel: bool) -> None:
    player_totals = list(range(4, 22))
    dealer_upcards = list(range(1, 11))
    axis.set_xticks(range(len(dealer_upcards)), dealer_upcards)
    axis.set_xlabel("Dealer upcard")
    axis.set_yticks(range(len(player_totals)), player_totals)
    if show_ylabel:
        axis.set_ylabel("Player total")
    axis.grid(False)


def _projection_source_label(projection: dict[str, Any]) -> str:
    if projection["source"] == "learned_states":
        return f"{projection['sample_count']:,.0f} learned exact states"
    return f"{projection['episodes']:,} greedy replay episodes"


def _plot_composition_policy_heatmaps(
    policies: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    output_path: Path,
) -> list[Path]:
    """Plot behavior, coverage, and count-conditioned finite-shoe projections."""
    plt = _pyplot()
    generated: list[Path] = []
    hit_color_map = plt.get_cmap("RdBu").with_extremes(bad="#d9d9d9")
    coverage_color_map = plt.get_cmap("viridis").with_extremes(bad="#d9d9d9")

    figure, axes = plt.subplots(
        len(policies),
        2,
        figsize=(12, max(6, 4.5 * len(policies))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row_index, (configuration, run, projection) in enumerate(policies):
        for column_index, usable_ace in enumerate((False, True)):
            axis = axes[row_index][column_index]
            image = axis.imshow(
                _heatmap_matrix(
                    projection["action_counts"],
                    metric="hit_frequency",
                    usable_ace=usable_ace,
                ),
                cmap=hit_color_map,
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                aspect="auto",
            )
            hand_type = "Usable ace" if usable_ace else "No usable ace"
            axis.set_title(
                f"{_algorithm_name(configuration['algorithm'])} — {hand_type}\n"
                f"{_projection_source_label(projection)}"
            )
            _format_policy_axis(axis, show_ylabel=column_index == 0)
    figure.subplots_adjust(
        left=0.08,
        right=0.84,
        top=0.72 if len(policies) == 1 else 0.9,
        bottom=0.06,
        hspace=0.38,
        wspace=0.12,
    )
    legend_axis = figure.add_axes((0.88, 0.15, 0.025, 0.7))
    figure.colorbar(
        image,
        cax=legend_axis,
        label="Fraction of contributing states/decisions that hit",
    )
    figure.suptitle("Finite-composition policy variation projected onto visible state", y=0.985)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    generated.append(output_path)

    coverage_path = output_path.with_name("best_policy_coverage_heatmap.png")
    maximum_coverage = max(
        math.log10(1.0 + sum(values))
        for _, _, projection in policies
        for values in projection["action_counts"].values()
    )
    figure, axes = plt.subplots(
        len(policies),
        2,
        figsize=(12, max(6, 5.0 * len(policies))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row_index, (configuration, run, projection) in enumerate(policies):
        for column_index, usable_ace in enumerate((False, True)):
            axis = axes[row_index][column_index]
            image = axis.imshow(
                _heatmap_matrix(
                    projection["action_counts"],
                    metric="coverage",
                    usable_ace=usable_ace,
                ),
                cmap=coverage_color_map,
                vmin=0.0,
                vmax=maximum_coverage,
                origin="lower",
                aspect="auto",
            )
            hand_type = "Usable ace" if usable_ace else "No usable ace"
            axis.set_title(
                f"{_algorithm_name(configuration['algorithm'])} — {hand_type}\n"
                f"{_projection_source_label(projection)}"
            )
            _format_policy_axis(axis, show_ylabel=column_index == 0)
    figure.subplots_adjust(
        left=0.08,
        right=0.84,
        top=0.72 if len(policies) == 1 else 0.9,
        bottom=0.06,
        hspace=0.52,
        wspace=0.12,
    )
    legend_axis = figure.add_axes((0.88, 0.15, 0.025, 0.7))
    figure.colorbar(
        image,
        cax=legend_axis,
        label="log10(1 + contributing states/decisions)",
    )
    figure.suptitle("Finite-composition policy projection coverage", y=0.985)
    figure.savefig(coverage_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    generated.append(coverage_path)

    count_labels = {
        "negative": "Negative true count (< -1)",
        "neutral": "Neutral true count (-1 to 1)",
        "positive": "Positive true count (> 1)",
    }
    for configuration, run, projection in policies:
        algorithm = configuration["algorithm"]
        count_path = output_path.with_name(
            f"best_policy_true_count_{algorithm}.png"
        )
        figure, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        image = None
        for row_index, usable_ace in enumerate((False, True)):
            for column_index, count_group in enumerate(
                ("negative", "neutral", "positive")
            ):
                axis = axes[row_index][column_index]
                image = axis.imshow(
                    _heatmap_matrix(
                        projection["count_action_counts"][count_group],
                        metric="hit_frequency",
                        usable_ace=usable_ace,
                    ),
                    cmap=hit_color_map,
                    vmin=0.0,
                    vmax=1.0,
                    origin="lower",
                    aspect="auto",
                )
                hand_type = "Usable ace" if usable_ace else "No usable ace"
                axis.set_title(f"{count_labels[count_group]} — {hand_type}")
                _format_policy_axis(axis, show_ylabel=column_index == 0)
        figure.subplots_adjust(
            left=0.07, right=0.87, top=0.88, bottom=0.08, hspace=0.3, wspace=0.14
        )
        legend_axis = figure.add_axes((0.91, 0.15, 0.02, 0.7))
        figure.colorbar(
            image,
            cax=legend_axis,
            label="Fraction of contributing states/decisions that hit",
        )
        figure.suptitle(
            f"{_algorithm_name(algorithm)} policy by Hi-Lo true-count band\n"
            f"representative seed {run.get('seed', 'unknown')}",
            y=0.985,
        )
        figure.savefig(count_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        generated.append(count_path)
    return generated


def plot_best_policy_heatmap(
    summary: dict[str, Any], output_path: Path
) -> list[Path]:
    """Plot exact tabular policies or finite-composition policy projections."""
    environment = summary.get("environment")
    variant = environment.get("variant", "standard") if isinstance(environment, dict) else "standard"
    if variant not in {"standard", "finite_hidden", "finite_composition"}:
        return []
    if variant == "finite_composition":
        policies = _composition_policies(summary, environment)
        if not policies:
            return []
        return _plot_composition_policy_heatmaps(policies, output_path)

    configurations = summary["configurations"]
    policies: list[
        tuple[dict[str, Any], dict[str, Any], dict[tuple[int | bool, ...], list[float]]]
    ] = []
    algorithms = ("monte_carlo", "sarsa", "q_learning")
    for algorithm in algorithms:
        matches = [
            item for item in configurations if item["algorithm"] == algorithm
        ]
        if not matches:
            continue
        best_configuration = max(
            matches, key=lambda item: float(item["mean_reward"]["mean"])
        )
        run = _representative_best_run(best_configuration, summary["runs"])
        if run is None:
            continue
        model_path = Path(run["model"])
        if not model_path.is_file():
            continue
        action_values = _load_tabular_q_values(model_path)
        if action_values:
            policies.append((best_configuration, run, action_values))
    if not policies:
        return []

    plt = _pyplot()
    from matplotlib.colors import BoundaryNorm, ListedColormap

    player_totals = list(range(4, 22))
    dealer_upcards = list(range(1, 11))
    color_map = ListedColormap(["#d9d9d9", "#e45756", "#4c78a8"])
    normalizer = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)
    figure, axes = plt.subplots(
        len(policies),
        2,
        figsize=(12, max(6, 4.5 * len(policies))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row_index, (configuration, run, q_values) in enumerate(policies):
        for axis, usable_ace in zip(
            axes[row_index], (False, True), strict=True
        ):
            actions = []
            for player_total in player_totals:
                row = []
                for dealer_upcard in dealer_upcards:
                    values = q_values.get((player_total, dealer_upcard, usable_ace))
                    row.append(-1 if values is None else values.index(max(values)))
                actions.append(row)
            image = axis.imshow(
                actions,
                cmap=color_map,
                norm=normalizer,
                origin="lower",
                aspect="auto",
            )
            hand_type = "Usable ace" if usable_ace else "No usable ace"
            axis.set_title(
                f"{_algorithm_name(configuration['algorithm'])} — {hand_type}\n"
                f"{int(configuration['training_episodes']):,} episodes, "
                f"representative seed {run.get('seed', 'unknown')}"
            )
            axis.set_xticks(range(len(dealer_upcards)), dealer_upcards)
            axis.set_xlabel("Dealer upcard")
            axis.set_yticks(range(len(player_totals)), player_totals)
            axis.grid(False)
        axes[row_index][0].set_ylabel("Player total")

    figure.subplots_adjust(
        left=0.08,
        right=0.84,
        top=0.82 if len(policies) == 1 else 0.9,
        bottom=0.06,
        hspace=0.38,
        wspace=0.12,
    )
    legend_axis = figure.add_axes((0.88, 0.15, 0.025, 0.7))
    figure.colorbar(
        image,
        cax=legend_axis,
        ticks=[-1, 0, 1],
        label="Greedy action",
    ).ax.set_yticklabels(["Unseen", "Stick", "Hit"])
    policy_algorithms = {item[0]["algorithm"] for item in policies}
    if policy_algorithms == {"double_dqn"}:
        title = "Double DQN projected policy"
    else:
        title = "Best policy from each tabular algorithm"
    figure.suptitle(title, y=0.985)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [output_path]


def write_configuration_csv(
    configurations: list[dict[str, Any]], output_path: Path
) -> None:
    fields = [
        "rank",
        "configuration_id",
        "algorithm",
        "training_episodes",
        "parameters",
        "seeds",
        "mean_reward",
        "reward_ci_95_lower",
        "reward_ci_95_upper",
        "win_rate",
        "training_seconds",
    ]
    ordered = sorted(
        configurations,
        key=lambda item: item["mean_reward"]["mean"],
        reverse=True,
    )
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(ordered, start=1):
            lower, upper = _interval(item, "mean_reward")
            writer.writerow(
                {
                    "rank": rank,
                    "configuration_id": item["configuration_id"],
                    "algorithm": item["algorithm"],
                    "training_episodes": item.get("training_episodes", ""),
                    "parameters": _parameters_text(item["parameters"]),
                    "seeds": item["completed_seeds"],
                    "mean_reward": item["mean_reward"]["mean"],
                    "reward_ci_95_lower": lower,
                    "reward_ci_95_upper": upper,
                    "win_rate": item["win_rate"]["mean"],
                    "training_seconds": item["training_seconds"]["mean"],
                }
            )


def build_report(
    summary: dict[str, Any],
    summary_path: Path,
    *,
    include_best_policy_heatmap: bool = False,
    policy_heatmap_paths: list[Path] | None = None,
) -> str:
    policy_heatmap_paths = policy_heatmap_paths or []
    include_best_policy_heatmap = (
        include_best_policy_heatmap or bool(policy_heatmap_paths)
    )
    configurations = summary["configurations"]
    runs = summary["runs"]
    metadata = summary.get("experiment_metadata", {})
    phase = metadata.get("phase") if isinstance(metadata, dict) else None
    seed_count = min(
        int(item.get("completed_seeds", 0)) for item in configurations
    )
    best_by_algorithm = _best_by_algorithm(configurations)
    overall_best = max(
        configurations, key=lambda item: item["mean_reward"]["mean"]
    )
    reward_lower, reward_upper = _interval(overall_best, "mean_reward")
    baseline = summary.get("baseline")
    environment = summary.get("environment", {})
    environment_variant = (
        environment.get("variant", "standard")
        if isinstance(environment, dict)
        else "standard"
    )
    lines = [
        f"# Analysis: {summary.get('experiment', 'Blackjack sweep')}",
        "",
        f"Environment variant: **{environment_variant}**.",
        "",
        "## Executive summary",
        "",
        (
            f"The highest observed final reward came from **{_algorithm_name(overall_best['algorithm'])}** "
            f"with `{_parameters_text(overall_best['parameters'])}`. Its mean reward was "
            f"**{overall_best['mean_reward']['mean']:.5f}** (approximate 95% CI "
            f"{reward_lower:.5f} to {reward_upper:.5f}), with a "
            f"{overall_best['win_rate']['mean']:.2%} win rate."
        ),
        "",
        (
            f"The sweep status is `{summary.get('status', 'unknown')}`: "
            f"{summary.get('completed_runs', 0)} of {summary.get('total_runs', 0)} runs "
            f"completed and {summary.get('failed_runs', 0)} failed."
        ),
        "",
        "![Configuration performance](configuration_performance.png)",
        "",
        "## Best configuration per algorithm",
        "",
        "| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |",
        "|---|---:|---|---:|---:|---:|",
    ]

    for algorithm, item in sorted(best_by_algorithm.items()):
        lower, upper = _interval(item, "mean_reward")
        lines.append(
            f"| {_algorithm_name(algorithm)} | {int(item.get('training_episodes', 0)):,} "
            f"| `{_parameters_text(item['parameters'])}` "
            f"| {item['mean_reward']['mean']:.5f} [{lower:.5f}, {upper:.5f}] "
            f"| {item['win_rate']['mean']:.2%} "
            f"| {item['training_seconds']['mean']:.2f} s |"
        )

    if isinstance(baseline, dict):
        lines.extend(
            [
                "",
                "## Literature baseline",
                "",
                (
                    f"The **stick-on-17** policy hits below 17 and sticks on 17 or above. "
                    f"On the same {int(baseline['episodes']):,} seeded evaluation episodes, "
                    f"its mean reward was **{float(baseline['mean_reward']):.5f}** with a "
                    f"{float(baseline['win_rate']):.2%} win rate."
                ),
                "",
                (
                    "Reference: Richard S. Sutton and Andrew G. Barto, "
                    "*Reinforcement Learning: An Introduction*, second edition, "
                    "Example 5.1: Blackjack (2018), "
                    "http://incompleteideas.net/book/RLbook2020.pdf."
                ),
                "",
            ]
        )

    interpretation_limits = []
    if seed_count < 10:
        interpretation_limits.append(
            f"- This experiment uses only {seed_count} training seeds per configuration. "
            "Use at least 10-20 fresh seeds for a stronger final comparison."
        )
    else:
        interpretation_limits.append(
            f"- Results use {seed_count} independent training seeds per configuration; "
            "retain the per-seed results when applying the final paired statistical test."
        )
    if phase == "final_validation":
        interpretation_limits.append(
            "- Hyperparameters were selected in a separate pilot sweep, reducing "
            "selection bias in this final evaluation."
        )
    else:
        interpretation_limits.append(
            "- The best settings were selected using the same evaluation results shown "
            "here. A separate final seed set reduces selection bias."
        )
    interpretation_limits.extend(
        [
            "- Confidence-interval overlap alone does not prove algorithms are equivalent.",
            "- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.",
        ]
    )
    if environment_variant == "standard":
        interpretation_limits.append(
            "- Evaluate environment variants before making claims about generalisation."
        )
    else:
        interpretation_limits.append(
            "- Compare this result with independently tuned standard and finite variants "
            "before attributing differences to the observation alone."
        )

    lines.extend(
        [
            "",
            "## Paired comparison of the selected configurations",
            "",
            (
                "Differences below are calculated seed by seed as the first algorithm minus "
                "the second. A positive value favors the first algorithm."
            ),
            "",
            "| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |",
            "|---|---:|---:|---|",
        ]
    )
    algorithms = sorted(best_by_algorithm)
    for index, first_name in enumerate(algorithms):
        for second_name in algorithms[index + 1 :]:
            comparison = _paired_comparison(
                best_by_algorithm[first_name], best_by_algorithm[second_name], runs
            )
            if comparison is None:
                continue
            lower, upper = comparison["confidence_interval_95"]
            if lower > 0 or upper < 0:
                interpretation = "interval excludes zero"
            else:
                interpretation = "difference is inconclusive at this precision"
            lines.append(
                f"| {_algorithm_name(first_name)} - {_algorithm_name(second_name)} "
                f"| {comparison['count']} | {comparison['mean_difference']:.5f} "
                f"[{lower:.5f}, {upper:.5f}] | {interpretation} |"
            )

    lines.extend(
        [
            "",
            "These normal-approximation intervals are descriptive; they are not a replacement "
            "for a pre-specified final statistical testing protocol.",
            "",
        ]
    )
    if phase == "final_validation":
        lines.extend(
            [
                "## Final configuration scope",
                "",
                "This stage intentionally evaluates only one configuration per algorithm. "
                "Hyperparameter sensitivity should be interpreted from the pilot grid search, "
                "not re-estimated from these final runs.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Sample efficiency",
                "",
                "![Sample efficiency](sample_efficiency.png)",
                "",
                "Each point is a separately trained agent at that episode budget. Higher reward with fewer episodes indicates better sample efficiency.",
                "",
            ]
        )

    if include_best_policy_heatmap:
        if environment_variant == "finite_composition":
            analyzed_algorithms = {
                item["algorithm"] for item in configurations
            }
            if analyzed_algorithms == {"double_dqn"}:
                policy_heading = "Projected Double DQN policy"
            else:
                policy_heading = "Projected finite-composition policies"
            policy_explanation = (
                "For each tabular agent, every learned exact-composition Q-table "
                "state contributes equally. A Double DQN has no enumerable state table, so its "
                f"projection instead uses the states visited during up to "
                f"{POLICY_PROJECTION_EPISODES:,} greedy replay episodes. The first heatmap "
                "reports the fraction of contributing states or decisions that hit, so values "
                "near 0.5 expose visible states whose action changes with exact shoe "
                "composition. The coverage heatmap reports log10(1 + contributing states or "
                "decisions), distinguishing broad evidence from rare states. The "
                "count-conditioned panels repeat the hit-frequency view for negative, neutral, "
                "and positive Hi-Lo true counts. Gray means that nothing contributed to that "
                "cell. These are compressed projections, not evidence that the policy ignores "
                "exact composition. Tabular coverage measures learned-state support, whereas "
                "Double DQN coverage measures greedy-replay visitation, so their absolute "
                "coverage values should not be compared directly."
            )
            policy_images = [
                "![Hit frequency by visible state](best_policy_heatmap.png)",
                "",
                "![Projection coverage](best_policy_coverage_heatmap.png)",
            ]
            generated_names = {path.name for path in policy_heatmap_paths}
            for algorithm in ("monte_carlo", "sarsa", "q_learning", "double_dqn"):
                filename = f"best_policy_true_count_{algorithm}.png"
                if filename in generated_names:
                    policy_images.extend(
                        [
                            "",
                            f"![{_algorithm_name(algorithm)} policy by true-count band]({filename})",
                        ]
                    )
        else:
            policy_heading = "Best policies by algorithm"
            policy_explanation = (
                "Each row shows the greedy policy for the highest-reward configuration "
                "of one tabular algorithm. Its representative seed is the completed run "
                "closest to that configuration's mean reward; gray cells were not present "
                "in that model's learned Q-table."
            )
            policy_images = [
                "![Best-per-algorithm action heatmaps](best_policy_heatmap.png)"
            ]
        lines.extend(
            [
                f"## {policy_heading}",
                "",
                *policy_images,
                "",
                policy_explanation,
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            *[
                line
                for budget in sorted(
                    {int(item["training_episodes"]) for item in configurations}
                )
                for line in (
                    f"### {budget:,} training episodes",
                    "",
                    f"![Performance versus training time at {budget:,} episodes]"
                    f"(performance_vs_training_time_{budget}.png)",
                    "",
                )
            ],
            (
                "Each chart holds the training budget fixed. Every point is one hyperparameter "
                "configuration, horizontal intervals show uncertainty in mean training time, "
                "and vertical intervals show uncertainty in mean evaluation reward. The preferred "
                "region is the upper-left; black outlines identify the best-reward configuration "
                "for each algorithm at that budget."
            ),
            "",
            "![Training time](training_time.png)",
            "",
            (
                "Training times were collected while independent runs could execute in parallel. "
                "They are useful operational measurements, but CPU contention means they should "
                "not be treated as clean single-process algorithm benchmarks."
            ),
            "",
            "## Interpretation limits and next steps",
            "",
            *interpretation_limits,
            "",
            "## Generated artifacts",
            "",
            f"- Source summary: `{summary_path}`",
            "- Full ranked table: `configuration_results.csv`",
            "- Final reward chart: `configuration_performance.png`",
            "- Sample-efficiency chart: `sample_efficiency.png`",
            "- Training-time chart: `training_time.png`",
            "- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget",
            *(
                [
                    "- Policy heatmaps: "
                    + ", ".join(f"`{path.name}`" for path in policy_heatmap_paths)
                ]
                if include_best_policy_heatmap
                else []
            ),
            "",
        ]
    )
    return "\n".join(lines)


def generate_analysis(summary_path: Path, output_dir: Path) -> list[Path]:
    """Generate all report artifacts and return their paths."""
    summary = load_summary(summary_path)
    configurations = summary["configurations"]
    output_dir.mkdir(parents=True, exist_ok=True)

    performance_path = output_dir / "configuration_performance.png"
    sample_efficiency_path = output_dir / "sample_efficiency.png"
    training_time_path = output_dir / "training_time.png"
    best_policy_path = output_dir / "best_policy_heatmap.png"
    csv_path = output_dir / "configuration_results.csv"
    report_path = output_dir / "analysis.md"

    baseline = summary.get("baseline")
    plot_configuration_performance(configurations, performance_path, baseline)
    plot_sample_efficiency(configurations, sample_efficiency_path, baseline)
    plot_training_time(configurations, training_time_path)
    performance_time_paths = plot_performance_vs_training_time(
        configurations, output_dir
    )
    policy_heatmap_paths = plot_best_policy_heatmap(summary, best_policy_path)
    write_configuration_csv(configurations, csv_path)
    report_path.write_text(
        build_report(
            summary,
            summary_path,
            policy_heatmap_paths=policy_heatmap_paths,
        ),
        encoding="utf-8",
    )
    return [
        report_path,
        performance_path,
        sample_efficiency_path,
        training_time_path,
        *performance_time_paths,
        *policy_heatmap_paths,
        csv_path,
    ]


def main() -> None:
    arguments = parse_arguments()
    summary_path = resolve_summary_path(arguments.summary)
    output_dir = arguments.output_dir or summary_path.parent / "analysis"
    generated = generate_analysis(summary_path, output_dir)
    print(f"summary={summary_path}")
    for path in generated:
        print(f"generated={path}")


if __name__ == "__main__":
    main()
