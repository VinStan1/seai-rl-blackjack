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
}
COLORS = {
    "monte_carlo": "#4C78A8",
    "sarsa": "#F58518",
    "q_learning": "#54A24B",
}


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
    return summary


def _algorithm_name(name: str) -> str:
    return ALGORITHM_NAMES.get(name, name.replace("_", " ").title())


def _parameters_text(parameters: dict[str, Any]) -> str:
    preferred_order = ("epsilon", "alpha", "gamma")
    ordered = [key for key in preferred_order if key in parameters]
    ordered.extend(sorted(set(parameters) - set(ordered)))
    return ", ".join(f"{key}={parameters[key]}" for key in ordered)


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


def plot_configuration_performance(
    configurations: list[dict[str, Any]], output_path: Path
) -> None:
    """Plot every configuration's final reward and 95% confidence interval."""
    plt = _pyplot()
    ordered = sorted(configurations, key=lambda item: item["mean_reward"]["mean"])
    means = [float(item["mean_reward"]["mean"]) for item in ordered]
    intervals = [_interval(item, "mean_reward") for item in ordered]
    errors = [
        [value - lower for value, (lower, _) in zip(means, intervals, strict=True)],
        [upper - value for value, (_, upper) in zip(means, intervals, strict=True)],
    ]
    labels = [
        f"{_algorithm_name(item['algorithm'])}: {_parameters_text(item['parameters'])}"
        for item in ordered
    ]
    colors = [COLORS.get(item["algorithm"], "#777777") for item in ordered]

    figure, axis = plt.subplots(figsize=(12, max(7, len(ordered) * 0.38)))
    positions = list(range(len(ordered)))
    axis.barh(positions, means, color=colors, alpha=0.85)
    axis.errorbar(
        means,
        positions,
        xerr=errors,
        fmt="none",
        ecolor="#202020",
        elinewidth=1.2,
        capsize=3,
    )
    axis.set_yticks(positions, labels, fontsize=8)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Mean evaluation reward (95% CI across seeds)")
    axis.set_title("Final performance of every hyperparameter configuration")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_hyperparameter_sensitivity(
    configurations: list[dict[str, Any]], output_path: Path
) -> None:
    """Show how epsilon and alpha affect final evaluation reward."""
    plt = _pyplot()
    algorithms = sorted(
        {item["algorithm"] for item in configurations},
        key=lambda name: list(ALGORITHM_NAMES).index(name)
        if name in ALGORITHM_NAMES
        else 999,
    )
    figure, axes = plt.subplots(
        1, len(algorithms), figsize=(5.2 * len(algorithms), 4.8), squeeze=False
    )

    for axis, algorithm in zip(axes[0], algorithms, strict=True):
        matches = [item for item in configurations if item["algorithm"] == algorithm]
        line_groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
        for item in matches:
            parameters = item["parameters"]
            key = (parameters.get("alpha"), parameters.get("gamma"))
            line_groups.setdefault(key, []).append(item)

        for (alpha, gamma), group in sorted(
            line_groups.items(), key=lambda item: str(item[0])
        ):
            group.sort(key=lambda item: float(item["parameters"].get("epsilon", 0)))
            x_values = [float(item["parameters"].get("epsilon", 0)) for item in group]
            y_values = [float(item["mean_reward"]["mean"]) for item in group]
            intervals = [_interval(item, "mean_reward") for item in group]
            y_errors = [
                [
                    value - lower
                    for value, (lower, _) in zip(y_values, intervals, strict=True)
                ],
                [
                    upper - value
                    for value, (_, upper) in zip(y_values, intervals, strict=True)
                ],
            ]
            label_parts = []
            if alpha is not None:
                label_parts.append(f"alpha={alpha}")
            if len({key[1] for key in line_groups}) > 1:
                label_parts.append(f"gamma={gamma}")
            axis.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=", ".join(label_parts) or f"gamma={gamma}",
            )

        axis.set_title(_algorithm_name(algorithm))
        axis.set_xlabel("Epsilon")
        axis.set_ylabel("Mean evaluation reward")
        axis.grid(alpha=0.25)
        if len(line_groups) > 1:
            axis.legend(fontsize=8)

    figure.suptitle("Hyperparameter sensitivity", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_reward_vs_training_time(
    configurations: list[dict[str, Any]], output_path: Path
) -> None:
    """Plot the final-performance/training-time trade-off."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 6))
    best = _best_by_algorithm(configurations)

    for algorithm in sorted({item["algorithm"] for item in configurations}):
        matches = [item for item in configurations if item["algorithm"] == algorithm]
        axis.scatter(
            [float(item["training_seconds"]["mean"]) for item in matches],
            [float(item["mean_reward"]["mean"]) for item in matches],
            s=70,
            alpha=0.8,
            color=COLORS.get(algorithm, "#777777"),
            label=_algorithm_name(algorithm),
        )
        selected = best[algorithm]
        axis.annotate(
            "best " + _algorithm_name(algorithm),
            (
                float(selected["training_seconds"]["mean"]),
                float(selected["mean_reward"]["mean"]),
            ),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_xlabel("Mean training seconds per seed")
    axis.set_ylabel("Mean evaluation reward")
    axis.set_title("Final performance versus training time")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_configuration_csv(
    configurations: list[dict[str, Any]], output_path: Path
) -> None:
    fields = [
        "rank",
        "configuration_id",
        "algorithm",
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
                    "parameters": _parameters_text(item["parameters"]),
                    "seeds": item["completed_seeds"],
                    "mean_reward": item["mean_reward"]["mean"],
                    "reward_ci_95_lower": lower,
                    "reward_ci_95_upper": upper,
                    "win_rate": item["win_rate"]["mean"],
                    "training_seconds": item["training_seconds"]["mean"],
                }
            )


def build_report(summary: dict[str, Any], summary_path: Path) -> str:
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
    lines = [
        f"# Analysis: {summary.get('experiment', 'Blackjack sweep')}",
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
        "| Algorithm | Parameters | Mean reward (95% CI) | Win rate | Training time |",
        "|---|---|---:|---:|---:|",
    ]

    for algorithm, item in sorted(best_by_algorithm.items()):
        lower, upper = _interval(item, "mean_reward")
        lines.append(
            f"| {_algorithm_name(algorithm)} | `{_parameters_text(item['parameters'])}` "
            f"| {item['mean_reward']['mean']:.5f} [{lower:.5f}, {upper:.5f}] "
            f"| {item['win_rate']['mean']:.2%} "
            f"| {item['training_seconds']['mean']:.2f} s |"
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
            "- This summary records only final performance. Add training checkpoints to compare learning speed and sample efficiency directly.",
            "- Evaluate environment variants before making claims about generalisation.",
        ]
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
                "## Hyperparameter sensitivity",
                "",
                "![Hyperparameter sensitivity](hyperparameter_sensitivity.png)",
                "",
            ]
        )
        for algorithm in sorted(best_by_algorithm):
            matches = [
                item for item in configurations if item["algorithm"] == algorithm
            ]
            best = max(matches, key=lambda item: item["mean_reward"]["mean"])
            worst = min(matches, key=lambda item: item["mean_reward"]["mean"])
            spread = best["mean_reward"]["mean"] - worst["mean_reward"]["mean"]
            lines.append(
                f"- **{_algorithm_name(algorithm)}:** the best setting was "
                f"`{_parameters_text(best['parameters'])}` and the worst was "
                f"`{_parameters_text(worst['parameters'])}`. The observed reward spread was "
                f"{spread:.5f}."
            )

    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "![Reward versus training time](reward_vs_training_time.png)",
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
            "- Sensitivity chart: `hyperparameter_sensitivity.png`",
            "- Efficiency chart: `reward_vs_training_time.png`",
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
    sensitivity_path = output_dir / "hyperparameter_sensitivity.png"
    efficiency_path = output_dir / "reward_vs_training_time.png"
    csv_path = output_dir / "configuration_results.csv"
    report_path = output_dir / "analysis.md"

    plot_configuration_performance(configurations, performance_path)
    plot_hyperparameter_sensitivity(configurations, sensitivity_path)
    plot_reward_vs_training_time(configurations, efficiency_path)
    write_configuration_csv(configurations, csv_path)
    report_path.write_text(
        build_report(summary, summary_path), encoding="utf-8"
    )
    return [report_path, performance_path, sensitivity_path, efficiency_path, csv_path]


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
