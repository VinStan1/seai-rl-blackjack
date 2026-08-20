# Analysis: blackjack_final_million

## Executive summary

The highest observed final reward came from **Monte Carlo** with `epsilon=0.2, gamma=1.0`. Its mean reward was **-0.04491** (approximate 95% CI -0.04586 to -0.04395), with a 43.44% win rate.

The sweep status is `completed`: 15 of 15 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---|---:|---:|---:|
| Monte Carlo | `epsilon=0.2, gamma=1.0` | -0.04491 [-0.04586, -0.04395] | 43.44% | 90.51 s |
| Q-learning | `epsilon=0.1, alpha=0.05, gamma=1.0` | -0.06000 [-0.06083, -0.05917] | 42.59% | 75.84 s |
| SARSA | `epsilon=0.2, alpha=0.05, gamma=1.0` | -0.06115 [-0.06663, -0.05566] | 42.69% | 77.31 s |

## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Monte Carlo - Q-learning | 5 | 0.01509 [0.01460, 0.01559] | interval excludes zero |
| Monte Carlo - SARSA | 5 | 0.01624 [0.01143, 0.02105] | interval excludes zero |
| Q-learning - SARSA | 5 | 0.00115 [-0.00378, 0.00607] | difference is inconclusive at this precision |

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Hyperparameter sensitivity

![Hyperparameter sensitivity](hyperparameter_sensitivity.png)

- **Monte Carlo:** the best setting was `epsilon=0.2, gamma=1.0` and the worst was `epsilon=0.2, gamma=1.0`. The observed reward spread was 0.00000.
- **Q-learning:** the best setting was `epsilon=0.1, alpha=0.05, gamma=1.0` and the worst was `epsilon=0.1, alpha=0.05, gamma=1.0`. The observed reward spread was 0.00000.
- **SARSA:** the best setting was `epsilon=0.2, alpha=0.05, gamma=1.0` and the worst was `epsilon=0.2, alpha=0.05, gamma=1.0`. The observed reward spread was 0.00000.

## Efficiency

![Reward versus training time](reward_vs_training_time.png)

Training times were collected while independent runs could execute in parallel. They are useful operational measurements, but CPU contention means they should not be treated as clean single-process algorithm benchmarks.

## Interpretation limits and next steps

- This experiment uses only 5 training seeds per configuration. Use at least 10-20 fresh seeds for a stronger final comparison.
- Hyperparameters were selected in a separate pilot sweep, reducing selection bias in this final evaluation.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- This summary records only final performance. Add training checkpoints to compare learning speed and sample efficiency directly.
- Evaluate environment variants before making claims about generalisation.

## Generated artifacts

- Source summary: `results/final/blackjack_final_million_20260816T184122Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sensitivity chart: `hyperparameter_sensitivity.png`
- Efficiency chart: `reward_vs_training_time.png`
