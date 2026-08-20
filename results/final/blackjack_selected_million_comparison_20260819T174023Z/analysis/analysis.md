# Analysis: blackjack_selected_million_comparison

## Executive summary

The highest observed final reward came from **Q-learning** with `epsilon=0.25, alpha=0.005, gamma=1.0`. Its mean reward was **-0.05022** (approximate 95% CI -0.05120 to -0.04924), with a 42.96% win rate.

The sweep status is `completed`: 90 of 90 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Monte Carlo | 1,000,000 | `epsilon=0.3, gamma=1.0` | -0.05243 [-0.05293, -0.05194] | 43.16% | 107.41 s |
| Q-learning | 1,000,000 | `epsilon=0.25, alpha=0.005, gamma=1.0` | -0.05022 [-0.05120, -0.04924] | 42.96% | 100.05 s |
| SARSA | 1,000,000 | `epsilon=0.25, alpha=0.005, gamma=1.0` | -0.05124 [-0.05240, -0.05009] | 43.08% | 106.94 s |

## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Monte Carlo - Q-learning | 10 | -0.00221 [-0.00354, -0.00089] | interval excludes zero |
| Monte Carlo - SARSA | 10 | -0.00119 [-0.00239, 0.00001] | difference is inconclusive at this precision |
| Q-learning - SARSA | 10 | 0.00102 [-0.00040, 0.00245] | difference is inconclusive at this precision |

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Final configuration scope

This stage intentionally evaluates only one configuration per algorithm. Hyperparameter sensitivity should be interpreted from the pilot grid search, not re-estimated from these final runs.


## Efficiency

![Training time](training_time.png)

Training times were collected while independent runs could execute in parallel. They are useful operational measurements, but CPU contention means they should not be treated as clean single-process algorithm benchmarks.

## Interpretation limits and next steps

- Results use 10 independent training seeds per configuration; retain the per-seed results when applying the final paired statistical test.
- Hyperparameters were selected in a separate pilot sweep, reducing selection bias in this final evaluation.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.
- Evaluate environment variants before making claims about generalisation.

## Generated artifacts

- Source summary: `results/final/blackjack_selected_million_comparison_20260819T174023Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
