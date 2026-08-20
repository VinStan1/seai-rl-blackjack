# Analysis: blackjack_selected_million_comparison

## Executive summary

The highest observed final reward came from **Q-learning** with `epsilon=0.25, alpha=0.005, gamma=1.0`. Its mean reward was **-0.04579** (approximate 95% CI -0.04656 to -0.04503), with a 43.20% win rate.

The sweep status is `completed`: 30 of 30 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Monte Carlo | 1,000,000 | `epsilon=0.25, gamma=1.0` | -0.04871 [-0.04955, -0.04786] | 43.29% | 38.29 s |
| Q-learning | 1,000,000 | `epsilon=0.25, alpha=0.005, gamma=1.0` | -0.04579 [-0.04656, -0.04503] | 43.20% | 38.76 s |
| SARSA | 1,000,000 | `epsilon=1.0->0.05 linear (80%), alpha=0.005, gamma=1.0` | -0.04587 [-0.04691, -0.04482] | 43.22% | 41.27 s |

## Literature baseline

The **stick-on-17** policy hits below 17 and sticks on 17 or above. On the same 100,000 seeded evaluation episodes, its mean reward was **-0.07926** with a 40.92% win rate.

Reference: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, Example 5.1: Blackjack (2018), http://incompleteideas.net/book/RLbook2020.pdf.


## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Monte Carlo - Q-learning | 10 | -0.00291 [-0.00415, -0.00167] | interval excludes zero |
| Monte Carlo - SARSA | 10 | -0.00284 [-0.00423, -0.00145] | interval excludes zero |
| Q-learning - SARSA | 10 | 0.00007 [-0.00137, 0.00152] | difference is inconclusive at this precision |

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Final configuration scope

This stage intentionally evaluates only one configuration per algorithm. Hyperparameter sensitivity should be interpreted from the pilot grid search, not re-estimated from these final runs.


## Efficiency

### 1,000,000 training episodes

![Performance versus training time at 1,000,000 episodes](performance_vs_training_time_1000000.png)

Each chart holds the training budget fixed. Every point is one hyperparameter configuration, horizontal intervals show uncertainty in mean training time, and vertical intervals show uncertainty in mean evaluation reward. The preferred region is the upper-left; black outlines identify the best-reward configuration for each algorithm at that budget.

![Training time](training_time.png)

Training times were collected while independent runs could execute in parallel. They are useful operational measurements, but CPU contention means they should not be treated as clean single-process algorithm benchmarks.

## Interpretation limits and next steps

- Results use 10 independent training seeds per configuration; retain the per-seed results when applying the final paired statistical test.
- Hyperparameters were selected in a separate pilot sweep, reducing selection bias in this final evaluation.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.
- Evaluate environment variants before making claims about generalisation.

## Generated artifacts

- Source summary: `results/final/blackjack_selected_million_comparison_20260820T110016Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget
