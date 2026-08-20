# Analysis: blackjack_selected_million_comparison

## Executive summary

The highest observed final reward came from **SARSA** with `epsilon=1.0->0.05 linear (80%), alpha=0.005, gamma=1.0`. Its mean reward was **-0.04505** (approximate 95% CI -0.04554 to -0.04456), with a 43.19% win rate.

The sweep status is `completed`: 90 of 90 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Monte Carlo | 1,000,000 | `epsilon=0.25, gamma=1.0` | -0.04871 [-0.04955, -0.04786] | 43.29% | 44.34 s |
| Q-learning | 2,000,000 | `epsilon=0.25, alpha=0.005, gamma=1.0` | -0.04553 [-0.04637, -0.04468] | 43.19% | 83.97 s |
| SARSA | 2,000,000 | `epsilon=1.0->0.05 linear (80%), alpha=0.005, gamma=1.0` | -0.04505 [-0.04554, -0.04456] | 43.19% | 78.63 s |

## Literature baseline

The **stick-on-17** policy hits below 17 and sticks on 17 or above. On the same 100,000 seeded evaluation episodes, its mean reward was **-0.07926** with a 40.92% win rate.

Reference: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, Example 5.1: Blackjack (2018), http://incompleteideas.net/book/RLbook2020.pdf.


## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Monte Carlo - Q-learning | 10 | -0.00318 [-0.00421, -0.00215] | interval excludes zero |
| Monte Carlo - SARSA | 10 | -0.00365 [-0.00463, -0.00267] | interval excludes zero |
| Q-learning - SARSA | 10 | -0.00047 [-0.00163, 0.00069] | difference is inconclusive at this precision |

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Final configuration scope

This stage intentionally evaluates only one configuration per algorithm. Hyperparameter sensitivity should be interpreted from the pilot grid search, not re-estimated from these final runs.


## Efficiency

### 500,000 training episodes

![Performance versus training time at 500,000 episodes](performance_vs_training_time_500000.png)

### 1,000,000 training episodes

![Performance versus training time at 1,000,000 episodes](performance_vs_training_time_1000000.png)

### 2,000,000 training episodes

![Performance versus training time at 2,000,000 episodes](performance_vs_training_time_2000000.png)

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

- Source summary: `results/final/blackjack_selected_million_comparison_20260820T112553Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget
