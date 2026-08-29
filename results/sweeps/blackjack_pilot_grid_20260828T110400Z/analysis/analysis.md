# Analysis: blackjack_pilot_grid

Environment variant: **standard**.

## Executive summary

The highest observed final reward came from **Q-learning** with `epsilon=0.3, alpha=0.01, gamma=1.0`. Its mean reward was **-0.05190** (approximate 95% CI -0.05437 to -0.04944), with a 42.91% win rate.

The sweep status is `completed`: 1800 of 1800 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Monte Carlo | 200,000 | `epsilon=0.3, gamma=1.0` | -0.05444 [-0.05543, -0.05346] | 43.08% | 8.62 s |
| Q-learning | 200,000 | `epsilon=0.3, alpha=0.01, gamma=1.0` | -0.05190 [-0.05437, -0.04944] | 42.91% | 9.03 s |
| SARSA | 200,000 | `epsilon=0.2, alpha=0.01, gamma=1.0` | -0.05392 [-0.05554, -0.05230] | 42.96% | 8.28 s |

## Literature baseline

The **stick-on-17** policy hits below 17 and sticks on 17 or above. On the same 100,000 seeded evaluation episodes, its mean reward was **-0.07925** with a 40.98% win rate.

Reference: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, Example 5.1: Blackjack (2018), http://incompleteideas.net/book/RLbook2020.pdf.


## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Monte Carlo - Q-learning | 10 | -0.00254 [-0.00505, -0.00003] | interval excludes zero |
| Monte Carlo - SARSA | 10 | -0.00052 [-0.00281, 0.00177] | difference is inconclusive at this precision |
| Q-learning - SARSA | 10 | 0.00202 [-0.00119, 0.00522] | difference is inconclusive at this precision |

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Sample efficiency

![Sample efficiency](sample_efficiency.png)

Each point is a separately trained agent at that episode budget. Higher reward with fewer episodes indicates better sample efficiency.


## Efficiency

### 20,000 training episodes

![Performance versus training time at 20,000 episodes](performance_vs_training_time_20000.png)

### 50,000 training episodes

![Performance versus training time at 50,000 episodes](performance_vs_training_time_50000.png)

### 100,000 training episodes

![Performance versus training time at 100,000 episodes](performance_vs_training_time_100000.png)

### 200,000 training episodes

![Performance versus training time at 200,000 episodes](performance_vs_training_time_200000.png)

Each chart holds the training budget fixed. Every point is one hyperparameter configuration, horizontal intervals show uncertainty in mean training time, and vertical intervals show uncertainty in mean evaluation reward. The preferred region is the upper-left; black outlines identify the best-reward configuration for each algorithm at that budget.

![Training time](training_time.png)

Training times were collected while independent runs could execute in parallel. They are useful operational measurements, but CPU contention means they should not be treated as clean single-process algorithm benchmarks.

## Interpretation limits and next steps

- Results use 10 independent training seeds per configuration; retain the per-seed results when applying the final paired statistical test.
- The best settings were selected using the same evaluation results shown here. A separate final seed set reduces selection bias.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.
- Evaluate environment variants before making claims about generalisation.

## Generated artifacts

- Source summary: `results/sweeps/blackjack_pilot_grid_20260828T110400Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget
