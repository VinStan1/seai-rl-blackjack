# Analysis: blackjack_finite_hidden_refined_grid

Environment variant: **finite_hidden**.

## Executive summary

The highest observed final reward came from **Q-learning** with `epsilon=0.35, alpha=0.005, gamma=1.0`. Its mean reward was **-0.04320** (approximate 95% CI -0.04418 to -0.04222), with a 43.39% win rate.

The sweep status is `completed`: 600 of 600 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Monte Carlo | 500,000 | `epsilon=0.25, gamma=1.0` | -0.04780 [-0.04878, -0.04681] | 43.45% | 6.74 s |
| Q-learning | 500,000 | `epsilon=0.35, alpha=0.005, gamma=1.0` | -0.04320 [-0.04418, -0.04222] | 43.39% | 4.84 s |
| SARSA | 500,000 | `epsilon=1.0->0.05 linear (80%), alpha=0.01, gamma=1.0` | -0.04368 [-0.04584, -0.04152] | 43.39% | 5.26 s |

## Literature baseline

The **stick-on-17** policy hits below 17 and sticks on 17 or above. On the same 100,000 seeded evaluation episodes, its mean reward was **-0.07665** with a 41.16% win rate.

Reference: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, Example 5.1: Blackjack (2018), http://incompleteideas.net/book/RLbook2020.pdf.


## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Monte Carlo - Q-learning | 10 | -0.00460 [-0.00623, -0.00296] | interval excludes zero |
| Monte Carlo - SARSA | 10 | -0.00412 [-0.00643, -0.00180] | interval excludes zero |
| Q-learning - SARSA | 10 | 0.00048 [-0.00222, 0.00318] | difference is inconclusive at this precision |

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Sample efficiency

![Sample efficiency](sample_efficiency.png)

Each point is a separately trained agent at that episode budget. Higher reward with fewer episodes indicates better sample efficiency.

## Best policies by algorithm

![Best-per-algorithm action heatmaps](best_policy_heatmap.png)

Each row shows the greedy policy for the highest-reward configuration of one tabular algorithm. Its representative seed is the completed run closest to that configuration's mean reward; gray cells were not present in that model's learned Q-table.


## Efficiency

### 100,000 training episodes

![Performance versus training time at 100,000 episodes](performance_vs_training_time_100000.png)

### 200,000 training episodes

![Performance versus training time at 200,000 episodes](performance_vs_training_time_200000.png)

### 500,000 training episodes

![Performance versus training time at 500,000 episodes](performance_vs_training_time_500000.png)

Each chart holds the training budget fixed. Every point is one hyperparameter configuration, horizontal intervals show uncertainty in mean training time, and vertical intervals show uncertainty in mean evaluation reward. The preferred region is the upper-left; black outlines identify the best-reward configuration for each algorithm at that budget.

![Training time](training_time.png)

Training times were collected while independent runs could execute in parallel. They are useful operational measurements, but CPU contention means they should not be treated as clean single-process algorithm benchmarks.

## Interpretation limits and next steps

- Results use 10 independent training seeds per configuration; retain the per-seed results when applying the final paired statistical test.
- The best settings were selected using the same evaluation results shown here. A separate final seed set reduces selection bias.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.
- Compare this result with independently tuned standard and finite variants before attributing differences to the observation alone.

## Generated artifacts

- Source summary: `results/sweeps/blackjack_finite_hidden_refined_grid_20260831T173256Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget
- Best-per-algorithm policy heatmaps: `best_policy_heatmap.png`
