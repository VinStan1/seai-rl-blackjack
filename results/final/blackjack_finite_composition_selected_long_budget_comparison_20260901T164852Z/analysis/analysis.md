# Analysis: blackjack_finite_composition_selected_long_budget_comparison

Environment variant: **finite_composition**.

## Executive summary

The highest observed final reward came from **SARSA** with `epsilon=0.2, alpha=0.01, gamma=1.0`. Its mean reward was **-0.17819** (approximate 95% CI -0.17866 to -0.17773), with a 38.65% win rate.

The sweep status is `completed`: 90 of 90 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Monte Carlo | 2,000,000 | `epsilon=0.3, gamma=1.0` | -0.17927 [-0.17987, -0.17867] | 38.61% | 20.90 s |
| Q-learning | 2,000,000 | `epsilon=0.35, alpha=0.005, gamma=1.0` | -0.17828 [-0.17860, -0.17796] | 38.64% | 14.62 s |
| SARSA | 2,000,000 | `epsilon=0.2, alpha=0.01, gamma=1.0` | -0.17819 [-0.17866, -0.17773] | 38.65% | 14.89 s |

## Literature baseline

The **stick-on-17** policy hits below 17 and sticks on 17 or above. On the same 100,000 seeded evaluation episodes, its mean reward was **-0.07544** with a 41.16% win rate.

Reference: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, Example 5.1: Blackjack (2018), http://incompleteideas.net/book/RLbook2020.pdf.


## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Monte Carlo - Q-learning | 10 | -0.00099 [-0.00169, -0.00029] | interval excludes zero |
| Monte Carlo - SARSA | 10 | -0.00108 [-0.00185, -0.00030] | interval excludes zero |
| Q-learning - SARSA | 10 | -0.00009 [-0.00061, 0.00043] | difference is inconclusive at this precision |

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Sample efficiency

![Sample efficiency](sample_efficiency.png)

Each point is a separately trained agent at that episode budget. Higher reward with fewer episodes indicates better sample efficiency.

## Projected finite-composition policies

![Hit frequency by visible state](best_policy_heatmap.png)

![Projection coverage](best_policy_coverage_heatmap.png)

![Monte Carlo policy by true-count band](best_policy_true_count_monte_carlo.png)

![SARSA policy by true-count band](best_policy_true_count_sarsa.png)

![Q-learning policy by true-count band](best_policy_true_count_q_learning.png)

For each tabular agent, every learned exact-composition Q-table state contributes equally. A Double DQN has no enumerable state table, so its projection instead uses the states visited during up to 10,000 greedy replay episodes. The first heatmap reports the fraction of contributing states or decisions that hit, so values near 0.5 expose visible states whose action changes with exact shoe composition. The coverage heatmap reports log10(1 + contributing states or decisions), distinguishing broad evidence from rare states. The count-conditioned panels repeat the hit-frequency view for negative, neutral, and positive Hi-Lo true counts. Gray means that nothing contributed to that cell. These are compressed projections, not evidence that the policy ignores exact composition. Tabular coverage measures learned-state support, whereas Double DQN coverage measures greedy-replay visitation, so their absolute coverage values should not be compared directly.


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
- The best settings were selected using the same evaluation results shown here. A separate final seed set reduces selection bias.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.
- Compare this result with independently tuned standard and finite variants before attributing differences to the observation alone.

## Generated artifacts

- Source summary: `results/final/blackjack_finite_composition_selected_long_budget_comparison_20260901T164852Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget
- Policy heatmaps: `best_policy_heatmap.png`, `best_policy_coverage_heatmap.png`, `best_policy_true_count_monte_carlo.png`, `best_policy_true_count_sarsa.png`, `best_policy_true_count_q_learning.png`
