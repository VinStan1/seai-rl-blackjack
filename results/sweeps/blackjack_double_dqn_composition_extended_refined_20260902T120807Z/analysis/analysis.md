# Analysis: blackjack_double_dqn_composition_extended_refined

Environment variant: **finite_composition**.

## Executive summary

The highest observed final reward came from **Double DQN** with `epsilon=1.0->0.01 linear (100%), gamma=1.0, batch_size=64, decks=6, gradient_clip=10.0, hidden_size=128, learning_rate=0.001, learning_starts=1000, replay_capacity=100000, target_update_interval=1000, train_frequency=4`. Its mean reward was **-0.04929** (approximate 95% CI -0.05234 to -0.04624), with a 42.83% win rate.

The sweep status is `completed`: 320 of 320 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

Compact labels `P01`, `P02`, and so on identify unique parameter settings within each algorithm panel. Their complete mapping is in the `plot_label` column of `configuration_results.csv`.

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Double DQN | 1,000,000 | `epsilon=1.0->0.01 linear (100%), gamma=1.0, batch_size=64, decks=6, gradient_clip=10.0, hidden_size=128, learning_rate=0.001, learning_starts=1000, replay_capacity=100000, target_update_interval=1000, train_frequency=4` | -0.04929 [-0.05234, -0.04624] | 42.83% | 546.00 s |

## Literature baseline

The **stick-on-17** policy hits below 17 and sticks on 17 or above. On the same 100,000 seeded evaluation episodes, its mean reward was **-0.07430** with a 41.22% win rate.

Reference: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, Example 5.1: Blackjack (2018), http://incompleteideas.net/book/RLbook2020.pdf.


## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Sample efficiency

![Sample efficiency](sample_efficiency.png)

Each point is a separately trained agent at that episode budget. Higher reward with fewer episodes indicates better sample efficiency.

## Projected Double DQN policy

![Hit frequency by visible state](best_policy_heatmap.png)

![Projection coverage](best_policy_coverage_heatmap.png)

![Double DQN policy by true-count band](best_policy_true_count_double_dqn.png)

For each tabular agent, every learned exact-composition Q-table state contributes equally. A Double DQN has no enumerable state table, so its projection instead uses the states visited during up to 10,000 greedy replay episodes. The first heatmap reports the fraction of contributing states or decisions that hit, so values near 0.5 expose visible states whose action changes with exact shoe composition. The coverage heatmap reports log10(1 + contributing states or decisions), distinguishing broad evidence from rare states. The count-conditioned panels repeat the hit-frequency view for negative, neutral, and positive Hi-Lo true counts. Gray means that nothing contributed to that cell. These are compressed projections, not evidence that the policy ignores exact composition. Tabular coverage measures learned-state support, whereas Double DQN coverage measures greedy-replay visitation, so their absolute coverage values should not be compared directly.


## Efficiency

### 100,000 training episodes

![Performance versus training time at 100,000 episodes](performance_vs_training_time_100000.png)

### 200,000 training episodes

![Performance versus training time at 200,000 episodes](performance_vs_training_time_200000.png)

### 500,000 training episodes

![Performance versus training time at 500,000 episodes](performance_vs_training_time_500000.png)

### 1,000,000 training episodes

![Performance versus training time at 1,000,000 episodes](performance_vs_training_time_1000000.png)

Each chart holds the training budget fixed. Every point is one hyperparameter configuration, horizontal intervals show uncertainty in mean training time, and vertical intervals show uncertainty in mean evaluation reward. The preferred region is the upper-left; black outlines identify the best-reward configuration for each algorithm at that budget.

![Training time](training_time.png)

Training times were collected while independent runs could execute in parallel. They are useful operational measurements, but CPU contention means they should not be treated as clean single-process algorithm benchmarks.

## Interpretation limits and next steps

- This experiment uses only 5 training seeds per configuration. Use at least 10-20 fresh seeds for a stronger final comparison.
- The best settings were selected using the same evaluation results shown here. A separate final seed set reduces selection bias.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.
- Compare this result with independently tuned standard and finite variants before attributing differences to the observation alone.

## Generated artifacts

- Source summary: `results/sweeps/blackjack_double_dqn_composition_extended_refined_20260902T120807Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget
- Policy heatmaps: `best_policy_heatmap.png`, `best_policy_coverage_heatmap.png`, `best_policy_true_count_double_dqn.png`
