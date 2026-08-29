# Analysis: blackjack_double_dqn_composition_pilot

Environment variant: **finite_composition**.

## Executive summary

The highest observed final reward came from **Double DQN** with `epsilon=1.0->0.05 linear (80%), gamma=1.0, batch_size=64, decks=6, gradient_clip=10.0, hidden_size=64, learning_rate=0.001, learning_starts=1000, replay_capacity=100000, target_update_interval=1000, train_frequency=4`. Its mean reward was **-0.05287** (approximate 95% CI -0.05424 to -0.05149), with a 42.70% win rate.

The sweep status is `completed`: 12 of 12 runs completed and 0 failed.

![Configuration performance](configuration_performance.png)

## Best configuration per algorithm

| Algorithm | Training episodes | Parameters | Mean reward (95% CI) | Win rate | Training time |
|---|---:|---|---:|---:|---:|
| Double DQN | 100,000 | `epsilon=1.0->0.05 linear (80%), gamma=1.0, batch_size=64, decks=6, gradient_clip=10.0, hidden_size=64, learning_rate=0.001, learning_starts=1000, replay_capacity=100000, target_update_interval=1000, train_frequency=4` | -0.05287 [-0.05424, -0.05149] | 42.70% | 40.95 s |

## Literature baseline

The **stick-on-17** policy hits below 17 and sticks on 17 or above. On the same 100,000 seeded evaluation episodes, its mean reward was **-0.08031** with a 40.96% win rate.

Reference: Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, Example 5.1: Blackjack (2018), http://incompleteideas.net/book/RLbook2020.pdf.


## Paired comparison of the selected configurations

Differences below are calculated seed by seed as the first algorithm minus the second. A positive value favors the first algorithm.

| Comparison | Seeds | Mean reward difference (95% CI) | Interpretation |
|---|---:|---:|---|

These normal-approximation intervals are descriptive; they are not a replacement for a pre-specified final statistical testing protocol.

## Sample efficiency

![Sample efficiency](sample_efficiency.png)

Each point is a separately trained agent at that episode budget. Higher reward with fewer episodes indicates better sample efficiency.


## Efficiency

### 50,000 training episodes

![Performance versus training time at 50,000 episodes](performance_vs_training_time_50000.png)

### 100,000 training episodes

![Performance versus training time at 100,000 episodes](performance_vs_training_time_100000.png)

Each chart holds the training budget fixed. Every point is one hyperparameter configuration, horizontal intervals show uncertainty in mean training time, and vertical intervals show uncertainty in mean evaluation reward. The preferred region is the upper-left; black outlines identify the best-reward configuration for each algorithm at that budget.

![Training time](training_time.png)

Training times were collected while independent runs could execute in parallel. They are useful operational measurements, but CPU contention means they should not be treated as clean single-process algorithm benchmarks.

## Interpretation limits and next steps

- This experiment uses only 3 training seeds per configuration. Use at least 10-20 fresh seeds for a stronger final comparison.
- The best settings were selected using the same evaluation results shown here. A separate final seed set reduces selection bias.
- Confidence-interval overlap alone does not prove algorithms are equivalent.
- Episode-budget points are trained independently from scratch; they estimate sample efficiency but are not checkpoints from one continuous run.
- Compare this result with independently tuned standard and finite variants before attributing differences to the observation alone.

## Generated artifacts

- Source summary: `results/sweeps/blackjack_double_dqn_composition_pilot_20260828T145624Z/summary.json`
- Full ranked table: `configuration_results.csv`
- Final reward chart: `configuration_performance.png`
- Sample-efficiency chart: `sample_efficiency.png`
- Training-time chart: `training_time.png`
- Performance-versus-training-time charts: one `performance_vs_training_time_<episodes>.png` file per training budget
