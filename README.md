# Reinforcement Learning for Blackjack

Docker-first implementation and evaluation of reinforcement-learning agents on
Gymnasium's `Blackjack-v1`. The project contains tabular first-visit Monte Carlo,
SARSA, and Q-learning agents, plus a configurable hyperparameter sweep runner.

## MDP model

The environment follows the Sutton and Barto rules (`sab=True`):

- **State:** `(player_sum, dealer_showing_card, usable_ace)`, a fully discrete
	observation. No normalization or frame stacking is required.
- **Actions:** `0` means stick and `1` means hit.
- **Reward:** `+1` for a win, `0` for a draw, and `-1` for a loss. Rewards are
	terminal and sparse; no custom shaping is applied, which prevents reward
	hacking through an altered objective.
- **Transitions:** cards are sampled with replacement from an infinite deck.
	The process is stochastic, while explicit seeds make experiments reproducible.
- **Episode end:** the player sticks or goes bust. Gymnasium's dealer policy
	then resolves the hand.

Monte Carlo control fits this episodic task because it estimates action values
from complete sampled returns without requiring a transition model. The policy
is epsilon-greedy during training and greedy during evaluation. Action values use
the incremental sample mean of first-visit returns with discount factor
`gamma=1.0` by default.

## Project layout

```text
.
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- src/
|   |-- agents/monte_carlo.py
|   |-- agents/temporal_difference.py
|   |-- environments/blackjack.py
|   |-- train.py
|   |-- evaluate.py
|   `-- sweep.py
|-- experiments/
|-- results/
`-- tests/
```

## Run with Docker

No local Python environment is required. Build and run the complete test suite:

```bash
docker compose build
docker compose run --rm test
```

Train five independent agents with the default 100,000 episodes per seed:

```bash
docker compose run --rm train
```

Pass custom CLI options after the service name when needed:

```bash
docker compose run --rm train python -m src.train \
	--episodes 250000 --epsilon 0.1 --seeds 11 22 33 44 55
```

Evaluate every saved model over 10,000 independent episodes:

```bash
docker compose run --rm evaluate
```

Run the supplied Monte Carlo, SARSA, and Q-learning hyperparameter grid:

```bash
docker compose run --rm sweep
```

Pass another configuration and optionally train independent runs in parallel:

```bash
docker compose run --rm sweep \
  --config experiments/hyperparameter_sweep.json --workers 4
```

The JSON configuration defines the environment, training and evaluation budgets,
seeds, algorithms, and parameter value lists. A scalar means one value; a list is
expanded as part of the Cartesian product. The supplied pilot grid trains every
agent setting at 20,000, 50,000, 100,000, and 200,000 episodes. Monte Carlo
accepts `epsilon` and `gamma`; SARSA and Q-learning also accept `alpha`.

The supplied grid contains 84 configurations: 12 Monte Carlo, 36 SARSA, and 36
Q-learning settings after including the four episode budgets. With five seeds,
this produces 420 independent runs.

After reviewing the coarse results, run the separate refined grid:

```bash
docker compose run --rm --build sweep \
  --config experiments/hyperparameter_refined_sweep.json --workers 4
```

The refined grid tests fixed epsilon values `0.20`, `0.25`, and `0.30`, plus a
linear schedule that decays from `1.0` to `0.05` over the first 80% of training.
SARSA and Q-learning test alpha values `0.005` and `0.01`. Every setting is
trained at 20,000, 50,000, 100,000, 200,000, and 500,000 episodes using the new
pilot seeds `10` through `14`. This produces 100 configurations and 500 runs.
Its timestamped summary is stored beside the coarse sweep under
`results/sweeps/`, so the two experiment histories remain separate.

Every invocation creates a timestamped directory under `results/sweeps/`. Each
seed/configuration run saves a model and run report. `summary.json` is refreshed
after every completed or failed run and ranks configurations by mean evaluation
reward across seeds. Worker processes only parallelize independent runs; they do
not alter their seeds or experimental settings.

While the sweep is running, its aggregate progress bar shows completed runs,
percentage, elapsed time, and an ETA. The ETA becomes available after the first
configuration/seed run finishes and adjusts as later runs complete.

Generate a human-readable analysis for the newest sweep:

```bash
docker compose run --rm analyze
```

Analyze a specific summary or choose another output directory:

```bash
docker compose run --rm analyze \
  --summary results/sweeps/blackjack_pilot_grid_20260816T181220Z/summary.json \
  --output-dir results/my_analysis
```

The analyzer creates `analysis.md`, a ranked configuration CSV, and three PNG
figures: a point-and-confidence-interval comparison of every configuration, a
sample-efficiency plot of reward against training episodes, and a training-time
scaling plot. Win rate remains in the report and CSV as a secondary diagnostic,
but is not given a separate plot because mean reward is the primary objective.
With `latest`, the most recently modified sweep summary is selected automatically.

## Final million-episode comparison

After the pilot grid search, run the best configuration from each algorithm on
fresh seeds:

```bash
docker compose run --rm final \
  --summary latest --workers 4
```

By default this command:

- selects the highest-mean-reward Monte Carlo, SARSA, and Q-learning settings
  at the largest pilot training budget;
- uses ten fresh training seeds (`100` through `109`);
- trains each algorithm for 1,000,000 episodes per seed;
- evaluates each trained model over 1,000,000 independent episodes;
- writes the normal sweep `summary.json`, models, and per-run reports;
- generates the Markdown/PNG/CSV analysis automatically; and
- writes `final_selection.json` with the observed winner and its paired
  comparison against the runner-up.

The full default run represents 30 million training episodes and 30 million
evaluation episodes. Reduce the evaluation budget or change the fresh seeds when
doing a quick check:

```bash
docker compose run --rm final \
  --summary latest --workers 4 \
  --evaluation-episodes 100000 \
  --seeds 100 101 102 103 104
```

The selected model is reported as provisional when its paired 95% confidence
interval versus the runner-up includes zero.

Generated models and JSON reports persist in `results/`. The reports include
per-seed reward, win/draw/loss rates, training time, inference latency, and a 95%
normal-approximation confidence interval across independent training seeds.

## Reproducibility and experimental scope

The default experiment uses five training seeds. Training and evaluation seeds
are stored in the generated artifacts, together with the environment and
hyperparameters. Evaluation uses a deterministic greedy policy and a separate
seed range from training. Each training run seeds its environment once and then
advances an independent reproducible random stream; it does not reseed every
episode with overlapping seed ranges.

The sweep supplies the multi-algorithm, multi-seed experiment data, but a final
course submission must still apply an appropriate statistical test, evaluate
generalisation variants, and discuss failure modes and sample efficiency.

## References

- Gymnasium, [Blackjack documentation](https://gymnasium.farama.org/environments/toy_text/blackjack/).
- Sutton, R. S. and Barto, A. G., *Reinforcement Learning: An Introduction*,
	second edition, Example 5.1.
