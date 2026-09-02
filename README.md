# Reinforcement Learning for Blackjack

Docker-first implementation and evaluation of reinforcement-learning agents on
Gymnasium's `Blackjack-v1` and three finite-shoe variants. The project contains
tabular first-visit Monte Carlo, SARSA, and Q-learning agents, an optional
composition-aware Double DQN, and a configurable hyperparameter sweep runner.

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

## Finite six-deck variants

All finite variants use the same Blackjack rules and one shared shoe engine. A
shoe contains six standard decks (312 cards): 24 cards for each value from ace
through 9 and 96 ten-valued cards. The cut card is placed at 75% penetration,
or 234 dealt cards. Crossing the cut card never interrupts a hand; the shoe is
reshuffled immediately before the next hand, with 78 cards nominally remaining.
An experiment seed initializes the shuffle once and the deterministic random
stream then advances across hands.

The initial deal and dealer behavior match the Sutton and Barto mode of
`Blackjack-v1`: the player receives exactly two initial cards, the dealer sticks
on 17, and a player natural beats a non-natural dealer 21. The
available finite observations are:

- `finite_hidden`: `(player_sum, dealer_upcard, usable_ace)`. The physical shoe
  is finite but its composition is hidden, so the observation is not Markov and
  the learning problem is a POMDP from the agent's perspective.
- `finite_hi_lo`: the hidden observation plus an integer Hi-Lo true-count
  bucket. Cards 2-6 contribute `+1`, 7-9 contribute `0`, and tens and aces
  contribute `-1`. The running count is divided by physical decks remaining,
  truncated toward zero, and clipped to `[-20, 20]`.
- `finite_composition`: the hidden observation plus ten counts ordered as
  `(A, 2, 3, 4, 5, 6, 7, 8, 9, 10)`. They describe cards not yet publicly
  observed. The dealer hole card is not removed from the observation until it
  is revealed, preventing privileged information leakage.

The composition representation is deliberately large for a tabular method.
Sparse state visitation and slower learning are expected experimental outcomes,
not reasons to collapse or hash the state differently.

## Experimental design

Mean greedy-policy evaluation reward is the primary metric. Win rate and
training time are secondary diagnostics. The intended from-scratch protocol is:

1. Run the broad and then refined tabular grids on standard Gymnasium Blackjack.
2. Select one configuration per tabular algorithm at the largest refined budget,
   then validate longer training with fresh seeds.
3. Retrain those base-selected hyperparameters in `finite_hidden` without
   retuning. This is a transfer/robustness control for hidden finite-shoe
   dynamics, not an environment-specific optimization.
4. Run the refined tabular grid and fresh-seed long validation in `finite_hi_lo`.
5. Repeat the refined tabular process in `finite_composition`, then compare it
   with a separately tuned Double DQN at matched training/evaluation budgets.

The comparisons have different interpretations. Standard versus finite-hidden
measures the effect of a persistent finite shoe when its composition is omitted
from the observation. Finite-hidden versus Hi-Lo measures the net effect of
adding a compressed count, including both useful information and the cost of a
larger state representation. Hi-Lo versus exact composition similarly changes
both information detail and representation size; it does not isolate those two
effects individually.

Grid-search seeds are selection data. Final comparisons must use fresh training
seeds, while every algorithm in a given final comparison shares the same seed
list and evaluation protocol. Mean-reward differences should be calculated
seed-by-seed; overlapping marginal confidence intervals alone are not an
equivalence test.

## Optional Double DQN

`src/agents/double_dqn.py` is an isolated neural alternative for the exact
`finite_composition` observation. It preserves all ten counts but normalizes
player sum, dealer upcard, and each count before passing the 13 features through
a `13 -> 64 -> 64 -> 2` multilayer perceptron. Training uses experience replay,
a target network, Huber loss, gradient clipping, and Double-DQN targets: the
online network selects the next action and the target network evaluates it.

PyTorch is pinned separately in `requirements-dqn.txt`. Neural checkpoints use
`.pt`; tabular models retain their JSON format. The sweep imports Double DQN only
when an algorithm entry names `double_dqn`, and rejects that agent unless the
environment is `finite_composition`.

Run the short end-to-end check first:

```bash
docker compose run --rm --build sweep \
  --config experiments/double_dqn_composition_smoke.json
```

If the smoke result is structurally valid, run the controlled pilot:

```bash
docker compose run --rm --build sweep \
  --config experiments/double_dqn_composition_pilot.json
```

The pilot evaluates two training budgets and two learning rates on three seeds,
for 12 runs rather than reusing the much larger tabular grid. Model selection and
analysis still use mean evaluation reward exactly as for the other agents.

After using the pilot to verify learning, run the extended neural analysis:

```bash
docker compose run --rm --build sweep \
  --config experiments/double_dqn_composition_refined.json --workers 2
```

This evaluates 100,000, 200,000, 500,000, and 1,000,000 training episodes. Its
curated grid ranges final epsilon (`0.01`, `0.05`, `0.10`), exploration-decay
fraction (`0.50`, `0.80`, `1.00`), learning rate (`0.001`, `0.0003`), hidden
width (64, 128, 256), and batch size (64, 128) without taking the prohibitively
large full Cartesian product. Sixteen parameter settings over four budgets and
five seeds produce 320 runs. It is intentionally separate from the tabular grid
because neural and tabular agents do not share the same meaningful parameters.
Finite-composition analysis now projects the best representative model in three
complementary ways: a hit-frequency heatmap, a coverage heatmap, and separate
hit-frequency policies for negative, neutral, and positive Hi-Lo true-count
bands. Tabular plots aggregate all learned exact Q-table states; Double DQN plots
use states encountered during greedy replay because a network has no finite table
to enumerate. These compressed summaries must not be interpreted as showing that
exact composition is irrelevant.

The extension can be removed without changing the environments or tabular
agents: remove `double_dqn.py`, `requirements-dqn.txt`, its Docker install layer,
the `double_dqn` registry/analysis entries, its two experiment JSON files, and
`test_double_dqn.py`.

## Project layout

```text
.
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- requirements-dqn.txt
|-- src/
|   |-- agents/double_dqn.py
|   |-- agents/monte_carlo.py
|   |-- agents/temporal_difference.py
|   |-- environments/blackjack.py
|   |-- environments/finite_blackjack.py
|   |-- environments/factory.py
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

The supplied broad grid contains 180 configurations after combining four episode
budgets with five epsilon values and, for the TD agents, four alpha values. With
ten seeds this produces 1,800 independent runs. It is deliberately broader than
the refined grid and uses 100,000 evaluation episodes per trained model.

After reviewing the coarse results, run the separate refined grid:

```bash
docker compose run --rm --build sweep \
  --config experiments/hyperparameter_refined_sweep.json --workers 4
```

The refined grid tests fixed epsilon values around `0.20` to `0.35`, plus a
linear schedule that decays from `1.0` to `0.05` over the first 80% of training.
SARSA and Q-learning test alpha values `0.005` and `0.01`. Every setting is
trained at 100,000, 200,000, and 500,000 episodes using seeds `10` through `19`.
This produces 60 configurations and 600 runs.
Its timestamped summary is stored beside the coarse sweep under
`results/sweeps/`, so the two experiment histories remain separate.

Run the same refined tabular grid in the finite variants:

```bash
docker compose run --rm --build sweep \
  --config experiments/finite_hidden_refined_sweep.json --workers 4
docker compose run --rm --build sweep \
  --config experiments/finite_hi_lo_sweep.json --workers 4
docker compose run --rm --build sweep \
  --config experiments/finite_composition_sweep.json --workers 4
```

These compact files inherit the complete refined grid through `extends` and
override only experiment metadata and environment settings. The generated
`config.json` is fully resolved, so results do not depend on the parent file
after execution. Composition runs can consume substantially more memory and
storage because their Q-tables contain many more distinct states.

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

The analyzer creates `analysis.md`, a ranked configuration CSV, a
point-and-confidence-interval performance comparison, a sample-efficiency plot,
a training-time scaling plot, and one reward-versus-training-time scatter plot
per episode budget. Each scatter plot holds training experience fixed; its
preferred region is upper-left (higher reward, less time). Win rate remains in
the report and CSV as a secondary diagnostic, but is not given a separate plot
because mean reward is the primary objective. With `latest`, the most recently
modified sweep summary is selected automatically.

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

To run the base-selected tabular configurations as the finite-hidden transfer
control, point the same command at the base refined summary and override only the
validation environment. Use the base grid's largest budget for a matched
comparison, fresh seeds, and a distinct experiment name:

```bash
docker compose run --rm final \
  --summary results/sweeps/<base-refined-run>/summary.json \
  --episodes 500000 --evaluation-episodes 100000 \
  --evaluation-seed 20000000 \
  --seeds 200 201 202 203 204 205 206 207 208 209 \
  --environment-variant finite_hidden --decks 6 --penetration 0.75 \
  --experiment-name blackjack_finite_hidden_transfer --workers 4
```

This selects hyperparameters from the base summary and retrains them in the
finite-hidden environment. Do not select new hyperparameters from the hidden
results if the intended claim is transfer robustness rather than hidden-specific
tuning.

Generated models and JSON reports persist in `results/`. The reports include
per-seed reward, win/draw/loss rates, episode and action counts, training time,
inference latency, Q-table size, evaluation actions on unseen states, and a 95%
normal-approximation confidence interval across independent training seeds.
The unseen-state rate is especially important for Hi-Lo and composition states:
a greedy tabular policy uses its deterministic tie-break action (`stick`) when
an evaluation state has no learned values. Environment wall-clock times are not directly
comparable across variants: Gymnasium's standard wrapper has substantially more
per-call overhead than the purpose-built finite-shoe engine.

## Reproducibility and experimental scope

Training and evaluation seeds are stored in the generated artifacts, together
with the environment and hyperparameters. Evaluation uses a deterministic greedy
policy and a separate seed from training. Each training or evaluation run seeds
its environment once and then advances an independent reproducible random stream;
it does not reseed every episode. This preserves finite-shoe continuity while
remaining deterministic.

Each new sweep also evaluates a deterministic **stick-on-17 baseline**: hit when
the player sum is below 17 and stick on 17 or above. It uses the same evaluation
episode count and initial evaluation seed as every learned policy, and its score
is stored in `summary.json` and drawn as a reference line in the reward plots.
This dealer-like policy is the policy described in Sutton
and Barto, *Reinforcement Learning: An Introduction*, second edition, Example
5.1: Blackjack (2018).

The sweep supplies the multi-algorithm, multi-seed experiment data, but a final
course submission must still apply an appropriate statistical test, evaluate
generalisation variants, and discuss failure modes and sample efficiency.

## References

- Gymnasium, [Blackjack documentation](https://gymnasium.farama.org/environments/toy_text/blackjack/).
- Sutton, R. S. and Barto, A. G., *Reinforcement Learning: An Introduction*,
  second edition, Example 5.1, 2018. [Online edition](http://incompleteideas.net/book/RLbook2020.pdf).
