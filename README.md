# Reinforcement Learning for Blackjack

Docker-first implementation and evaluation of reinforcement-learning agents on
Gymnasium's `Blackjack-v1`. The current milestone contains a tabular first-visit
Monte Carlo control agent. Finite-deck Blackjack, SARSA, and Q-learning are
intentionally left for later milestones.

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
|   |-- environments/blackjack.py
|   |-- train.py
|   `-- evaluate.py
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

Generated models and JSON reports persist in `results/`. The reports include
per-seed reward, win/draw/loss rates, training time, inference latency, and a 95%
normal-approximation confidence interval across independent training seeds.

## Reproducibility and experimental scope

The default experiment uses five training seeds. Training and evaluation seeds
are stored in the generated artifacts, together with the environment and
hyperparameters. Evaluation uses a deterministic greedy policy and a separate
seed range from training.

This bootstrap does **not yet satisfy the final course requirement to compare at
least two RL techniques**. A later milestone must add another algorithm, apply
the same multi-seed protocol, perform an appropriate statistical test, evaluate
generalisation variants, and discuss failure modes and sample efficiency.

## References

- Gymnasium, [Blackjack documentation](https://gymnasium.farama.org/environments/toy_text/blackjack/).
- Sutton, R. S. and Barto, A. G., *Reinforcement Learning: An Introduction*,
	second edition, Example 5.1.