"""First-visit Monte Carlo control with an epsilon-greedy policy."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeAlias

BlackjackState: TypeAlias = tuple[int, int, bool]
EpisodeStep: TypeAlias = tuple[BlackjackState, int, float]


class BlackjackEnvironment(Protocol):
    """Subset of the Gymnasium API required by the agent."""

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[BlackjackState, dict[str, Any]]: ...

    def step(
        self, action: int
    ) -> tuple[BlackjackState, float, bool, bool, dict[str, Any]]: ...


class MonteCarloAgent:
    """Tabular first-visit Monte Carlo control agent."""

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        number_actions: int = 2,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if number_actions < 1:
            raise ValueError("number_actions must be positive")

        self.epsilon = epsilon
        self.gamma = gamma
        self.number_actions = number_actions
        self.seed = seed
        self._random = random.Random(seed)
        self.q_values: defaultdict[BlackjackState, list[float]] = defaultdict(
            self._empty_action_values
        )
        self.visit_counts: defaultdict[BlackjackState, list[int]] = defaultdict(
            self._empty_visit_counts
        )

    def _empty_action_values(self) -> list[float]:
        return [0.0] * self.number_actions

    def _empty_visit_counts(self) -> list[int]:
        return [0] * self.number_actions

    def select_action(self, state: BlackjackState, *, explore: bool = True) -> int:
        """Select an epsilon-greedy action, using stable tie-breaking for evaluation."""
        if explore and self._random.random() < self.epsilon:
            return self._random.randrange(self.number_actions)

        action_values = self.q_values[state]
        best_value = max(action_values)
        best_actions = [
            action
            for action, value in enumerate(action_values)
            if value == best_value
        ]
        if explore:
            return self._random.choice(best_actions)
        return best_actions[0]

    def generate_episode(
        self,
        environment: BlackjackEnvironment,
        *,
        seed: int | None = None,
        explore: bool = True,
    ) -> list[EpisodeStep]:
        """Generate one complete episode using the current policy."""
        state, _ = environment.reset(seed=seed)
        episode: list[EpisodeStep] = []

        while True:
            action = self.select_action(state, explore=explore)
            next_state, reward, terminated, truncated, _ = environment.step(action)
            episode.append((state, action, float(reward)))
            if terminated or truncated:
                return episode
            state = next_state

    def update_episode(self, episode: Sequence[EpisodeStep]) -> None:
        """Update action values from the first occurrence of each state-action pair."""
        returns = [0.0] * len(episode)
        discounted_return = 0.0
        for index in range(len(episode) - 1, -1, -1):
            discounted_return = episode[index][2] + self.gamma * discounted_return
            returns[index] = discounted_return

        visited: set[tuple[BlackjackState, int]] = set()
        for (state, action, _), observed_return in zip(episode, returns, strict=True):
            state_action = (state, action)
            if state_action in visited:
                continue
            visited.add(state_action)

            self.visit_counts[state][action] += 1
            count = self.visit_counts[state][action]
            current_value = self.q_values[state][action]
            self.q_values[state][action] += (observed_return - current_value) / count

    def train(
        self,
        environment: BlackjackEnvironment,
        episodes: int,
    ) -> list[float]:
        """Train for a fixed number of episodes and return each episode reward."""
        if episodes < 1:
            raise ValueError("episodes must be positive")

        rewards: list[float] = []
        for episode_index in range(episodes):
            episode = self.generate_episode(
                environment,
                seed=self.seed + episode_index,
                explore=True,
            )
            self.update_episode(episode)
            rewards.append(sum(step[2] for step in episode))
        return rewards

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Save the learned action values and experiment metadata as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = [
            {
                "state": list(state),
                "values": self.q_values[state],
                "visits": self.visit_counts[state],
            }
            for state in sorted(self.q_values)
        ]
        payload = {
            "algorithm": "first_visit_monte_carlo_control",
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "number_actions": self.number_actions,
            "seed": self.seed,
            "metadata": dict(metadata or {}),
            "q_table": table,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> MonteCarloAgent:
        """Restore an agent from a JSON artifact produced by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            epsilon=float(payload["epsilon"]),
            gamma=float(payload["gamma"]),
            number_actions=int(payload["number_actions"]),
            seed=int(payload["seed"]),
        )
        for entry in payload["q_table"]:
            raw_state = entry["state"]
            state = (int(raw_state[0]), int(raw_state[1]), bool(raw_state[2]))
            agent.q_values[state] = [float(value) for value in entry["values"]]
            agent.visit_counts[state] = [int(value) for value in entry["visits"]]
        return agent