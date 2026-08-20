"""Tabular SARSA and Q-learning agents for Gymnasium Blackjack."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeAlias

BlackjackState: TypeAlias = tuple[int | bool, ...]
EpisodeStep: TypeAlias = tuple[BlackjackState, int, float]


def _deserialize_state(raw_state: Sequence[object]) -> BlackjackState:
    if len(raw_state) < 3:
        raise ValueError("serialized Blackjack states need at least three values")
    return (
        int(raw_state[0]),
        int(raw_state[1]),
        bool(raw_state[2]),
        *(int(value) for value in raw_state[3:]),
    )


class BlackjackEnvironment(Protocol):
    def reset(
        self, *, seed: int | None = None
    ) -> tuple[BlackjackState, dict[str, Any]]: ...

    def step(
        self, action: int
    ) -> tuple[BlackjackState, float, bool, bool, dict[str, Any]]: ...


class TabularTDAgent:
    """Shared behavior for one-step tabular temporal-difference agents."""

    algorithm = "tabular_td"

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay_fraction: float = 0.8,
        alpha: float = 0.05,
        gamma: float = 1.0,
        number_actions: int = 2,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if (epsilon_start is None) != (epsilon_end is None):
            raise ValueError("epsilon_start and epsilon_end must be provided together")
        if epsilon_start is not None and not 0.0 <= epsilon_start <= 1.0:
            raise ValueError("epsilon_start must be between 0 and 1")
        if epsilon_end is not None and not 0.0 <= epsilon_end <= 1.0:
            raise ValueError("epsilon_end must be between 0 and 1")
        if not 0.0 < epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction must be greater than 0 and at most 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be greater than 0 and at most 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if number_actions < 1:
            raise ValueError("number_actions must be positive")

        self.epsilon = epsilon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_fraction = epsilon_decay_fraction
        self._active_epsilon = epsilon_start if epsilon_start is not None else epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.number_actions = number_actions
        self.seed = seed
        self._random = random.Random(seed)
        self.q_values: defaultdict[BlackjackState, list[float]] = defaultdict(
            self._empty_action_values
        )

    def _empty_action_values(self) -> list[float]:
        return [0.0] * self.number_actions

    def epsilon_for_episode(self, episode_index: int, episodes: int) -> float:
        """Return fixed epsilon or a linearly decayed value for this episode."""
        if self.epsilon_start is None or self.epsilon_end is None:
            return self.epsilon
        decay_episodes = max(1, round((episodes - 1) * self.epsilon_decay_fraction))
        if episode_index >= decay_episodes:
            return self.epsilon_end
        progress = episode_index / decay_episodes
        return self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

    def select_action(self, state: BlackjackState, *, explore: bool = True) -> int:
        """Choose an epsilon-greedy action with stable evaluation tie-breaking."""
        if explore and self._random.random() < self._active_epsilon:
            return self._random.randrange(self.number_actions)

        # Looking up an unseen evaluation state must not mutate the learned table.
        values = self.q_values.get(state, self._empty_action_values())
        best_value = max(values)
        best_actions = [index for index, value in enumerate(values) if value == best_value]
        if explore:
            return self._random.choice(best_actions)
        return best_actions[0]

    def generate_episode(
        self,
        environment: BlackjackEnvironment,
        *,
        seed: int | None = None,
        explore: bool = False,
    ) -> list[EpisodeStep]:
        """Play one complete episode without updating the value table."""
        state, _ = environment.reset(seed=seed)
        episode: list[EpisodeStep] = []
        while True:
            action = self.select_action(state, explore=explore)
            next_state, reward, terminated, truncated, _ = environment.step(action)
            episode.append((state, action, float(reward)))
            if terminated or truncated:
                return episode
            state = next_state

    def _next_value(
        self,
        next_state: BlackjackState,
        next_action: int,
    ) -> float:
        raise NotImplementedError

    def train(self, environment: BlackjackEnvironment, episodes: int) -> list[float]:
        """Train online with one-step temporal-difference updates."""
        if episodes < 1:
            raise ValueError("episodes must be positive")

        rewards: list[float] = []
        for episode_index in range(episodes):
            self._active_epsilon = self.epsilon_for_episode(
                episode_index, episodes
            )
            state, _ = environment.reset(
                seed=self.seed if episode_index == 0 else None
            )
            action = self.select_action(state, explore=True)
            episode_reward = 0.0

            while True:
                next_state, reward, terminated, truncated, _ = environment.step(action)
                reward = float(reward)
                episode_reward += reward
                finished = terminated or truncated

                if finished:
                    target = reward
                    next_action = 0
                else:
                    next_action = self.select_action(next_state, explore=True)
                    target = reward + self.gamma * self._next_value(
                        next_state, next_action
                    )

                current = self.q_values[state][action]
                self.q_values[state][action] += self.alpha * (target - current)
                if finished:
                    break
                state, action = next_state, next_action

            rewards.append(episode_reward)
        return rewards

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Save the learned Q-table and experiment metadata as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_fraction": self.epsilon_decay_fraction,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "number_actions": self.number_actions,
            "seed": self.seed,
            "metadata": dict(metadata or {}),
            "q_table": [
                {"state": list(state), "values": self.q_values[state]}
                for state in sorted(self.q_values)
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TabularTDAgent:
        """Restore an agent saved by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            epsilon=float(payload["epsilon"]),
            epsilon_start=(
                float(payload["epsilon_start"])
                if payload.get("epsilon_start") is not None
                else None
            ),
            epsilon_end=(
                float(payload["epsilon_end"])
                if payload.get("epsilon_end") is not None
                else None
            ),
            epsilon_decay_fraction=float(
                payload.get("epsilon_decay_fraction", 0.8)
            ),
            alpha=float(payload["alpha"]),
            gamma=float(payload["gamma"]),
            number_actions=int(payload["number_actions"]),
            seed=int(payload["seed"]),
        )
        for entry in payload["q_table"]:
            raw_state = entry["state"]
            state = _deserialize_state(raw_state)
            agent.q_values[state] = [float(value) for value in entry["values"]]
        return agent


class SarsaAgent(TabularTDAgent):
    """On-policy one-step SARSA control."""

    algorithm = "sarsa"

    def _next_value(self, next_state: BlackjackState, next_action: int) -> float:
        return self.q_values[next_state][next_action]


class QLearningAgent(TabularTDAgent):
    """Off-policy one-step Q-learning control."""

    algorithm = "q_learning"

    def _next_value(self, next_state: BlackjackState, next_action: int) -> float:
        del next_action
        return max(self.q_values[next_state])
