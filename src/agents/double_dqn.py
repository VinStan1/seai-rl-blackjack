"""Double DQN with function approximation for composition-aware Blackjack."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import torch
from torch import nn

BlackjackState: TypeAlias = tuple[int | bool, ...]


class BlackjackEnvironment(Protocol):
    def reset(
        self, *, seed: int | None = None
    ) -> tuple[BlackjackState, dict[str, Any]]: ...

    def step(
        self, action: int
    ) -> tuple[BlackjackState, float, bool, bool, dict[str, Any]]: ...


def encode_composition_state(
    state: Sequence[int | bool], *, decks: int = 6
) -> torch.Tensor:
    """Normalize a 13-value exact-composition observation for a neural network."""
    if len(state) != 13:
        raise ValueError(
            "Double DQN requires finite_composition states with 13 values"
        )
    if decks < 1:
        raise ValueError("decks must be positive")
    initial_counts = [decks * 4.0] * 9 + [decks * 16.0]
    features = [
        float(state[0]) / 21.0,
        float(state[1]) / 10.0,
        float(bool(state[2])),
        *(
            float(count) / initial
            for count, initial in zip(state[3:], initial_counts, strict=True)
        ),
    ]
    return torch.tensor(features, dtype=torch.float32)


def double_dqn_targets(
    online_next_values: torch.Tensor,
    target_next_values: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Select next actions online and evaluate those actions with the target net."""
    next_actions = online_next_values.argmax(dim=1, keepdim=True)
    selected_target_values = target_next_values.gather(1, next_actions).squeeze(1)
    return rewards + gamma * (1.0 - terminated) * selected_target_values


@dataclass(frozen=True)
class Transition:
    state: BlackjackState
    action: int
    reward: float
    next_state: BlackjackState
    terminated: bool


class ReplayBuffer:
    """Fixed-size replay memory with deterministic sampling from the agent RNG."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._items: list[Transition] = []
        self._next_index = 0

    def append(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
            return
        self._items[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, size: int, random_source: random.Random) -> list[Transition]:
        return [self._items[index] for index in random_source.sample(range(len(self)), size)]

    def __len__(self) -> int:
        return len(self._items)


class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, number_actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, number_actions),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DoubleDQNAgent:
    """CPU Double DQN agent for exact finite-shoe composition observations."""

    algorithm = "double_dqn"
    model_extension = ".pt"
    input_size = 13

    def __init__(
        self,
        *,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_fraction: float = 0.8,
        learning_rate: float = 0.001,
        gamma: float = 1.0,
        batch_size: int = 64,
        replay_capacity: int = 100_000,
        learning_starts: int = 1_000,
        train_frequency: int = 4,
        target_update_interval: int = 1_000,
        hidden_size: int = 64,
        gradient_clip: float = 10.0,
        decks: int = 6,
        number_actions: int = 2,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 < epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction must be in (0, 1]")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if batch_size < 1 or replay_capacity < batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if learning_starts < 0:
            raise ValueError("learning_starts must not be negative")
        if train_frequency < 1 or target_update_interval < 1:
            raise ValueError("training and target-update intervals must be positive")
        if hidden_size < 1 or number_actions < 1 or decks < 1:
            raise ValueError("network size, actions, and decks must be positive")
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_fraction = epsilon_decay_fraction
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_interval = target_update_interval
        self.hidden_size = hidden_size
        self.gradient_clip = gradient_clip
        self.decks = decks
        self.number_actions = number_actions
        self.seed = seed
        self._active_epsilon = epsilon_start
        self._random = random.Random(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)

        self.online_network = QNetwork(
            self.input_size, hidden_size, number_actions
        )
        self.target_network = QNetwork(
            self.input_size, hidden_size, number_actions
        )
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.last_training_action_count = 0
        self.gradient_steps = 0

    def epsilon_for_episode(self, episode_index: int, episodes: int) -> float:
        decay_episodes = max(1, round((episodes - 1) * self.epsilon_decay_fraction))
        if episode_index >= decay_episodes:
            return self.epsilon_end
        progress = episode_index / decay_episodes
        return self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )

    def _encode(self, state: Sequence[int | bool]) -> torch.Tensor:
        return encode_composition_state(state, decks=self.decks)

    def select_action(self, state: BlackjackState, *, explore: bool = True) -> int:
        if explore and self._random.random() < self._active_epsilon:
            return self._random.randrange(self.number_actions)
        with torch.no_grad():
            values = self.online_network(self._encode(state).unsqueeze(0))
        return int(values.argmax(dim=1).item())

    def _optimize(self) -> None:
        transitions = self.replay_buffer.sample(self.batch_size, self._random)
        states = torch.stack([self._encode(item.state) for item in transitions])
        actions = torch.tensor(
            [item.action for item in transitions], dtype=torch.int64
        )
        rewards = torch.tensor(
            [item.reward for item in transitions], dtype=torch.float32
        )
        next_states = torch.stack(
            [self._encode(item.next_state) for item in transitions]
        )
        terminated = torch.tensor(
            [item.terminated for item in transitions], dtype=torch.float32
        )

        selected_values = self.online_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            targets = double_dqn_targets(
                self.online_network(next_states),
                self.target_network(next_states),
                rewards,
                terminated,
                self.gamma,
            )
        loss = nn.functional.smooth_l1_loss(selected_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.gradient_steps += 1

    def train(
        self, environment: BlackjackEnvironment, episodes: int
    ) -> list[float]:
        if episodes < 1:
            raise ValueError("episodes must be positive")

        rewards: list[float] = []
        self.last_training_action_count = 0
        for episode_index in range(episodes):
            self._active_epsilon = self.epsilon_for_episode(
                episode_index, episodes
            )
            state, _ = environment.reset(
                seed=self.seed if episode_index == 0 else None
            )
            episode_reward = 0.0
            while True:
                action = self.select_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                finished = terminated or truncated
                reward = float(reward)
                self.replay_buffer.append(
                    Transition(state, action, reward, next_state, finished)
                )
                self.last_training_action_count += 1
                episode_reward += reward

                if (
                    self.last_training_action_count >= self.learning_starts
                    and len(self.replay_buffer) >= self.batch_size
                    and self.last_training_action_count % self.train_frequency == 0
                ):
                    self._optimize()
                if (
                    self.last_training_action_count % self.target_update_interval
                    == 0
                ):
                    self.target_network.load_state_dict(
                        self.online_network.state_dict()
                    )
                if finished:
                    break
                state = next_state
            rewards.append(episode_reward)
        return rewards

    def training_diagnostics(self) -> dict[str, int]:
        return {
            "network_parameters": sum(
                parameter.numel() for parameter in self.online_network.parameters()
            ),
            "replay_transitions": len(self.replay_buffer),
            "gradient_steps": self.gradient_steps,
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "parameters": self._constructor_parameters(),
                "online_state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "gradient_steps": self.gradient_steps,
                "metadata": dict(metadata or {}),
            },
            output_path,
        )

    def _constructor_parameters(self) -> dict[str, int | float]:
        return {
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_fraction": self.epsilon_decay_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "learning_starts": self.learning_starts,
            "train_frequency": self.train_frequency,
            "target_update_interval": self.target_update_interval,
            "hidden_size": self.hidden_size,
            "gradient_clip": self.gradient_clip,
            "decks": self.decks,
            "number_actions": self.number_actions,
            "seed": self.seed,
        }

    @classmethod
    def load(cls, path: str | Path) -> DoubleDQNAgent:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(**payload["parameters"])
        agent.online_network.load_state_dict(payload["online_state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.gradient_steps = int(payload.get("gradient_steps", 0))
        return agent