"""Typed adapter for Gymnasium's infinite-deck Blackjack environment."""

from __future__ import annotations

from typing import Any, TypeAlias

import gymnasium as gym

BlackjackState: TypeAlias = tuple[int, int, bool]


def _normalize_state(state: tuple[int, int, int]) -> BlackjackState:
    return int(state[0]), int(state[1]), bool(state[2])


class BlackjackEnvironment:
    """Expose the Gymnasium Blackjack-v1 API used by the tabular agents."""

    def __init__(self, *, natural: bool = False, sab: bool = True) -> None:
        self.natural = natural
        self.sab = sab
        self._environment = gym.make(
            "Blackjack-v1",
            natural=natural,
            sab=sab,
        )

    @property
    def number_actions(self) -> int:
        """Return the number of discrete actions: stick and hit."""
        return int(self._environment.action_space.n)

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[BlackjackState, dict[str, Any]]:
        state, info = self._environment.reset(seed=seed)
        return _normalize_state(state), info

    def step(
        self, action: int
    ) -> tuple[BlackjackState, float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self._environment.step(action)
        return (
            _normalize_state(state),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Release resources owned by the Gymnasium environment."""
        self._environment.close()

    def __enter__(self) -> BlackjackEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()