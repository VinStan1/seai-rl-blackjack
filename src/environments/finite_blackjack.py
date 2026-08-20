"""Finite-shoe Blackjack variants with realistic between-hand reshuffling."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Literal, TypeAlias

BlackjackState: TypeAlias = tuple[int | bool, ...]
ObservationMode: TypeAlias = Literal["hidden", "hi_lo", "composition"]

CARD_VALUES = tuple(range(1, 11))
HI_LO_VALUES = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}


def _shoe_cards(decks: int) -> list[int]:
    cards: list[int] = []
    for _ in range(decks):
        cards.extend([1, 2, 3, 4, 5, 6, 7, 8, 9] * 4)
        cards.extend([10] * 16)
    return cards


def _usable_ace(hand: list[int]) -> bool:
    return 1 in hand and sum(hand) + 10 <= 21


def _hand_value(hand: list[int]) -> int:
    return sum(hand) + (10 if _usable_ace(hand) else 0)


def _natural(hand: list[int]) -> bool:
    return len(hand) == 2 and sorted(hand) == [1, 10]


def _compare(left: int, right: int) -> float:
    return float((left > right) - (left < right))


class FiniteBlackjackEnvironment:
    """Blackjack backed by a persistent finite shoe shared across hands.

    Six decks and 75% penetration are the defaults. The cut card is checked only
    before a new hand, so a shoe is never shuffled during an active hand.
    Informative observations include public cards only; the dealer hole card is
    incorporated when it is revealed.
    """

    def __init__(
        self,
        *,
        observation: ObservationMode = "hidden",
        decks: int = 6,
        penetration: float = 0.75,
        natural: bool = False,
        sab: bool = True,
    ) -> None:
        if observation not in {"hidden", "hi_lo", "composition"}:
            raise ValueError("observation must be hidden, hi_lo, or composition")
        if decks < 1:
            raise ValueError("decks must be positive")
        if not 0.0 < penetration < 1.0:
            raise ValueError("penetration must be between 0 and 1")

        self.observation = observation
        self.decks = decks
        self.penetration = penetration
        self.natural = natural
        self.sab = sab
        self._shoe_size = decks * 52
        self._cut_card = round(self._shoe_size * penetration)
        self._random = random.Random()
        self._shoe: list[int] = []
        self._cards_dealt = 0
        self._running_count = 0
        self._unseen_counts: Counter[int] = Counter()
        self._player: list[int] = []
        self._dealer: list[int] = []
        self._hole_revealed = False
        self._terminated = True
        self._shuffle_count = 0

    @property
    def number_actions(self) -> int:
        return 2

    @property
    def shuffle_count(self) -> int:
        return self._shuffle_count

    def _shuffle(self) -> None:
        self._shoe = _shoe_cards(self.decks)
        self._random.shuffle(self._shoe)
        self._cards_dealt = 0
        self._running_count = 0
        self._unseen_counts = Counter(self._shoe)
        self._shuffle_count += 1

    def _draw(self, *, visible: bool) -> int:
        card = self._shoe.pop()
        self._cards_dealt += 1
        if visible:
            self._observe(card)
        return card

    def _observe(self, card: int) -> None:
        self._running_count += HI_LO_VALUES[card]
        self._unseen_counts[card] -= 1

    def _reveal_hole_card(self) -> None:
        if not self._hole_revealed:
            self._observe(self._dealer[1])
            self._hole_revealed = True

    def _true_count_bucket(self) -> int:
        decks_remaining = max(len(self._shoe) / 52.0, 0.25)
        true_count = int(self._running_count / decks_remaining)
        return max(-20, min(20, true_count))

    def _observation(self) -> BlackjackState:
        base: BlackjackState = (
            _hand_value(self._player),
            self._dealer[0],
            _usable_ace(self._player),
        )
        if self.observation == "hidden":
            return base
        if self.observation == "hi_lo":
            return (*base, self._true_count_bucket())
        return (*base, *(self._unseen_counts[value] for value in CARD_VALUES))

    def _info(self, *, shuffled: bool = False) -> dict[str, Any]:
        return {
            "cards_dealt": self._cards_dealt,
            "cards_remaining": len(self._shoe),
            "cut_card": self._cut_card,
            "penetration": self.penetration,
            "shuffle_count": self._shuffle_count,
            "shuffled": shuffled,
        }

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[BlackjackState, dict[str, Any]]:
        if seed is not None:
            self._random.seed(seed)
            self._shoe.clear()

        shuffled = False
        if not self._shoe or self._cards_dealt >= self._cut_card:
            self._shuffle()
            shuffled = True

        self._player = []
        self._dealer = []
        self._hole_revealed = False
        self._terminated = False
        self._player.append(self._draw(visible=True))
        self._dealer.append(self._draw(visible=True))
        self._player.append(self._draw(visible=True))
        self._dealer.append(self._draw(visible=False))
        while _hand_value(self._player) < 12:
            self._player.append(self._draw(visible=True))
        return self._observation(), self._info(shuffled=shuffled)

    def step(
        self, action: int
    ) -> tuple[BlackjackState, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("step called after the hand terminated; call reset")
        if action not in {0, 1}:
            raise ValueError("action must be 0 (stick) or 1 (hit)")

        if action == 1:
            self._player.append(self._draw(visible=True))
            if _hand_value(self._player) > 21:
                self._terminated = True
                return self._observation(), -1.0, True, False, self._info()
            return self._observation(), 0.0, False, False, self._info()

        self._reveal_hole_card()
        while _hand_value(self._dealer) < 17:
            self._dealer.append(self._draw(visible=True))

        player_value = _hand_value(self._player)
        dealer_value = _hand_value(self._dealer)
        reward = _compare(player_value, dealer_value) if dealer_value <= 21 else 1.0
        if self.sab and _natural(self._player) and not _natural(self._dealer):
            reward = 1.0
        elif not self.sab and self.natural and _natural(self._player) and reward == 1.0:
            reward = 1.5
        self._terminated = True
        return self._observation(), reward, True, False, self._info()

    def close(self) -> None:
        pass

    def __enter__(self) -> FiniteBlackjackEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()