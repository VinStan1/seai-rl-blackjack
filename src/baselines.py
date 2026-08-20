"""Reference policies used to contextualize learned Blackjack agents."""

from __future__ import annotations

from typing import TypeAlias

BlackjackState: TypeAlias = tuple[int | bool, ...]

DEALER_POLICY_REFERENCE = {
    "title": "Reinforcement Learning: An Introduction",
    "authors": "Richard S. Sutton and Andrew G. Barto",
    "edition": "Second edition",
    "year": 2018,
    "location": "Example 5.1: Blackjack",
    "url": "http://incompleteideas.net/book/RLbook2020.pdf",
}


class StickOnSeventeenPolicy:
    """Dealer-like policy from Sutton and Barto: hit below 17, otherwise stick."""

    name = "stick_on_17"
    description = "Hit when the player sum is below 17; stick on 17 or above."
    reference = DEALER_POLICY_REFERENCE

    def select_action(self, state: BlackjackState, *, explore: bool = False) -> int:
        del explore
        player_sum = int(state[0])
        return 1 if player_sum < 17 else 0