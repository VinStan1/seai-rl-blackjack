"""Tests for finite-shoe Blackjack observation variants."""

import unittest
from collections import Counter

from src.environments.finite_blackjack import (
    HI_LO_VALUES,
    FiniteBlackjackEnvironment,
    _shoe_cards,
)


class FiniteBlackjackEnvironmentTests(unittest.TestCase):
    def test_six_deck_shoe_has_standard_value_multiplicities(self) -> None:
        counts = Counter(_shoe_cards(6))

        self.assertEqual(sum(counts.values()), 312)
        self.assertEqual([counts[value] for value in range(1, 10)], [24] * 9)
        self.assertEqual(counts[10], 96)

    def test_variants_share_six_deck_shoe_but_encode_different_states(self) -> None:
        expected_lengths = {"hidden": 3, "hi_lo": 4, "composition": 13}

        for mode, expected_length in expected_lengths.items():
            with self.subTest(mode=mode):
                environment = FiniteBlackjackEnvironment(observation=mode)
                state, info = environment.reset(seed=42)

                self.assertEqual(len(state), expected_length)
                self.assertGreaterEqual(info["cards_dealt"], 4)
                self.assertEqual(
                    info["cards_remaining"], 312 - info["cards_dealt"]
                )
                self.assertEqual(info["cut_card"], 234)
                self.assertTrue(info["shuffled"])

    def test_reset_does_not_auto_play_player_totals_below_twelve(self) -> None:
        environment = FiniteBlackjackEnvironment(observation="hidden")
        environment._shoe = [10, 2, 10, 2]

        state, info = environment.reset()

        self.assertEqual(state, (4, 10, False))
        self.assertEqual(info["cards_dealt"], 4)

    def test_composition_tracks_public_cards_without_revealing_hole_card(self) -> None:
        environment = FiniteBlackjackEnvironment(observation="composition")
        state, _ = environment.reset(seed=7)

        public_cards = environment._cards_dealt - 1
        self.assertEqual(sum(state[3:]), 312 - public_cards)

        terminal_state, _, terminated, _, _ = environment.step(0)

        self.assertTrue(terminated)
        self.assertLess(sum(terminal_state[3:]), sum(state[3:]))

    def test_hi_lo_count_excludes_the_dealer_hole_card_until_reveal(self) -> None:
        environment = FiniteBlackjackEnvironment(observation="hi_lo")
        environment.reset(seed=23)
        expected_public_count = sum(
            HI_LO_VALUES[card]
            for card in [*environment._player, environment._dealer[0]]
        )

        self.assertEqual(environment._running_count, expected_public_count)

        environment.step(0)

        self.assertTrue(environment._hole_revealed)

    def test_cut_card_reshuffles_only_before_the_next_hand(self) -> None:
        environment = FiniteBlackjackEnvironment(penetration=0.01)
        _, first_info = environment.reset(seed=11)
        first_shuffle_count = environment.shuffle_count

        _, second_info = environment.reset()

        self.assertGreaterEqual(first_info["cards_dealt"], 4)
        self.assertEqual(environment.shuffle_count, first_shuffle_count + 1)
        self.assertTrue(second_info["shuffled"])
        self.assertEqual(second_info["cards_dealt"], 4)

    def test_cut_card_never_triggers_a_mid_hand_shuffle(self) -> None:
        environment = FiniteBlackjackEnvironment(penetration=0.01)
        environment.reset(seed=11)
        shuffle_count = environment.shuffle_count

        environment.step(1)

        self.assertEqual(environment.shuffle_count, shuffle_count)

    def test_shoe_persists_between_hands_before_cutoff(self) -> None:
        environment = FiniteBlackjackEnvironment(observation="hidden")
        _, first_info = environment.reset(seed=19)
        environment.step(0)
        _, second_info = environment.reset()

        self.assertFalse(second_info["shuffled"])
        self.assertGreater(second_info["cards_dealt"], first_info["cards_dealt"])
        self.assertEqual(environment.shuffle_count, 1)


if __name__ == "__main__":
    unittest.main()