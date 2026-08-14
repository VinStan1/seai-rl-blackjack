"""Contract tests for the Gymnasium Blackjack adapter."""

import unittest

from src.environments.blackjack import BlackjackEnvironment


class BlackjackEnvironmentTests(unittest.TestCase):
    def test_reset_and_step_follow_expected_discrete_contract(self) -> None:
        with BlackjackEnvironment() as environment:
            state, info = environment.reset(seed=42)
            next_state, reward, terminated, truncated, step_info = environment.step(0)

        self.assertEqual(environment.number_actions, 2)
        self.assertEqual(len(state), 3)
        self.assertEqual(len(next_state), 3)
        self.assertIsInstance(state[2], bool)
        self.assertIn(reward, {-1.0, 0.0, 1.0})
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIsInstance(info, dict)
        self.assertIsInstance(step_info, dict)

    def test_seeded_resets_are_reproducible(self) -> None:
        with BlackjackEnvironment() as environment:
            first_state, _ = environment.reset(seed=7)
            second_state, _ = environment.reset(seed=7)

        self.assertEqual(first_state, second_state)


if __name__ == "__main__":
    unittest.main()