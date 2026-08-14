"""Tests for first-visit Monte Carlo control."""

import unittest

from src.agents.monte_carlo import MonteCarloAgent


class MonteCarloAgentTests(unittest.TestCase):
    def test_update_uses_first_state_action_visit(self) -> None:
        agent = MonteCarloAgent(gamma=0.5)
        repeated_state = (15, 10, False)
        episode = [
            (repeated_state, 1, 1.0),
            ((17, 10, False), 1, 2.0),
            (repeated_state, 1, 4.0),
        ]

        agent.update_episode(episode)

        self.assertEqual(agent.visit_counts[repeated_state][1], 1)
        self.assertAlmostEqual(agent.q_values[repeated_state][1], 3.0)

    def test_incremental_mean_averages_observed_returns(self) -> None:
        agent = MonteCarloAgent()
        state = (20, 10, False)

        agent.update_episode([(state, 0, 1.0)])
        agent.update_episode([(state, 0, -1.0)])

        self.assertEqual(agent.visit_counts[state][0], 2)
        self.assertAlmostEqual(agent.q_values[state][0], 0.0)


if __name__ == "__main__":
    unittest.main()