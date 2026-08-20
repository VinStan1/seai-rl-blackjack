"""Tests for first-visit Monte Carlo control."""

import tempfile
import unittest
from pathlib import Path

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

    def test_training_seeds_environment_once_per_run(self) -> None:
        class RecordingEnvironment:
            def __init__(self) -> None:
                self.seeds = []

            def reset(self, *, seed=None):
                self.seeds.append(seed)
                return (20, 10, False), {}

            def step(self, action):
                return (20, 10, False), 1.0, True, False, {}

        environment = RecordingEnvironment()
        MonteCarloAgent(seed=17).train(environment, episodes=3)

        self.assertEqual(environment.seeds, [17, None, None])

    def test_training_records_every_agent_action(self) -> None:
        class TwoStepEnvironment:
            def reset(self, *, seed=None):
                return (12, 10, False), {}

            def step(self, action):
                if not hasattr(self, "second_step"):
                    self.second_step = True
                    return (16, 10, False), 0.0, False, False, {}
                del self.second_step
                return (20, 10, False), 1.0, True, False, {}

        agent = MonteCarloAgent(seed=17)
        agent.train(TwoStepEnvironment(), episodes=3)

        self.assertEqual(agent.last_training_action_count, 6)

    def test_linear_epsilon_schedule_reaches_end_after_decay_fraction(self) -> None:
        agent = MonteCarloAgent(
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_fraction=0.8,
        )

        self.assertEqual(agent.epsilon_for_episode(0, 101), 1.0)
        self.assertAlmostEqual(agent.epsilon_for_episode(40, 101), 0.525)
        self.assertEqual(agent.epsilon_for_episode(80, 101), 0.05)
        self.assertEqual(agent.epsilon_for_episode(100, 101), 0.05)

    def test_saved_agent_round_trips_extended_state(self) -> None:
        state = (18, 9, True, 24, 23, 24, 24, 24, 24, 24, 24, 95)
        agent = MonteCarloAgent(seed=7)
        agent.q_values[state] = [-0.5, 0.25]
        agent.visit_counts[state] = [3, 5]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            agent.save(path)
            restored = MonteCarloAgent.load(path)

        self.assertEqual(restored.q_values[state], [-0.5, 0.25])
        self.assertEqual(restored.visit_counts[state], [3, 5])


if __name__ == "__main__":
    unittest.main()
