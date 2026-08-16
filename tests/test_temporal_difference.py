"""Tests for the tabular SARSA and Q-learning agents."""

import tempfile
import unittest
from pathlib import Path

from src.agents.temporal_difference import QLearningAgent, SarsaAgent


class OneStepEnvironment:
    def reset(self, *, seed: int | None = None):
        del seed
        return (12, 5, False), {}

    def step(self, action: int):
        del action
        return (20, 5, False), 1.0, True, False, {}


class TemporalDifferenceAgentTests(unittest.TestCase):
    def test_terminal_reward_updates_q_value(self) -> None:
        for agent_type in (SarsaAgent, QLearningAgent):
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(alpha=0.5, epsilon=0.0)
                agent.train(OneStepEnvironment(), episodes=1)
                values = agent.q_values[(12, 5, False)]
                self.assertAlmostEqual(sum(values), 0.5)
                self.assertEqual(sum(value != 0.0 for value in values), 1)

    def test_sarsa_uses_selected_action_but_q_learning_uses_maximum(self) -> None:
        state = (16, 10, False)
        sarsa = SarsaAgent()
        q_learning = QLearningAgent()
        sarsa.q_values[state] = [3.0, 1.0]
        q_learning.q_values[state] = [3.0, 1.0]

        self.assertEqual(sarsa._next_value(state, 1), 1.0)
        self.assertEqual(q_learning._next_value(state, 1), 3.0)

    def test_saved_agent_round_trips(self) -> None:
        agent = QLearningAgent(alpha=0.1, epsilon=0.2, seed=7)
        agent.q_values[(18, 9, True)] = [-0.5, 0.25]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            agent.save(path)
            restored = QLearningAgent.load(path)

        self.assertEqual(restored.seed, 7)
        self.assertEqual(restored.q_values[(18, 9, True)], [-0.5, 0.25])


if __name__ == "__main__":
    unittest.main()
