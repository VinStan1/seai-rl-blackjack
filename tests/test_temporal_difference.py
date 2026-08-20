"""Tests for the tabular SARSA and Q-learning agents."""

import tempfile
import unittest
from pathlib import Path

from src.agents.temporal_difference import QLearningAgent, SarsaAgent


class OneStepEnvironment:
    def __init__(self) -> None:
        self.seeds = []

    def reset(self, *, seed: int | None = None):
        self.seeds.append(seed)
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
        agent = QLearningAgent(
            alpha=0.1,
            epsilon=0.2,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_fraction=0.8,
            seed=7,
        )
        agent.q_values[(18, 9, True)] = [-0.5, 0.25]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            agent.save(path)
            restored = QLearningAgent.load(path)

        self.assertEqual(restored.seed, 7)
        self.assertEqual(restored.epsilon_start, 1.0)
        self.assertEqual(restored.epsilon_end, 0.05)
        self.assertEqual(restored.q_values[(18, 9, True)], [-0.5, 0.25])

    def test_saved_agent_round_trips_extended_state(self) -> None:
        state = (18, 9, True, -2)
        agent = SarsaAgent(seed=8)
        agent.q_values[state] = [0.125, -0.25]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.json"
            agent.save(path)
            restored = SarsaAgent.load(path)

        self.assertEqual(restored.q_values[state], [0.125, -0.25])

    def test_training_seeds_environment_once_per_run(self) -> None:
        for agent_type in (SarsaAgent, QLearningAgent):
            with self.subTest(agent=agent_type.__name__):
                environment = OneStepEnvironment()
                agent = agent_type(seed=23)
                agent.train(environment, episodes=3)
                self.assertEqual(environment.seeds, [23, None, None])
                self.assertEqual(agent.last_training_action_count, 3)

    def test_linear_epsilon_schedule_is_shared_by_td_agents(self) -> None:
        for agent_type in (SarsaAgent, QLearningAgent):
            with self.subTest(agent=agent_type.__name__):
                agent = agent_type(
                    epsilon_start=1.0,
                    epsilon_end=0.05,
                    epsilon_decay_fraction=0.8,
                )
                self.assertEqual(agent.epsilon_for_episode(0, 101), 1.0)
                self.assertAlmostEqual(
                    agent.epsilon_for_episode(40, 101), 0.525
                )
                self.assertEqual(agent.epsilon_for_episode(80, 101), 0.05)


if __name__ == "__main__":
    unittest.main()
