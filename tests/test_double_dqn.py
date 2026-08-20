"""Tests for the optional composition-aware Double DQN agent."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.agents.double_dqn import (
    DoubleDQNAgent,
    double_dqn_targets,
    encode_composition_state,
)


class DoubleDQNAgentTests(unittest.TestCase):
    def test_composition_encoder_normalizes_all_thirteen_features(self) -> None:
        state = (21, 10, True, 24, 24, 24, 24, 24, 24, 24, 24, 24, 96)

        encoded = encode_composition_state(state, decks=6)

        self.assertEqual(tuple(encoded.shape), (13,))
        self.assertTrue(torch.equal(encoded, torch.ones(13)))

    def test_encoder_rejects_non_composition_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite_composition"):
            encode_composition_state((18, 10, False))

    def test_double_dqn_target_uses_online_action_and_target_value(self) -> None:
        online = torch.tensor([[1.0, 3.0], [4.0, 2.0]])
        target = torch.tensor([[10.0, 5.0], [7.0, 20.0]])
        rewards = torch.tensor([1.0, -1.0])
        terminated = torch.tensor([0.0, 1.0])

        targets = double_dqn_targets(
            online, target, rewards, terminated, gamma=0.5
        )

        self.assertTrue(torch.equal(targets, torch.tensor([3.5, -1.0])))

    def test_training_updates_network_and_records_actions(self) -> None:
        state = (18, 10, False, 24, 24, 24, 24, 24, 24, 24, 24, 24, 96)

        class OneStepEnvironment:
            def reset(self, *, seed=None):
                return state, {}

            def step(self, action):
                return state, 1.0, True, False, {}

        agent = DoubleDQNAgent(
            batch_size=1,
            replay_capacity=10,
            learning_starts=1,
            train_frequency=1,
            target_update_interval=1,
            seed=4,
        )

        rewards = agent.train(OneStepEnvironment(), episodes=3)

        self.assertEqual(rewards, [1.0, 1.0, 1.0])
        self.assertEqual(agent.last_training_action_count, 3)
        self.assertEqual(agent.gradient_steps, 3)

    def test_checkpoint_round_trips_greedy_policy(self) -> None:
        state = (18, 10, False, 24, 24, 24, 24, 24, 24, 24, 24, 24, 96)
        agent = DoubleDQNAgent(seed=5)
        expected_action = agent.select_action(state, explore=False)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            agent.save(path)
            restored = DoubleDQNAgent.load(path)

        self.assertEqual(restored.select_action(state, explore=False), expected_action)


if __name__ == "__main__":
    unittest.main()