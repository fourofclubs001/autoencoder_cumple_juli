from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from .dqn_agent import DQNAgent
from .env import TwentyFortyEightWrapper


def epsilon_by_episode(
    episode: int, total_episodes: int, eps_start: float, eps_end: float
) -> float:
    decay_rate = -np.log(eps_end / eps_start) / total_episodes
    return max(eps_end, eps_start * np.exp(-decay_rate * episode))


def train(
    episodes: int = 5000,
    max_steps: int = 10000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    log_every: int = 20,
) -> DQNAgent:
    env = TwentyFortyEightWrapper()
    agent = DQNAgent()

    scores = deque(maxlen=100)
    max_tiles = deque(maxlen=100)

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        legal_mask = env.legal_action_mask()
        epsilon = epsilon_by_episode(episode, episodes, eps_start, eps_end)

        info = {}
        for _ in range(max_steps):
            action = agent.act(state, legal_mask, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_legal_mask = env.legal_action_mask()
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done, next_legal_mask)
            agent.learn()

            state = next_state
            legal_mask = next_legal_mask

            if done:
                break

        scores.append(info["total_score"])
        max_tiles.append(2 ** info["max"])

        if episode % log_every == 0:
            print(
                f"episode {episode:5d} | eps {epsilon:.3f} | "
                f"avg_score(100) {np.mean(scores):8.1f} | "
                f"avg_max_tile(100) {np.mean(max_tiles):7.1f}"
            )

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()
    train(episodes=args.episodes, max_steps=args.max_steps)
