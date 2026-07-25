from __future__ import annotations

import argparse
import time

import torch

from .dqn_agent import DQNAgent
from .env import TwentyFortyEightWrapper


def play(
    checkpoint: str = "./out/dqn_2048.pt",
    episodes: int = 3,
    max_steps: int = 10000,
    delay: float = 0.3,
) -> None:
    agent = DQNAgent()
    ckpt = torch.load(checkpoint, map_location=agent.device)
    agent.online.load_state_dict(ckpt["online"])
    agent.online.eval()

    env = TwentyFortyEightWrapper(render_mode="human")
    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            legal_mask = env.legal_action_mask()

            info = {}
            for _ in range(max_steps):
                action = agent.act(state, legal_mask, epsilon=0.0)
                state, _, terminated, truncated, info = env.step(action)
                legal_mask = env.legal_action_mask()
                time.sleep(delay)
                if terminated or truncated:
                    break

            print(
                f"Episode {episode}: score={info['total_score']} "
                f"max_tile={2 ** info['max']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./out/dqn_2048.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument(
        "--delay", type=float, default=0.3, help="seconds to pause between moves"
    )
    args = parser.parse_args()
    play(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        delay=args.delay,
    )
