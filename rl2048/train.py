from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    checkpoint_every: int = 500,
    output_dir: str | Path = "./out",
) -> DQNAgent:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "dqn_2048.pt"
    history_path = out_dir / "dqn_2048_history.csv"
    plot_path = out_dir / "dqn_2048_training.png"

    env = TwentyFortyEightWrapper()
    agent = DQNAgent()

    start_episode = 1
    history: list[dict] = []

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=agent.device)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = ckpt["episode"] + 1
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed from episode {ckpt['episode']}")

    scores = deque(maxlen=100)
    max_tiles = deque(maxlen=100)

    for episode in range(start_episode, episodes + 1):
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
        history.append({
            "episode": episode,
            "score": info["total_score"],
            "max_tile": 2 ** info["max"],
            "epsilon": epsilon,
        })

        if episode % log_every == 0:
            print(
                f"episode {episode:5d} | eps {epsilon:.3f} | "
                f"avg_score(100) {np.mean(scores):8.1f} | "
                f"avg_max_tile(100) {np.mean(max_tiles):7.1f}"
            )

        if episode % checkpoint_every == 0 or episode == episodes:
            torch.save({
                "episode": episode,
                "online": agent.online.state_dict(),
                "target": agent.target.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
            }, checkpoint_path)
            pd.DataFrame(history).to_csv(history_path, index=False)

    _plot_history(history, plot_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved history to {history_path}")
    print(f"Saved plot to {plot_path}")

    return agent


def _plot_history(history: list[dict], plot_path: Path) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    df["avg_score_100"] = df["score"].rolling(100, min_periods=1).mean()
    df["avg_max_tile_100"] = df["max_tile"].rolling(100, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["episode"], df["score"], alpha=0.3, color="tab:orange")
    axes[0].plot(df["episode"], df["avg_score_100"], color="tab:red")
    axes[0].set_title("Score per episode")
    axes[0].set_xlabel("Episode")

    axes[1].plot(df["episode"], df["max_tile"], alpha=0.3, color="tab:orange")
    axes[1].plot(df["episode"], df["avg_max_tile_100"], color="tab:red")
    axes[1].set_title("Max tile per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_yscale("log", base=2)

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="./out")
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
    )
