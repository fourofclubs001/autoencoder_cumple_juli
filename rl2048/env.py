from __future__ import annotations

import gymnasium as gym
import numpy as np

import gymnasium_2048  # noqa: F401 (registers the env id on import)
from gymnasium_2048.envs.twenty_forty_eight import TwentyFortyEightEnv

ENV_ID = "gymnasium_2048/TwentyFortyEight-v0"
NUM_ACTIONS = 4


def monotonicity_bonus(board: np.ndarray) -> float:
    """Rewards rows/columns whose log2 tile values are monotonic, as in
    Saligram et al. 2025 (Section 4.1). Board entries are exponents (0 = empty)."""

    def line_score(line: np.ndarray) -> float:
        increasing = 0.0
        decreasing = 0.0
        for a, b in zip(line[:-1], line[1:]):
            a, b = float(a), float(b)
            if a > b:
                decreasing += b - a
            elif b > a:
                increasing += a - b
        return max(increasing, decreasing)

    return sum(line_score(row) for row in board) + sum(
        line_score(col) for col in board.T
    )


class TwentyFortyEightWrapper(gym.Wrapper):
    """Reproduces the DQN baseline setup from Saligram et al. 2025:
    - state: log2-scaled board flattened to 16 dims (Section 3.1)
    - reward: merge score + lambda_mono * monotonicity bonus (Section 4.1)
    """

    def __init__(self, mono_weight: float = 0.01, render_mode: str | None = None):
        env = gym.make(ENV_ID, render_mode=render_mode)
        super().__init__(env)
        self.mono_weight = mono_weight
        self.observation_space = gym.spaces.Box(
            low=0, high=16, shape=(16,), dtype=np.float32
        )

    def _encode(self, info: dict) -> np.ndarray:
        return info["board"].flatten().astype(np.float32)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._encode(info), info

    def step(self, action: int):
        _, score, terminated, truncated, info = self.env.step(action)
        reward = float(score) + self.mono_weight * monotonicity_bonus(info["board"])
        return self._encode(info), reward, terminated, truncated, info

    def legal_action_mask(self) -> np.ndarray:
        """Valid moves, computed via the env's own deterministic slide function
        rather than sampling blindly (Section 3.1: 'Only valid moves are sampled')."""
        board = self.unwrapped.board
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for action in range(NUM_ACTIONS):
            _, _, is_legal = TwentyFortyEightEnv.apply_action(board, action)
            mask[action] = is_legal
        return mask
