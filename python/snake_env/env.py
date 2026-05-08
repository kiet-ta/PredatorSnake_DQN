from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import predator_snake_dqn


class SnakeEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"], "name": "snake-rust-v0"}

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        max_steps: int = 1_000,
        render: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.render_enabled = render

        self._core = predator_snake_dqn.PySnakeCore(
            width=width,
            height=height,
            max_steps=max_steps,
            render=render,
            seed=seed,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.uint8(0),
            high=np.uint8(1),
            shape=(5, height, width),
            dtype=np.uint8,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._core = predator_snake_dqn.PySnakeCore(
                width=self.width,
                height=self.height,
                max_steps=self.max_steps,
                render=self.render_enabled,
                seed=seed,
            )

        obs, info = self._core.reset()
        return obs, dict(info)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._core.step(int(action))
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)

    def render(self) -> None:
        if self.render_enabled:
            self._core.render_tick()
        return None

    def close(self) -> None:
        return None
