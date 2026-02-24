"""MountainCar: build momentum to climb a hill.

A car stuck in a valley must rock back and forth to build
enough momentum to reach the goal flag on the right hill.
Demonstrates continuous state, discrete actions, and the
challenge of sparse reward / exploration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .config import (
    ACTION_LABELS,
    ACTION_LEFT,
    ACTION_RIGHT,
    MountainCarConfig,
    height,
)
from .renderer import MountainCarRenderer


class MountainCarEnv(gym.Env):
    """Car-on-a-hill with momentum-based physics.

    Observation:
        Box(2,) — [position, velocity].
    Actions:
        Discrete(3) — push left / no push / push right.
    Rewards:
        -1 per step (penalizes slow solutions).
    Termination:
        Car reaches goal_position with velocity >= goal_velocity.
    Truncation:
        Episode exceeds ``max_steps``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[MountainCarConfig] = None,
    ) -> None:
        self.cfg = config or MountainCarConfig()
        self.metadata["render_fps"] = self.cfg.render_fps

        low = np.array([self.cfg.min_position, -self.cfg.max_speed], dtype=np.float32)
        high = np.array([self.cfg.max_position, self.cfg.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.action_labels = ACTION_LABELS

        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self.render_mode = render_mode
        self._renderer: Optional[MountainCarRenderer] = None

    # -- core interface -------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        position = self.np_random.uniform(low=-0.6, high=-0.4)
        self.state = np.array([position, 0.0], dtype=np.float32)
        self._steps = 0
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() first"
        cfg = self.cfg
        position, velocity = float(self.state[0]), float(self.state[1])

        velocity += (action - 1) * cfg.force + math.cos(3 * position) * (-cfg.gravity)
        velocity = np.clip(velocity, -cfg.max_speed, cfg.max_speed)
        position += velocity
        position = np.clip(position, cfg.min_position, cfg.max_position)

        if position == cfg.min_position and velocity < 0:
            velocity = 0.0

        self.state = np.array([position, velocity], dtype=np.float32)
        self._steps += 1

        terminated = bool(
            position >= cfg.goal_position and velocity >= cfg.goal_velocity
        )
        truncated = self._steps >= cfg.max_steps
        reward = cfg.goal_reward if terminated else cfg.step_penalty

        if self.render_mode == "human":
            self.render()
        return self.state.copy(), reward, terminated, truncated, {"steps": self._steps}

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = MountainCarRenderer(self.cfg, self.render_mode)
        return self._renderer.draw(self.state, self._steps)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
