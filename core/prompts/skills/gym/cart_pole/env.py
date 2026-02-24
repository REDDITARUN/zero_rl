"""CartPole: balance a pole on a sliding cart.

Classic control task. Push the cart left or right to keep
the pole upright. Demonstrates continuous observation space,
Euler physics integration, and per-step alive reward.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .config import ACTION_LABELS, ACTION_LEFT, CartPoleConfig
from .renderer import CartPoleRenderer


class CartPoleEnv(gym.Env):
    """Cart-pole balancing with configurable physics.

    Observation:
        Box(4,) — [x, x_dot, theta, theta_dot].
    Actions:
        Discrete(2) — push left / push right.
    Rewards:
        +1 for every step the pole remains upright.
    Termination:
        Cart leaves bounds or pole angle exceeds threshold.
    Truncation:
        Episode exceeds ``max_steps``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[CartPoleConfig] = None,
    ) -> None:
        self.cfg = config or CartPoleConfig()
        self.metadata["render_fps"] = self.cfg.render_fps

        high = np.array(
            [
                self.cfg.x_threshold * 2,
                np.finfo(np.float32).max,
                self.cfg.theta_threshold_rad * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.action_labels = ACTION_LABELS

        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self.render_mode = render_mode
        self._renderer: Optional[CartPoleRenderer] = None

    # -- core interface -------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(
            np.float32
        )
        self._steps = 0
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() first"
        x, x_dot, theta, theta_dot = self.state
        cfg = self.cfg

        force = cfg.force_mag if action == ACTION_LEFT else -cfg.force_mag
        force = -force  # convention: action 0 pushes left

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        # Euler integration of the equations of motion
        temp = (
            force + cfg.pole_mass_length * theta_dot**2 * sin_th
        ) / cfg.total_mass
        theta_acc = (cfg.gravity * sin_th - cos_th * temp) / (
            cfg.pole_length
            * (4.0 / 3.0 - cfg.mass_pole * cos_th**2 / cfg.total_mass)
        )
        x_acc = temp - cfg.pole_mass_length * theta_acc * cos_th / cfg.total_mass

        x = x + cfg.tau * x_dot
        x_dot = x_dot + cfg.tau * x_acc
        theta = theta + cfg.tau * theta_dot
        theta_dot = theta_dot + cfg.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self._steps += 1

        terminated = bool(
            x < -cfg.x_threshold
            or x > cfg.x_threshold
            or theta < -cfg.theta_threshold_rad
            or theta > cfg.theta_threshold_rad
        )
        truncated = self._steps >= cfg.max_steps
        reward = cfg.reward_alive if not terminated else 0.0

        if self.render_mode == "human":
            self.render()
        return self.state.copy(), reward, terminated, truncated, {"steps": self._steps}

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = CartPoleRenderer(self.cfg, self.render_mode)
        return self._renderer.draw(self.state, self._steps)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
