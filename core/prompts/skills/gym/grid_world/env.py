"""GridWorld: minimal navigation environment on a 2D grid.

The agent must navigate to a randomly placed target.
Simplest possible custom Gymnasium environment — ideal as a
template for grid-based RL tasks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .config import (
    ACTION_LABELS,
    ACTION_TO_DIRECTION,
    GridWorldConfig,
)
from .renderer import GridWorldRenderer


class GridWorldEnv(gym.Env):
    """A square grid where an agent navigates to a target cell.

    Observation:
        Dict with 'agent' and 'target' — each an (x, y) coordinate.
    Actions:
        Discrete(4) — right / up / left / down.
    Rewards:
        +1 on reaching the target, optional small step penalty.
    Termination:
        Agent lands on target cell.
    Truncation:
        Episode exceeds ``max_steps``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[GridWorldConfig] = None,
    ) -> None:
        self.cfg = config or GridWorldConfig()
        self.metadata["render_fps"] = self.cfg.render_fps

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.cfg.size - 1, shape=(2,), dtype=np.int32),
                "target": spaces.Box(0, self.cfg.size - 1, shape=(2,), dtype=np.int32),
            }
        )
        self.action_space = spaces.Discrete(4)

        self.action_labels = ACTION_LABELS
        self.render_mode = render_mode
        self._renderer: Optional[GridWorldRenderer] = None

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._steps = 0

    # -- core interface -------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0

        self._agent_location = self.np_random.integers(
            0, self.cfg.size, size=2
        ).astype(np.int32)
        self._target_location = self._agent_location.copy()
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.cfg.size, size=2
            ).astype(np.int32)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        direction = ACTION_TO_DIRECTION[int(action)]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.cfg.size - 1
        ).astype(np.int32)
        self._steps += 1

        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        truncated = self._steps >= self.cfg.max_steps
        reward = self.cfg.goal_reward if terminated else self.cfg.step_penalty

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = GridWorldRenderer(self.cfg, self.render_mode)
        return self._renderer.draw(
            self._agent_location,
            self._target_location,
            self._steps,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- helpers --------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "distance": int(
                np.linalg.norm(
                    self._agent_location - self._target_location, ord=1
                )
            ),
            "steps": self._steps,
        }
