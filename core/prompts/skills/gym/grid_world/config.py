"""Configuration and constants for GridWorld environment."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class GridWorldConfig:
    """All tunable parameters for the GridWorld environment."""

    size: int = 5
    max_steps: int = 100
    render_fps: int = 8
    cell_pixels: int = 96
    step_penalty: float = -0.01
    goal_reward: float = 1.0

    bg_color: Tuple[int, int, int] = (245, 240, 230)
    grid_color: Tuple[int, int, int] = (200, 195, 185)
    agent_color: Tuple[int, int, int] = (65, 92, 87)
    target_color: Tuple[int, int, int] = (103, 138, 99)
    text_color: Tuple[int, int, int] = (80, 80, 80)


ACTION_RIGHT = 0
ACTION_UP = 1
ACTION_LEFT = 2
ACTION_DOWN = 3

ACTION_LABELS: List[str] = ["right", "up", "left", "down"]

ACTION_TO_DIRECTION: Dict[int, np.ndarray] = {
    ACTION_RIGHT: np.array([0, 1]),
    ACTION_UP: np.array([-1, 0]),
    ACTION_LEFT: np.array([0, -1]),
    ACTION_DOWN: np.array([1, 0]),
}

KEY_ACTION_MAP: Dict[int, int] = {}  # populated by interactive.py at runtime
