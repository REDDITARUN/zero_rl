"""Configuration and constants for MountainCar environment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MountainCarConfig:
    """Tunable physics and rendering parameters for MountainCar."""

    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    goal_velocity: float = 0.0
    force: float = 0.001
    gravity: float = 0.0025
    max_steps: int = 200
    step_penalty: float = -1.0
    goal_reward: float = 0.0

    # rendering
    screen_width: int = 600
    screen_height: int = 400
    render_fps: int = 30

    bg_color: Tuple[int, int, int] = (245, 240, 230)
    mountain_color: Tuple[int, int, int] = (139, 137, 112)
    car_color: Tuple[int, int, int] = (200, 60, 60)
    flag_color: Tuple[int, int, int] = (103, 138, 99)
    track_color: Tuple[int, int, int] = (80, 80, 80)
    text_color: Tuple[int, int, int] = (80, 80, 80)


ACTION_LEFT = 0
ACTION_NOOP = 1
ACTION_RIGHT = 2

ACTION_LABELS: List[str] = ["push_left", "no_push", "push_right"]


def height(x: float) -> float:
    """Mountain profile: y = sin(3x) * 0.45 + 0.55."""
    return math.sin(3 * x) * 0.45 + 0.55
