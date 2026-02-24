"""Configuration and constants for CartPole environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import math


@dataclass
class CartPoleConfig:
    """Tunable physics and rendering parameters for CartPole."""

    gravity: float = 9.8
    mass_cart: float = 1.0
    mass_pole: float = 0.1
    pole_length: float = 0.5  # half-length
    force_mag: float = 10.0
    tau: float = 0.02  # time step (seconds)

    # termination thresholds
    x_threshold: float = 2.4
    theta_threshold_rad: float = 12 * 2 * math.pi / 360  # ~0.2094 rad

    max_steps: int = 500
    reward_alive: float = 1.0

    # rendering
    screen_width: int = 600
    screen_height: int = 400
    render_fps: int = 50
    cart_width: float = 50.0
    cart_height: float = 30.0
    pole_width: float = 6.0

    bg_color: Tuple[int, int, int] = (245, 240, 230)
    track_color: Tuple[int, int, int] = (80, 80, 80)
    cart_color: Tuple[int, int, int] = (65, 92, 87)
    pole_color: Tuple[int, int, int] = (180, 120, 60)
    axle_color: Tuple[int, int, int] = (90, 90, 90)
    text_color: Tuple[int, int, int] = (80, 80, 80)

    @property
    def total_mass(self) -> float:
        return self.mass_cart + self.mass_pole

    @property
    def pole_mass_length(self) -> float:
        return self.mass_pole * self.pole_length


ACTION_LEFT = 0
ACTION_RIGHT = 1

ACTION_LABELS: List[str] = ["push_left", "push_right"]
