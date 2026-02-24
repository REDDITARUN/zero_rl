"""Configuration and constants for FrozenLake environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium.utils import seeding


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTION_LABELS: List[str] = ["left", "down", "right", "up"]

MAPS: Dict[str, List[str]] = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


@dataclass
class FrozenLakeConfig:
    """Tunable parameters for FrozenLake."""

    map_name: Optional[str] = "4x4"
    custom_map: Optional[List[str]] = None
    is_slippery: bool = True
    success_rate: float = 1.0 / 3.0
    goal_reward: float = 1.0
    hole_reward: float = 0.0
    step_reward: float = 0.0
    render_fps: int = 4
    cell_pixels: int = 96

    bg_color: Tuple[int, int, int] = (200, 220, 240)
    ice_color: Tuple[int, int, int] = (180, 210, 240)
    hole_color: Tuple[int, int, int] = (40, 40, 60)
    start_color: Tuple[int, int, int] = (180, 200, 160)
    goal_color: Tuple[int, int, int] = (103, 138, 99)
    agent_color: Tuple[int, int, int] = (200, 60, 60)
    grid_color: Tuple[int, int, int] = (160, 180, 200)
    text_color: Tuple[int, int, int] = (80, 80, 80)

    def get_map(self) -> List[str]:
        if self.custom_map is not None:
            return self.custom_map
        if self.map_name is not None and self.map_name in MAPS:
            return MAPS[self.map_name]
        return generate_random_map()


def is_valid(board: List[List[str]], max_size: int) -> bool:
    """DFS check that a path exists from S to G."""
    frontier, discovered = [(0, 0)], set()
    while frontier:
        r, c = frontier.pop()
        if (r, c) in discovered:
            continue
        discovered.add((r, c))
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < max_size and 0 <= nc < max_size:
                if board[nr][nc] == "G":
                    return True
                if board[nr][nc] != "H":
                    frontier.append((nr, nc))
    return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generate a random map guaranteed to have a solvable path."""
    np_random, _ = seeding.np_random(seed)
    valid = False
    board: np.ndarray = np.empty((size, size), dtype="U1")
    while not valid:
        board = np_random.choice(["F", "H"], (size, size), p=[min(1, p), 1 - min(1, p)])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board.tolist(), size)
    return ["".join(row) for row in board]
