"""Pygame renderer for MountainCar — draws mountain profile, car, and goal flag."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import MountainCarConfig, height


class MountainCarRenderer:
    """Renders mountain curve, car (circle on slope), and goal flag."""

    def __init__(self, cfg: MountainCarConfig, mode: str) -> None:
        import pygame

        self.cfg = cfg
        self.mode = mode
        self.w = cfg.screen_width
        self.h = cfg.screen_height

        pygame.init()
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("MountainCar")
            self.surface = pygame.display.set_mode((self.w, self.h))
        else:
            self.surface = pygame.Surface((self.w, self.h))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self._pygame = pygame

        self._mountain_points = self._build_mountain()

    def _build_mountain(self) -> list:
        """Pre-compute the mountain polygon points."""
        cfg = self.cfg
        xs = np.linspace(cfg.min_position, cfg.max_position, 100)
        points = []
        for x in xs:
            sx = self._scale_x(x)
            sy = self._scale_y(height(x))
            points.append((int(sx), int(sy)))
        points.append((self.w, self.h))
        points.append((0, self.h))
        return points

    def _scale_x(self, x: float) -> float:
        cfg = self.cfg
        return (x - cfg.min_position) / (cfg.max_position - cfg.min_position) * self.w

    def _scale_y(self, y: float) -> float:
        return self.h - y * self.h * 0.8 - 20

    def draw(
        self,
        state: Optional[np.ndarray],
        step: int,
    ) -> Optional[np.ndarray]:
        if state is None:
            return None

        pg = self._pygame
        cfg = self.cfg
        self.surface.fill(cfg.bg_color)

        pg.draw.polygon(self.surface, cfg.mountain_color, self._mountain_points)

        position = float(state[0])
        velocity = float(state[1])

        car_x = self._scale_x(position)
        car_y = self._scale_y(height(position))
        car_radius = 10
        pg.draw.circle(
            self.surface, cfg.car_color, (int(car_x), int(car_y) - car_radius), car_radius
        )

        flag_x = self._scale_x(cfg.goal_position)
        flag_y = self._scale_y(height(cfg.goal_position))
        pg.draw.line(
            self.surface,
            cfg.track_color,
            (int(flag_x), int(flag_y)),
            (int(flag_x), int(flag_y) - 40),
            2,
        )
        flag_points = [
            (int(flag_x), int(flag_y) - 40),
            (int(flag_x) + 20, int(flag_y) - 32),
            (int(flag_x), int(flag_y) - 24),
        ]
        pg.draw.polygon(self.surface, cfg.flag_color, flag_points)

        info_text = f"step {step}  pos={position:.3f}  vel={velocity:.4f}"
        label = self.font.render(info_text, True, cfg.text_color)
        self.surface.blit(label, (4, 2))

        if self.mode == "human":
            pg.event.pump()
            pg.display.update()
            self.clock.tick(cfg.render_fps)
            return None

        return np.transpose(
            np.array(pg.surfarray.pixels3d(self.surface)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        self._pygame.display.quit()
        self._pygame.quit()
