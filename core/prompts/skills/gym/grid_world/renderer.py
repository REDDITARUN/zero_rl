"""Pygame renderer for GridWorld — separated from env logic for modularity."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import GridWorldConfig


class GridWorldRenderer:
    """Draws the grid, agent (circle), and target (diamond) using Pygame."""

    def __init__(self, cfg: GridWorldConfig, mode: str) -> None:
        import pygame

        self.cfg = cfg
        self.mode = mode
        self.cell = cfg.cell_pixels
        self.win_size = cfg.size * self.cell

        pygame.init()
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("GridWorld")
            self.surface = pygame.display.set_mode((self.win_size, self.win_size))
        else:
            self.surface = pygame.Surface((self.win_size, self.win_size))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self._pygame = pygame

    def draw(
        self,
        agent: np.ndarray,
        target: np.ndarray,
        step: int,
    ) -> Optional[np.ndarray]:
        pg = self._pygame
        self.surface.fill(self.cfg.bg_color)

        for r in range(self.cfg.size):
            for c in range(self.cfg.size):
                rect = (c * self.cell, r * self.cell, self.cell, self.cell)
                pg.draw.rect(self.surface, self.cfg.grid_color, rect, 1)

        tx, ty = int(target[1]) * self.cell, int(target[0]) * self.cell
        cx_t, cy_t = tx + self.cell // 2, ty + self.cell // 2
        half = self.cell // 3
        diamond = [
            (cx_t, cy_t - half),
            (cx_t + half, cy_t),
            (cx_t, cy_t + half),
            (cx_t - half, cy_t),
        ]
        pg.draw.polygon(self.surface, self.cfg.target_color, diamond)

        ax, ay = int(agent[1]) * self.cell, int(agent[0]) * self.cell
        cx_a, cy_a = ax + self.cell // 2, ay + self.cell // 2
        pg.draw.circle(self.surface, self.cfg.agent_color, (cx_a, cy_a), self.cell // 3)

        label = self.font.render(f"step {step}", True, self.cfg.text_color)
        self.surface.blit(label, (4, 2))

        if self.mode == "human":
            pg.event.pump()
            pg.display.update()
            self.clock.tick(self.cfg.render_fps)
            return None

        return np.transpose(
            np.array(pg.surfarray.pixels3d(self.surface)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        self._pygame.display.quit()
        self._pygame.quit()
