"""Pygame renderer for FrozenLake — shape-based, no image assets needed."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import ACTION_LABELS, FrozenLakeConfig


class FrozenLakeRenderer:
    """Draws the lake grid with colored tiles, agent circle, and info overlay."""

    def __init__(
        self,
        cfg: FrozenLakeConfig,
        desc: np.ndarray,
        mode: str,
    ) -> None:
        import pygame

        self.cfg = cfg
        self.desc = desc
        self.mode = mode
        self.nrow, self.ncol = desc.shape
        self.cell = cfg.cell_pixels
        self.win_w = self.ncol * self.cell
        self.win_h = self.nrow * self.cell

        pygame.init()
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Frozen Lake")
            self.surface = pygame.display.set_mode((self.win_w, self.win_h))
        else:
            self.surface = pygame.Surface((self.win_w, self.win_h))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", max(12, self.cell // 6))
        self._pygame = pygame

    def draw(self, state: int, last_action: Optional[int]) -> Optional[np.ndarray]:
        pg = self._pygame
        self.surface.fill(self.cfg.bg_color)

        agent_row, agent_col = state // self.ncol, state % self.ncol

        tile_colors = {
            b"F": self.cfg.ice_color,
            b"S": self.cfg.start_color,
            b"H": self.cfg.hole_color,
            b"G": self.cfg.goal_color,
        }
        tile_labels = {b"S": "S", b"H": "H", b"G": "G"}

        for r in range(self.nrow):
            for c in range(self.ncol):
                x, y = c * self.cell, r * self.cell
                letter = bytes(self.desc[r, c])
                color = tile_colors.get(letter, self.cfg.ice_color)
                pg.draw.rect(self.surface, color, (x, y, self.cell, self.cell))
                pg.draw.rect(self.surface, self.cfg.grid_color, (x, y, self.cell, self.cell), 1)

                label = tile_labels.get(letter)
                if label:
                    txt = self.font.render(label, True, (255, 255, 255))
                    self.surface.blit(
                        txt,
                        (x + self.cell // 2 - txt.get_width() // 2,
                         y + self.cell // 2 - txt.get_height() // 2),
                    )

        ax = agent_col * self.cell + self.cell // 2
        ay = agent_row * self.cell + self.cell // 2
        pg.draw.circle(self.surface, self.cfg.agent_color, (ax, ay), self.cell // 3)

        if last_action is not None:
            info = self.font.render(ACTION_LABELS[last_action], True, self.cfg.text_color)
            self.surface.blit(info, (4, 2))

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
