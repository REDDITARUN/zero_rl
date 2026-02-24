"""Pygame renderer for CartPole — draws cart, pole, track, and info."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import CartPoleConfig


class CartPoleRenderer:
    """Renders cart (rectangle), pole (rotated line), and track."""

    def __init__(self, cfg: CartPoleConfig, mode: str) -> None:
        import pygame

        self.cfg = cfg
        self.mode = mode
        self.w = cfg.screen_width
        self.h = cfg.screen_height

        pygame.init()
        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("CartPole")
            self.surface = pygame.display.set_mode((self.w, self.h))
        else:
            self.surface = pygame.Surface((self.w, self.h))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self._pygame = pygame

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

        world_width = cfg.x_threshold * 2
        scale = self.w / world_width
        cart_y = self.h * 0.6

        x, _, theta, _ = state

        # track
        pg.draw.line(
            self.surface,
            cfg.track_color,
            (0, int(cart_y + cfg.cart_height / 2)),
            (self.w, int(cart_y + cfg.cart_height / 2)),
            2,
        )

        # cart
        cart_x = x * scale + self.w / 2.0
        cart_rect = pg.Rect(
            int(cart_x - cfg.cart_width / 2),
            int(cart_y - cfg.cart_height / 2),
            int(cfg.cart_width),
            int(cfg.cart_height),
        )
        pg.draw.rect(self.surface, cfg.cart_color, cart_rect)

        # pole
        pole_len_px = scale * (2 * cfg.pole_length)
        pole_end_x = cart_x + pole_len_px * math.sin(theta)
        pole_end_y = cart_y - pole_len_px * math.cos(theta)
        pg.draw.line(
            self.surface,
            cfg.pole_color,
            (int(cart_x), int(cart_y)),
            (int(pole_end_x), int(pole_end_y)),
            int(cfg.pole_width),
        )

        # axle
        pg.draw.circle(self.surface, cfg.axle_color, (int(cart_x), int(cart_y)), 5)

        # info
        angle_deg = math.degrees(theta)
        info_text = f"step {step}  x={x:.2f}  θ={angle_deg:.1f}°"
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
