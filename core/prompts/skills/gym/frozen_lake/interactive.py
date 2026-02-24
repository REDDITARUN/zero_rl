#!/usr/bin/env python3
"""Play FrozenLake with arrow keys.

Run:
    python -m skills_prompts.gym.frozen_lake.interactive
    python -m skills_prompts.gym.frozen_lake.interactive --map 8x8 --no-slip
"""

from __future__ import annotations

import argparse
import sys

import pygame

from .config import DOWN, LEFT, RIGHT, UP, FrozenLakeConfig
from .env import FrozenLakeEnv

KEY_MAP = {
    pygame.K_LEFT: LEFT,
    pygame.K_DOWN: DOWN,
    pygame.K_RIGHT: RIGHT,
    pygame.K_UP: UP,
    pygame.K_a: LEFT,
    pygame.K_s: DOWN,
    pygame.K_d: RIGHT,
    pygame.K_w: UP,
}


def run(map_name: str = "4x4", slippery: bool = True) -> None:
    """Launch interactive FrozenLake session."""
    cfg = FrozenLakeConfig(
        map_name=map_name,
        is_slippery=slippery,
        render_fps=6,
        cell_pixels=96,
    )
    env = FrozenLakeEnv(render_mode="human", config=cfg)
    obs, info = env.reset()

    clock = pygame.time.Clock()
    slip_str = "slippery" if slippery else "deterministic"
    print(f"FrozenLake {map_name} ({slip_str}) — arrows/WASD to move, R reset, Q quit")
    total_reward = 0.0
    desc = env.desc
    row, col = obs // env.ncol, obs % env.ncol
    tile = desc[row, col].decode()
    print(f"  Pos: ({row},{col}) [{tile}]")

    running = True
    while running:
        clock.tick(30)

        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    row, col = obs // env.ncol, obs % env.ncol
                    tile = desc[row, col].decode()
                    print(f"\n--- RESET --- Pos: ({row},{col}) [{tile}]")
                elif event.key in KEY_MAP:
                    action = KEY_MAP[event.key]

        if not running:
            break

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            row, col = obs // env.ncol, obs % env.ncol
            tile = desc[row, col].decode()
            act_name = env.action_labels[action]
            print(
                f"  [{act_name:>5}] → ({row},{col}) [{tile}]  "
                f"R: {reward:+.1f}  Total: {total_reward:+.1f}  "
                f"prob: {info['prob']:.2f}"
            )
            if terminated:
                outcome = "GOAL!" if tile == "G" else "FELL IN HOLE!"
                print(f"  *** {outcome} Total: {total_reward:+.1f} ***")
                obs, info = env.reset()
                total_reward = 0.0
                row, col = obs // env.ncol, obs % env.ncol
                print(f"  New episode — Pos: ({row},{col})")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="4x4", choices=["4x4", "8x8"])
    parser.add_argument("--no-slip", action="store_true")
    args = parser.parse_args()
    run(map_name=args.map, slippery=not args.no_slip)
