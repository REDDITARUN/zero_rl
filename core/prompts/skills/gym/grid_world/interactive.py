#!/usr/bin/env python3
"""Play GridWorld with arrow keys.

Run:
    python -m skills_prompts.gym.grid_world.interactive
"""

from __future__ import annotations

import sys

import pygame

from .config import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, GridWorldConfig
from .env import GridWorldEnv

KEY_MAP = {
    pygame.K_RIGHT: ACTION_RIGHT,
    pygame.K_UP: ACTION_UP,
    pygame.K_LEFT: ACTION_LEFT,
    pygame.K_DOWN: ACTION_DOWN,
    pygame.K_d: ACTION_RIGHT,
    pygame.K_w: ACTION_UP,
    pygame.K_a: ACTION_LEFT,
    pygame.K_s: ACTION_DOWN,
}


def run(size: int = 5) -> None:
    """Launch interactive GridWorld session."""
    cfg = GridWorldConfig(size=size, render_fps=10, cell_pixels=96)
    env = GridWorldEnv(render_mode="human", config=cfg)
    obs, info = env.reset()

    total_reward = 0.0
    clock = pygame.time.Clock()
    print(f"GridWorld {size}x{size} — arrow keys / WASD to move, R to reset, Q to quit")
    print(f"  Agent: {obs['agent']}  Target: {obs['target']}  Dist: {info['distance']}")

    running = True
    while running:
        clock.tick(30)

        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("\n--- RESET ---")
                    print(f"  Agent: {obs['agent']}  Target: {obs['target']}")
                elif event.key in KEY_MAP:
                    action = KEY_MAP[event.key]

        if not running:
            break

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            act_name = env.action_labels[action]
            print(
                f"  [{act_name:>5}] → Agent: {obs['agent']}  "
                f"Reward: {reward:+.2f}  Total: {total_reward:+.2f}  "
                f"Dist: {info['distance']}"
            )
            if terminated:
                print(f"  *** GOAL REACHED! Total reward: {total_reward:+.2f} ***")
                obs, info = env.reset()
                total_reward = 0.0
                print(f"  New episode — Agent: {obs['agent']}  Target: {obs['target']}")
            elif truncated:
                print(f"  *** TRUNCATED (max steps). Total: {total_reward:+.2f} ***")
                obs, info = env.reset()
                total_reward = 0.0

    env.close()


if __name__ == "__main__":
    sz = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run(size=sz)
