#!/usr/bin/env python3
"""Play MountainCar with arrow keys.

Run:
    python -m skills_prompts.gym.mountain_car.interactive
"""

from __future__ import annotations

import pygame

from .config import ACTION_LEFT, ACTION_NOOP, ACTION_RIGHT, MountainCarConfig
from .env import MountainCarEnv

KEY_MAP = {
    pygame.K_LEFT: ACTION_LEFT,
    pygame.K_a: ACTION_LEFT,
    pygame.K_RIGHT: ACTION_RIGHT,
    pygame.K_d: ACTION_RIGHT,
}


def run() -> None:
    """Launch interactive MountainCar session."""
    cfg = MountainCarConfig(render_fps=30, max_steps=500)
    env = MountainCarEnv(render_mode="human", config=cfg)
    obs, info = env.reset()

    total_reward = 0.0
    print("MountainCar — LEFT/RIGHT to push, release for no-op, R reset, Q quit")
    print(f"  pos={obs[0]:.3f}  vel={obs[1]:.4f}")

    running = True
    while running:
        action = ACTION_NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print("\n--- RESET ---")
                    continue

        if not running:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = ACTION_LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = ACTION_RIGHT

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"  *** REACHED THE TOP! Steps: {info['steps']}  Total: {total_reward:.0f} ***")
            obs, _ = env.reset()
            total_reward = 0.0
            print(f"  New episode — pos={obs[0]:.3f}")
        elif truncated:
            print(f"  *** TRUNCATED at {info['steps']} steps. Total: {total_reward:.0f} ***")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    run()
