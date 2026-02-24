#!/usr/bin/env python3
"""Play CartPole with left/right arrow keys (held).

CartPole is a real-time physics sim — it steps every frame.
Hold LEFT/RIGHT to push the cart. Release to let it coast
(defaults to a small random nudge so it's not perfectly stable).

Run:
    python -m skills_prompts.gym.cart_pole.interactive
"""

from __future__ import annotations

import pygame

from .config import ACTION_LEFT, ACTION_RIGHT, CartPoleConfig
from .env import CartPoleEnv


def run() -> None:
    """Launch interactive CartPole session."""
    cfg = CartPoleConfig(render_fps=30, max_steps=500)
    env = CartPoleEnv(render_mode="human", config=cfg)
    obs, info = env.reset()

    total_reward = 0.0
    clock = pygame.time.Clock()
    print("CartPole — HOLD LEFT/RIGHT arrows to push cart, R to reset, Q to quit")
    print("  Try to keep the pole upright as long as possible!")
    print(f"  x={obs[0]:.3f}  θ={obs[2]:.3f}")

    running = True
    while running:
        clock.tick(cfg.render_fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
                    print(f"\n--- RESET --- x={obs[0]:.3f}  θ={obs[2]:.3f}")

        if not running:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = ACTION_LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = ACTION_RIGHT
        else:
            action = env.np_random.choice([ACTION_LEFT, ACTION_RIGHT])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(
                f"  *** FELL! Steps survived: {info['steps']}  "
                f"Total: {total_reward:.0f} ***"
            )
            obs, _ = env.reset()
            total_reward = 0.0
            print(f"  New episode — x={obs[0]:.3f}  θ={obs[2]:.3f}")
        elif truncated:
            print(f"  *** MAX STEPS REACHED! Total: {total_reward:.0f} ***")
            obs, _ = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    run()
