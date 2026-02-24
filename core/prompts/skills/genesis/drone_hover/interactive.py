#!/usr/bin/env python3
"""Visualize drone hovering with viewer and random actions.

Demonstrates:
  1. Creating the hover env with obstacles and show_viewer=True
  2. Running a rollout with random RPM perturbations
  3. Target sphere shows the goal; boxes are obstacles

Run:
    python -m skills_prompts.genesis.drone_hover.interactive
"""

from __future__ import annotations

import argparse

import torch

import genesis as gs

from .config import get_command_cfg, get_env_cfg, get_obs_cfg, get_reward_cfg
from .env import DroneHoverEnv


def run(num_envs: int = 1, steps: int = 500) -> None:
    """Launch drone hovering viewer with random RPM perturbations."""
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    env_cfg = get_env_cfg()
    obs_cfg = get_obs_cfg()
    reward_cfg = get_reward_cfg()
    command_cfg = get_command_cfg()

    env = DroneHoverEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    obs, _ = env.reset()
    total_reward = torch.zeros(num_envs, device=gs.device)

    print(f"Drone Hover — {num_envs} env(s), {steps} steps")
    print("  3D viewer opened. Close window or Ctrl+C to stop.")

    for step in range(steps):
        actions = torch.randn(num_envs, env_cfg["num_actions"], device=gs.device) * 0.15
        obs, rew, done, extras = env.step(actions)
        total_reward += rew

        if step % 100 == 0:
            dist = env.target_dist[0].item()
            z = env.pos[0, 2].item()
            mean_rew = total_reward.mean().item()
            print(
                f"  step {step:>4}  z={z:.2f}  "
                f"target_dist={dist:.3f}  mean_reward={mean_rew:+.2f}"
            )

    print(f"\nDone. Final mean reward: {total_reward.mean().item():+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    run(num_envs=args.num_envs, steps=args.steps)
