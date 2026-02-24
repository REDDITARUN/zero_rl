"""Modular reward functions for Go2 locomotion.

Each function takes the env instance and returns a (num_envs,)
tensor. Reward functions are registered by name in env.__init__
and summed with configurable weights per step.

Pattern:
    def _reward_<name>(env) -> Tensor:
        ...
    Registered in config as reward_scales = {"<name>": weight}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env import Go2Env


def reward_tracking_lin_vel(env: Go2Env) -> torch.Tensor:
    """Track commanded xy linear velocity. Exponential kernel."""
    error = torch.sum(
        torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1
    )
    return torch.exp(-error / env.reward_cfg["tracking_sigma"])


def reward_tracking_ang_vel(env: Go2Env) -> torch.Tensor:
    """Track commanded yaw angular velocity."""
    error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
    return torch.exp(-error / env.reward_cfg["tracking_sigma"])


def reward_lin_vel_z(env: Go2Env) -> torch.Tensor:
    """Penalize vertical base velocity (keep body level)."""
    return torch.square(env.base_lin_vel[:, 2])


def reward_action_rate(env: Go2Env) -> torch.Tensor:
    """Penalize rapid changes in actions (smooth control)."""
    return torch.sum(torch.square(env.last_actions - env.actions), dim=1)


def reward_similar_to_default(env: Go2Env) -> torch.Tensor:
    """Penalize deviation from default standing pose."""
    return torch.sum(torch.abs(env.dof_pos - env.default_dof_pos), dim=1)


def reward_base_height(env: Go2Env) -> torch.Tensor:
    """Penalize base height away from target."""
    return torch.square(
        env.base_pos[:, 2] - env.reward_cfg["base_height_target"]
    )


REWARD_REGISTRY = {
    "tracking_lin_vel": reward_tracking_lin_vel,
    "tracking_ang_vel": reward_tracking_ang_vel,
    "lin_vel_z": reward_lin_vel_z,
    "action_rate": reward_action_rate,
    "similar_to_default": reward_similar_to_default,
    "base_height": reward_base_height,
}
