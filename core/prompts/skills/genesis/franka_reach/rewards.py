"""Modular reward functions for Franka reaching task.

Each function takes the env instance and returns a per-env reward tensor.
Distance-based reaching reward dominates; regularization terms keep
actions smooth and joint velocities low.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

import torch

if TYPE_CHECKING:
    from .env import FrankaReachEnv


def reward_reach_distance(env: FrankaReachEnv) -> torch.Tensor:
    """L2 distance from end-effector to target (negated by scale in config)."""
    return env.ee_to_target_dist


def reward_reach_success(env: FrankaReachEnv) -> torch.Tensor:
    """Binary bonus when EE is within success_threshold of target."""
    return (env.ee_to_target_dist < env.env_cfg["success_threshold"]).float()


def reward_action_rate(env: FrankaReachEnv) -> torch.Tensor:
    """Penalise change in actions between timesteps."""
    return torch.sum(torch.square(env.actions - env.last_actions), dim=-1)


def reward_action_magnitude(env: FrankaReachEnv) -> torch.Tensor:
    """Penalise large actions."""
    return torch.sum(torch.square(env.actions), dim=-1)


def reward_joint_vel(env: FrankaReachEnv) -> torch.Tensor:
    """Penalise high joint velocities."""
    return torch.sum(torch.square(env.dof_vel), dim=-1)


REWARD_REGISTRY: Dict[str, Callable] = {
    "reach_distance": reward_reach_distance,
    "reach_success": reward_reach_success,
    "action_rate": reward_action_rate,
    "action_magnitude": reward_action_magnitude,
    "joint_vel": reward_joint_vel,
}
