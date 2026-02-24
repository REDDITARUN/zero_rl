"""Modular reward functions for drone hovering task.

Each function takes the env instance and returns a per-env reward tensor.
Rewards cover target tracking, flight stability, action smoothness,
and crash penalties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

import torch

if TYPE_CHECKING:
    from .env import DroneHoverEnv


def reward_target_distance(env: DroneHoverEnv) -> torch.Tensor:
    """L2 distance from drone to target position."""
    return env.target_dist


def reward_target_reached(env: DroneHoverEnv) -> torch.Tensor:
    """Binary bonus when drone is within success radius of target."""
    return (env.target_dist < env.command_cfg["success_radius"]).float()


def reward_angular_velocity(env: DroneHoverEnv) -> torch.Tensor:
    """Penalise high angular velocities for stable flight."""
    return torch.sum(torch.square(env.ang_vel), dim=-1)


def reward_action_smoothness(env: DroneHoverEnv) -> torch.Tensor:
    """Penalise change in actions between timesteps."""
    return torch.sum(torch.square(env.actions - env.last_actions), dim=-1)


def reward_uprightness(env: DroneHoverEnv) -> torch.Tensor:
    """Reward the drone for staying upright (z-component of up vector).

    When perfectly upright, the projected gravity z-component is -1.
    We reward 1 + projected_gravity_z, which is 0 when upright and
    negative when tilted.
    """
    return 1.0 + env.projected_gravity[:, 2]


def reward_crash(env: DroneHoverEnv) -> torch.Tensor:
    """Large penalty when a termination condition is hit."""
    return env.crash_buf.float()


REWARD_REGISTRY: Dict[str, Callable] = {
    "target_distance": reward_target_distance,
    "target_reached": reward_target_reached,
    "angular_velocity": reward_angular_velocity,
    "action_smoothness": reward_action_smoothness,
    "uprightness": reward_uprightness,
    "crash": reward_crash,
}
