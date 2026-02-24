"""Reward functions for terrain locomotion environment."""

import torch


def reward_tracking_lin_vel(env) -> torch.Tensor:
    """Reward for following commanded XY velocity."""
    error = torch.sum(
        torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1
    )
    return torch.exp(-error / env.reward_cfg["tracking_sigma"])


def reward_tracking_ang_vel(env) -> torch.Tensor:
    """Reward for following commanded yaw rate."""
    error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
    return torch.exp(-error / env.reward_cfg["tracking_sigma"])


def reward_lin_vel_z(env) -> torch.Tensor:
    """Penalize vertical bouncing."""
    return torch.square(env.base_lin_vel[:, 2])


def reward_base_height(env) -> torch.Tensor:
    """Penalize deviation from target base height."""
    target = env.reward_cfg["base_height_target"]
    return torch.square(env.base_pos[:, 2] - target)


def reward_action_rate(env) -> torch.Tensor:
    """Penalize jerky control (change in actions between steps)."""
    return torch.sum(torch.square(env.actions - env.last_actions), dim=1)


def reward_similar_to_default(env) -> torch.Tensor:
    """Penalize deviation from default standing pose."""
    dof_pos = env.robot.get_dofs_position(env.motors_dof_idx)
    return torch.sum(torch.abs(dof_pos - env.init_dof_pos), dim=1)


def reward_alive(env) -> torch.Tensor:
    """Constant reward for staying upright (not terminated)."""
    return torch.ones(env.num_envs, device=env.device)


def reward_terrain_progress(env) -> torch.Tensor:
    """Reward forward distance traveled over terrain."""
    return env.base_lin_vel[:, 0].clamp(min=0.0)


REWARD_REGISTRY = {
    "tracking_lin_vel": reward_tracking_lin_vel,
    "tracking_ang_vel": reward_tracking_ang_vel,
    "lin_vel_z": reward_lin_vel_z,
    "base_height": reward_base_height,
    "action_rate": reward_action_rate,
    "similar_to_default": reward_similar_to_default,
    "alive": reward_alive,
    "terrain_progress": reward_terrain_progress,
}
