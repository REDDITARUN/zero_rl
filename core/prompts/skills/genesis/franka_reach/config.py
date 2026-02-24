"""Configuration for Franka Panda reaching environment.

Defines joint layout, PD gains, observation scales, reward weights,
and target workspace bounds for a 7-DOF reaching task.
"""

from __future__ import annotations

from typing import Any, Dict


def get_env_cfg() -> Dict[str, Any]:
    """Environment configuration: robot setup, control, and termination."""
    return {
        "num_actions": 7,
        "episode_length_s": 8.0,
        "simulate_action_latency": False,
        "action_scale": 0.1,
        "clip_actions": 1.0,
        "kp": 400.0,
        "kd": 40.0,
        "base_init_pos": (0.0, 0.0, 0.0),
        "ee_link_name": "panda_link7",
        "joint_names": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        "default_joint_angles": {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
        },
        "success_threshold": 0.02,
        "termination_if_ee_below": 0.05,
    }


def get_obs_cfg() -> Dict[str, Any]:
    """Observation space: joint state + EE position + target position."""
    return {
        # joint_pos(7) + joint_vel(7) + ee_pos(3) + target_pos(3) + actions(7)
        "num_obs": 27,
        "obs_scales": {
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }


def get_reward_cfg() -> Dict[str, Any]:
    """Reward weights — distance dominates, with regularization terms."""
    return {
        "reward_scales": {
            "reach_distance": -20.0,
            "reach_success": 10.0,
            "action_rate": -0.01,
            "action_magnitude": -0.005,
            "joint_vel": -0.001,
        },
    }


def get_command_cfg() -> Dict[str, Any]:
    """Target position sampling bounds (workspace of the arm)."""
    return {
        "num_commands": 3,
        "target_x_range": (0.3, 0.7),
        "target_y_range": (-0.3, 0.3),
        "target_z_range": (0.2, 0.8),
    }
