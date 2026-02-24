"""Configuration for Crazyflie drone hovering environment.

Defines drone physics, control parameters, workspace bounds,
obstacle layout, observation scales, and reward weights for
a target-tracking hover task with box obstacles.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def get_env_cfg() -> Dict[str, Any]:
    """Environment configuration: drone setup, control, and termination."""
    mass = 0.027
    g = 9.81
    kf = 3.16e-10
    hover_rpm = math.sqrt(mass * g / (4 * kf))

    return {
        "num_actions": 4,
        "episode_length_s": 10.0,
        "dt": 0.01,
        "drone_urdf": "urdf/drones/cf2x.urdf",
        "drone_init_pos": (0.0, 0.0, 1.0),
        "mass": mass,
        "g": g,
        "kf": kf,
        "hover_rpm": hover_rpm,
        "rpm_scale": 5000.0,
        "clip_actions": 1.0,
        "termination_if_pitch_gt": 60.0,
        "termination_if_roll_gt": 60.0,
        "termination_if_z_below": 0.05,
        "termination_if_z_above": 3.0,
        "boundary_xy": 2.0,
        "obstacles": _default_obstacles(),
    }


def _default_obstacles() -> List[Dict[str, Any]]:
    """Box obstacles scattered in the flight volume."""
    return [
        {"size": (0.3, 0.3, 0.8), "pos": (0.8, 0.5, 0.4)},
        {"size": (0.2, 0.6, 0.2), "pos": (-0.6, -0.4, 0.5)},
        {"size": (0.5, 0.2, 0.3), "pos": (0.3, -0.7, 0.7)},
    ]


def get_obs_cfg() -> Dict[str, Any]:
    """Observation space: pose + velocities + target + actions.

    pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + target(3) + actions(4) = 20
    """
    return {
        "num_obs": 20,
        "obs_scales": {
            "pos": 1.0,
            "quat": 1.0,
            "lin_vel": 0.2,
            "ang_vel": 0.25,
        },
    }


def get_reward_cfg() -> Dict[str, Any]:
    """Reward weights — target tracking dominates, with stability terms."""
    return {
        "reward_scales": {
            "target_distance": -5.0,
            "target_reached": 10.0,
            "angular_velocity": -0.1,
            "action_smoothness": -0.02,
            "uprightness": 1.0,
            "crash": -50.0,
        },
    }


def get_command_cfg() -> Dict[str, Any]:
    """Target position sampling bounds."""
    return {
        "num_commands": 3,
        "target_x_range": (-1.0, 1.0),
        "target_y_range": (-1.0, 1.0),
        "target_z_range": (0.5, 2.0),
        "success_radius": 0.15,
        "resample_time_s": 5.0,
    }
