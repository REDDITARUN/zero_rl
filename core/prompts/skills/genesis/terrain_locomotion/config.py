"""Configuration for terrain locomotion environment.

Demonstrates an env with procedural terrain + custom mesh obstacles.
"""


def get_env_cfg() -> dict:
    """Environment configuration."""
    return {
        "num_actions": 12,
        "episode_length_s": 20.0,
        "dt": 0.02,
        "base_init_pos": [0.0, 0.0, 0.6],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "action_scale": 0.25,
        "kp": 50.0,
        "kd": 1.0,
        "termination_contact_link_names": ["base"],
        "joint_names": [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
        },
        # Mesh obstacles loaded from assets/ (generated via cad_generate)
        "obstacle_meshes": [
            {"file": "assets/rock_large/rock_large.stl", "pos": (4.0, 2.0, 0.0), "scale": 0.001},
            {"file": "assets/rock_large/rock_large.stl", "pos": (-3.0, 5.0, 0.0), "scale": 0.0008},
            {"file": "assets/pine_tree/pine_tree.stl", "pos": (6.0, -1.0, 0.0), "scale": 0.001},
        ],
    }


def get_obs_cfg() -> dict:
    """Observation space configuration."""
    return {
        "num_obs": 48,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }


def get_reward_cfg() -> dict:
    """Reward weights and parameters."""
    return {
        "tracking_sigma": 0.25,
        "base_height_target": 0.34,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -30.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "alive": 0.5,
            "terrain_progress": 2.0,
        },
    }


def get_command_cfg() -> dict:
    """Velocity command ranges."""
    return {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 1.5],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }
