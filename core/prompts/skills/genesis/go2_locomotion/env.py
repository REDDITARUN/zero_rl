"""Go2 Locomotion: quadruped walking via Genesis physics.

Massively parallel environment for training a Unitree Go2 robot
to walk using PPO. Demonstrates the Genesis env pattern:
- Plain class (no gym.Env inheritance)
- Batched tensor buffers (num_envs, ...)
- scene.build(n_envs=N) for GPU-parallel simulation
- Modular reward functions from rewards.py
- Selective reset via boolean mask

Reference: Genesis official examples/locomotion/go2_env.py
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

import genesis as gs
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)

from .rewards import REWARD_REGISTRY


def _gs_rand(lower: torch.Tensor, upper: torch.Tensor, batch_shape: tuple) -> torch.Tensor:
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class Go2Env:
    """Parallel locomotion environment for Unitree Go2 quadruped.

    Observation (45-dim):
        base_ang_vel(3), projected_gravity(3), commands(3),
        dof_pos(12), dof_vel(12), actions(12).

    Actions (12-dim):
        Joint position offsets from default pose, scaled by action_scale.

    Rewards:
        Weighted sum of modular reward functions (see rewards.py).

    Returns from step():
        (obs_buf, rew_buf, reset_buf, extras)
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: Dict[str, Any],
        obs_cfg: Dict[str, Any],
        reward_cfg: Dict[str, Any],
        command_cfg: Dict[str, Any],
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = dict(reward_cfg["reward_scales"])

        # ---- scene setup (production visuals) ----
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                lights=[
                    {"type": "directional", "dir": (-1, -0.8, -1.2), "color": (1.0, 0.98, 0.95), "intensity": 5.5},
                    {"type": "directional", "dir": (0.6, 0.3, -0.8), "color": (0.45, 0.55, 0.75), "intensity": 2.5},
                    {"type": "directional", "dir": (0.2, -0.6, -0.4), "color": (0.40, 0.38, 0.45), "intensity": 1.5},
                ],
                ambient_light=(0.25, 0.25, 0.28),
                background_color=(0.14, 0.16, 0.20),
                shadow=True,
                plane_reflection=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
            surface=gs.surfaces.Reflective(color=(0.82, 0.84, 0.86)),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
            surface=gs.surfaces.Plastic(color=(0.18, 0.20, 0.24), roughness=0.35),
        )
        self.scene.build(n_envs=num_envs)

        # ---- motor setup ----
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # ---- initial state ----
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][j.name] for j in self.robot.joints[1:]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # ---- buffers ----
        D = gs.device
        F = gs.tc_float
        self.base_lin_vel = torch.empty((num_envs, 3), dtype=F, device=D)
        self.base_ang_vel = torch.empty((num_envs, 3), dtype=F, device=D)
        self.projected_gravity = torch.empty((num_envs, 3), dtype=F, device=D)
        self.obs_buf = torch.empty((num_envs, self.num_obs), dtype=F, device=D)
        self.rew_buf = torch.empty((num_envs,), dtype=F, device=D)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=D)
        self.episode_length_buf = torch.empty((num_envs,), dtype=gs.tc_int, device=D)
        self.commands = torch.empty((num_envs, self.num_commands), dtype=F, device=D)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            dtype=F, device=D,
        )
        self.commands_limits = [
            torch.tensor(v, dtype=F, device=D) for v in zip(
                command_cfg["lin_vel_x_range"],
                command_cfg["lin_vel_y_range"],
                command_cfg["ang_vel_range"],
            )
        ]
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=F, device=D)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((num_envs, 3), dtype=F, device=D)
        self.base_quat = torch.empty((num_envs, 4), dtype=F, device=D)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=F, device=D,
        )

        self.extras: Dict[str, Any] = {"observations": {}}

        # ---- reward setup ----
        self.reward_functions: Dict[str, Any] = {}
        self.episode_sums: Dict[str, torch.Tensor] = {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = REWARD_REGISTRY[name]
            self.episode_sums[name] = torch.zeros((num_envs,), dtype=F, device=D)

    # ---- core interface -----------------------------------------------------

    def reset(self) -> Tuple[torch.Tensor, None]:
        self._reset_idx()
        self._update_observation()
        return self.obs_buf, None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        cfg = self.env_cfg
        self.actions = torch.clip(actions, -cfg["clip_actions"], cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target = exec_actions * cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()
        self._compute_rewards()
        self._resample_commands(self.episode_length_buf % int(cfg["resampling_time_s"] / self.dt) == 0)
        self._check_termination()
        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    # ---- internal -----------------------------------------------------------

    def _update_state(self) -> None:
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_q)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_q)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

    def _compute_rewards(self) -> None:
        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew = fn(self) * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def _check_termination(self) -> None:
        cfg = self.env_cfg
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > cfg["termination_if_roll_greater_than"]
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

    def _update_observation(self) -> None:
        self.obs_buf = torch.concatenate((
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions,
        ), dim=-1)

    def _resample_commands(self, envs_idx: torch.Tensor) -> None:
        cmds = _gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(cmds)
        else:
            torch.where(envs_idx[:, None], cmds, self.commands, out=self.commands)

    def _reset_idx(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel,
                        self.actions, self.last_actions, self.last_dof_vel):
                buf.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            for src, dst in [
                (self.init_base_pos, self.base_pos),
                (self.init_base_quat, self.base_quat),
                (self.init_projected_gravity, self.projected_gravity),
                (self.init_dof_pos, self.dof_pos),
            ]:
                torch.where(envs_idx[:, None], src, dst, out=dst)
            for buf in (self.base_lin_vel, self.base_ang_vel, self.dof_vel,
                        self.actions, self.last_actions, self.last_dof_vel):
                buf.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        self.extras["episode"] = {}
        if envs_idx is None:
            for key, val in self.episode_sums.items():
                self.extras["episode"]["rew_" + key] = val.mean() / self.env_cfg["episode_length_s"]
                val.zero_()
        else:
            n = envs_idx.sum()
            for key, val in self.episode_sums.items():
                mean = torch.where(n > 0, val[envs_idx].sum() / n, 0.0)
                self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
                val.masked_fill_(envs_idx, 0.0)

        self._resample_commands(envs_idx)
