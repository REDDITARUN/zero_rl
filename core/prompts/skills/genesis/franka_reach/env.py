"""Franka Reach: 7-DOF arm reaching a target position via Genesis physics.

Parallel environment for training a Franka Panda arm to reach random
target positions in its workspace. Demonstrates the Genesis pattern
for manipulation tasks:
- Fixed-base articulated robot (URDF)
- End-effector tracking via forward kinematics
- Distance-based reward with success bonus
- Target resampling on episode reset

Contrasts with locomotion envs:
- Fixed base (no base velocity/orientation tracking)
- Observation includes EE + target Cartesian positions
- Reward is distance-based rather than velocity-tracking
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

import genesis as gs

from .rewards import REWARD_REGISTRY


class FrankaReachEnv:
    """Parallel reaching environment for Franka Panda 7-DOF arm.

    Observation (27-dim):
        dof_pos(7), dof_vel(7), ee_pos(3), target_pos(3), actions(7).

    Actions (7-dim):
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
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

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
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, -1.0, 1.5),
                camera_lookat=(0.4, 0.0, 0.4),
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
            gs.morphs.Plane(),
            surface=gs.surfaces.Reflective(color=(0.82, 0.84, 0.86)),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/panda_bullet/panda.urdf",
                pos=env_cfg["base_init_pos"],
                fixed=True,
            ),
            surface=gs.surfaces.Aluminium(),
        )
        self.target_vis = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.02,
                pos=(0.5, 0.0, 0.5),
                fixed=True,
                collision=False,
                visualization=True,
            ),
            surface=gs.surfaces.Emission(color=(0.15, 0.85, 0.45)),
        )
        self.scene.build(n_envs=num_envs)

        # ---- motor setup ----
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        self.ee_link = self.robot.get_link(env_cfg["ee_link_name"])

        # ---- initial state ----
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        full_qpos = torch.zeros(self.robot.n_dofs, dtype=gs.tc_float, device=gs.device)
        for name in env_cfg["joint_names"]:
            j = self.robot.get_joint(name)
            full_qpos[j.dof_start] = env_cfg["default_joint_angles"][name]
        self.init_qpos = full_qpos

        # ---- target workspace bounds ----
        self.target_lower = torch.tensor(
            [command_cfg["target_x_range"][0],
             command_cfg["target_y_range"][0],
             command_cfg["target_z_range"][0]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.target_upper = torch.tensor(
            [command_cfg["target_x_range"][1],
             command_cfg["target_y_range"][1],
             command_cfg["target_z_range"][1]],
            dtype=gs.tc_float, device=gs.device,
        )

        # ---- buffers ----
        D, F = gs.device, gs.tc_float
        self.obs_buf = torch.empty((num_envs, self.num_obs), dtype=F, device=D)
        self.rew_buf = torch.empty((num_envs,), dtype=F, device=D)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=D)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=D)
        self.target_pos = torch.empty((num_envs, 3), dtype=F, device=D)
        self.ee_pos = torch.empty((num_envs, 3), dtype=F, device=D)
        self.ee_to_target_dist = torch.empty((num_envs,), dtype=F, device=D)
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=F, device=D)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty((num_envs, self.num_actions), dtype=F, device=D)
        self.dof_vel = torch.empty((num_envs, self.num_actions), dtype=F, device=D)

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
        """Reset all environments and return initial observation."""
        self._reset_idx()
        self._update_state()
        self._update_observation()
        return self.obs_buf, None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute one env step: apply actions, simulate, compute rewards."""
        cfg = self.env_cfg
        self.actions = torch.clip(actions, -cfg["clip_actions"], cfg["clip_actions"])
        target_pos = self.actions * cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_pos, self.motors_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()
        self._compute_rewards()
        self._check_termination()
        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ---- internal -----------------------------------------------------------

    def _update_state(self) -> None:
        """Read joint state and end-effector position from simulator."""
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.ee_pos = self.ee_link.get_pos()
        diff = self.ee_pos - self.target_pos
        self.ee_to_target_dist = torch.norm(diff, dim=-1)

    def _compute_rewards(self) -> None:
        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew = fn(self) * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def _check_termination(self) -> None:
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.ee_pos[:, 2] < self.env_cfg["termination_if_ee_below"]
        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

    def _update_observation(self) -> None:
        self.obs_buf = torch.cat((
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.ee_pos,
            self.target_pos,
            self.actions,
        ), dim=-1)

    def _sample_targets(self, num: int) -> torch.Tensor:
        """Sample random target positions within the workspace."""
        span = self.target_upper - self.target_lower
        return self.target_lower + span * torch.rand(
            (num, 3), dtype=gs.tc_float, device=gs.device
        )

    def _reset_idx(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Reset specified envs (or all if envs_idx is None)."""
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True)

        if envs_idx is None:
            self.target_pos = self._sample_targets(self.num_envs)
            self.dof_pos.copy_(self.default_dof_pos)
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            n_reset = envs_idx.sum().item()
            if n_reset == 0:
                return
            new_targets = self._sample_targets(self.num_envs)
            torch.where(envs_idx[:, None], new_targets, self.target_pos, out=self.target_pos)
            for buf in (self.actions, self.last_actions, self.dof_vel):
                buf.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_pos = torch.where(
                envs_idx[:, None], self.default_dof_pos, self.dof_pos
            )
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # Move target visual sphere to env-0's target
        self.target_vis.set_pos(self.target_pos[0:1])

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
