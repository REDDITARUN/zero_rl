"""Drone Hover: Crazyflie quadrotor target-tracking via Genesis physics.

Parallel environment for training a Crazyflie 2.x drone to hover and
reach random target positions while avoiding box obstacles. Demonstrates
the Genesis pattern for aerial vehicles:
- gs.morphs.Drone with propeller RPM control
- 6-DOF free-floating body (no joints, pure thrust)
- Quaternion-based orientation tracking
- Obstacle placement via Box primitives

Contrasts with locomotion/manipulation envs:
- No joints to control — actions map to 4 propeller RPMs
- Full 6-DOF state (position + orientation + velocities)
- Crash detection via orientation/boundary thresholds
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat

from .rewards import REWARD_REGISTRY


class DroneHoverEnv:
    """Parallel hovering environment for Crazyflie 2.x quadrotor.

    Observation (20-dim):
        pos(3), quat(4), lin_vel(3), ang_vel(3), target(3), actions(4).

    Actions (4-dim):
        RPM offsets from hover RPM for each propeller, scaled by rpm_scale.

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
        self.device = gs.device

        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = dict(reward_cfg["reward_scales"])

        self.hover_rpm = env_cfg["hover_rpm"]
        self.rpm_scale = env_cfg["rpm_scale"]

        # ---- scene setup (production visuals) ----
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -2.0, 2.5),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                show_world_frame=False,
                lights=[
                    {"type": "directional", "dir": (-1, -0.8, -1.2), "color": (1.0, 0.98, 0.95), "intensity": 5.5},
                    {"type": "directional", "dir": (0.6, 0.3, -0.8), "color": (0.45, 0.55, 0.75), "intensity": 2.5},
                    {"type": "directional", "dir": (0.2, -0.6, -0.4), "color": (0.40, 0.38, 0.45), "intensity": 1.5},
                ],
                ambient_light=(0.25, 0.25, 0.28),
                background_color=(0.55, 0.70, 0.92),
                shadow=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Rough(color=(0.75, 0.72, 0.68)),
        )

        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file=env_cfg["drone_urdf"],
                pos=env_cfg["drone_init_pos"],
            ),
            surface=gs.surfaces.Plastic(color=(0.18, 0.20, 0.24), roughness=0.35),
        )

        # Box obstacles
        for obs_cfg_item in env_cfg.get("obstacles", []):
            self.scene.add_entity(
                gs.morphs.Box(
                    size=obs_cfg_item["size"],
                    pos=obs_cfg_item["pos"],
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(color=(0.28, 0.30, 0.33), roughness=0.65),
            )

        # Glowing target sphere
        self.target_vis = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.04,
                pos=(0.0, 0.0, 1.0),
                fixed=True,
                collision=False,
                visualization=True,
            ),
            surface=gs.surfaces.Emission(color=(0.10, 0.82, 0.88)),
        )

        self.scene.build(n_envs=num_envs)

        self.drone.set_dofs_damping(
            torch.tensor([0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4], device=gs.device)
        )

        # ---- initial state ----
        self.init_pos = torch.tensor(
            env_cfg["drone_init_pos"], dtype=gs.tc_float, device=gs.device
        )
        self.init_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device
        )

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
        self.crash_buf = torch.zeros((num_envs,), dtype=gs.tc_bool, device=D)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=D)

        self.pos = torch.empty((num_envs, 3), dtype=F, device=D)
        self.quat = torch.empty((num_envs, 4), dtype=F, device=D)
        self.lin_vel = torch.empty((num_envs, 3), dtype=F, device=D)
        self.ang_vel = torch.empty((num_envs, 3), dtype=F, device=D)
        self.projected_gravity = torch.empty((num_envs, 3), dtype=F, device=D)
        self.euler = torch.empty((num_envs, 3), dtype=F, device=D)

        self.target_pos = torch.empty((num_envs, 3), dtype=F, device=D)
        self.target_dist = torch.empty((num_envs,), dtype=F, device=D)

        self.actions = torch.zeros((num_envs, self.num_actions), dtype=F, device=D)
        self.last_actions = torch.zeros_like(self.actions)

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
        """Execute one env step: apply RPMs, simulate, compute rewards."""
        cfg = self.env_cfg
        self.actions = torch.clip(actions, -cfg["clip_actions"], cfg["clip_actions"])

        rpms = self.hover_rpm + self.actions * self.rpm_scale
        rpms = torch.clamp(rpms, min=0.0)
        self.drone.set_propellels_rpm(rpms)
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()
        self._check_termination()
        self._compute_rewards()

        resample_mask = (
            self.episode_length_buf % int(self.command_cfg["resample_time_s"] / self.dt) == 0
        )
        self._resample_targets(resample_mask)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ---- internal -----------------------------------------------------------

    def _update_state(self) -> None:
        """Read drone pose and velocity from simulator."""
        self.pos = self.drone.get_pos()
        self.quat = self.drone.get_quat()
        inv_q = inv_quat(self.quat)
        self.lin_vel = transform_by_quat(self.drone.get_vel(), inv_q)
        self.ang_vel = transform_by_quat(self.drone.get_ang(), inv_q)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_q)
        self.euler = quat_to_xyz(self.quat, rpy=True, degrees=True)
        self.target_dist = torch.norm(self.pos - self.target_pos, dim=-1)

    def _check_termination(self) -> None:
        cfg = self.env_cfg
        timeout = self.episode_length_buf > self.max_episode_length

        crash = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        crash |= torch.abs(self.euler[:, 0]) > cfg["termination_if_roll_gt"]
        crash |= torch.abs(self.euler[:, 1]) > cfg["termination_if_pitch_gt"]
        crash |= self.pos[:, 2] < cfg["termination_if_z_below"]
        crash |= self.pos[:, 2] > cfg["termination_if_z_above"]
        crash |= torch.abs(self.pos[:, 0]) > cfg["boundary_xy"]
        crash |= torch.abs(self.pos[:, 1]) > cfg["boundary_xy"]
        self.crash_buf = crash

        self.reset_buf = (timeout | crash).bool()
        self.extras["time_outs"] = timeout.to(dtype=gs.tc_float)

    def _compute_rewards(self) -> None:
        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            rew = fn(self) * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def _update_observation(self) -> None:
        s = self.obs_scales
        self.obs_buf = torch.cat((
            self.pos * s["pos"],
            self.quat * s["quat"],
            self.lin_vel * s["lin_vel"],
            self.ang_vel * s["ang_vel"],
            self.target_pos,
            self.actions,
        ), dim=-1)

    def _sample_targets(self, num: int) -> torch.Tensor:
        """Sample random target positions within the workspace."""
        span = self.target_upper - self.target_lower
        return self.target_lower + span * torch.rand(
            (num, 3), dtype=gs.tc_float, device=gs.device
        )

    def _resample_targets(self, mask: torch.Tensor) -> None:
        """Resample target positions for envs where mask is True."""
        mask = mask.bool()
        new = self._sample_targets(self.num_envs)
        torch.where(mask[:, None], new, self.target_pos, out=self.target_pos)
        self.target_vis.set_pos(self.target_pos[0:1])

    def _reset_idx(self, envs_idx: Optional[torch.Tensor] = None) -> None:
        """Reset specified envs (or all if envs_idx is None)."""
        if envs_idx is not None:
            envs_idx = envs_idx.bool()
        if envs_idx is None:
            self.target_pos = self._sample_targets(self.num_envs)
            self.drone.set_pos(
                self.init_pos.unsqueeze(0).expand(self.num_envs, -1),
                zero_velocity=True,
            )
            self.drone.set_quat(
                self.init_quat.unsqueeze(0).expand(self.num_envs, -1),
                zero_velocity=True,
            )
            self.actions.zero_()
            self.last_actions.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.crash_buf.zero_()
        else:
            n_reset = envs_idx.sum().item()
            if n_reset == 0:
                return
            reset_idx = envs_idx.nonzero(as_tuple=False).squeeze(-1)
            self.drone.set_pos(
                self.init_pos.unsqueeze(0).expand(n_reset, -1),
                envs_idx=reset_idx,
                zero_velocity=True,
            )
            self.drone.set_quat(
                self.init_quat.unsqueeze(0).expand(n_reset, -1),
                envs_idx=reset_idx,
                zero_velocity=True,
            )
            new_targets = self._sample_targets(self.num_envs)
            torch.where(envs_idx[:, None], new_targets, self.target_pos, out=self.target_pos)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.crash_buf.masked_fill_(envs_idx, False)

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
