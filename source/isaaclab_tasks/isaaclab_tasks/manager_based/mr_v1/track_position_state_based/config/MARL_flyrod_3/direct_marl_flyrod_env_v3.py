# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
import numpy as np
import torch
from collections.abc import Sequence

from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor
from MARL_mav_carry_ext.tasks.managerbased.mdp_llc.utils import get_drone_pdist, get_drone_rpos

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, MultiMeshRayCaster
from isaaclab.utils import CircularBuffer, DelayBuffer
from isaaclab.utils.math import (
    compute_pose_error,
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    quat_rotate,
    sample_uniform,
)

from .direct_marl_flyrod_env_cfg_v3 import DirectMARLFlyrodEnvCfg


class DirectMARLFlyrodEnv(DirectMARLEnv):
    cfg: DirectMARLFlyrodEnvCfg
    _use_lidar: bool = False

    def __init__(self, cfg: DirectMARLFlyrodEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # body indices
        self._falcon_idx, self._falcon_body_names = self.robot.find_bodies(cfg.falcon_names)
        self._falcon_rotor_idx, self._falcon_rotor_names = self.robot.find_bodies(cfg.falcon_rotor_names)
        self._payload_idx = self.robot.find_bodies(cfg.payload_name)[0]
        self._rope_idx = self._resolve_rope_indices()

        falcon1_rotor_ids, falcon1_rotor_names = self.robot.find_bodies("Falcon1_rotor_.*")
        falcon2_rotor_ids, falcon2_rotor_names = self.robot.find_bodies("Falcon2_rotor_.*")
        falcon1_pairs = sorted(zip(falcon1_rotor_names, falcon1_rotor_ids))
        falcon2_pairs = sorted(zip(falcon2_rotor_names, falcon2_rotor_ids))
        self._falcon1_rotor_idx = [int(body_id) for _, body_id in falcon1_pairs]
        self._falcon2_rotor_idx = [int(body_id) for _, body_id in falcon2_pairs]

        rotor_id_to_force_index = {int(body_id): idx for idx, body_id in enumerate(self._falcon_rotor_idx)}
        self._falcon1_rotor_force_idx = torch.tensor(
            [rotor_id_to_force_index[body_id] for body_id in self._falcon1_rotor_idx],
            device=self.device,
            dtype=torch.long,
        )
        self._falcon2_rotor_force_idx = torch.tensor(
            [rotor_id_to_force_index[body_id] for body_id in self._falcon2_rotor_idx],
            device=self.device,
            dtype=torch.long,
        )

        # configuration
        self._num_drones = len(self._falcon_idx)
        self._control_mode = cfg.control_mode
        self._use_lidar = self.cfg.num_lidar_rays > 0
        self._rope_terms_enabled = len(self._rope_idx) > 0 and len(self._rope_idx) % self._num_drones == 0
        self._rope_group_size = len(self._rope_idx) // self._num_drones if self._rope_terms_enabled else 0
        self._rope_endpoint_idx = (
            torch.tensor(self._rope_idx, device=self.device, dtype=torch.long).view(self._num_drones, self._rope_group_size)[:, -1]
            if self._rope_terms_enabled
            else torch.empty(0, device=self.device, dtype=torch.long)
        )
        if not self._rope_terms_enabled:
            print(
                "[Flyrod] Rope-angle/cable terms disabled: expected a nonzero rope-body count "
                f"divisible by {self._num_drones}, found {len(self._rope_idx)}"
            )

        # observation buffers
        self._observation_buffers = {}
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent] = CircularBuffer(cfg.history_len, self.num_envs, device=self.device)

        # action buffers
        # buffers -- initialize z-forces to hover thrust to avoid cold-start fall
        _hover_thrust_per_rotor = (2 * 0.6017 + self.cfg.rod_mass) * 9.8066 / 8
        self._forces = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), 3, device=self.device)
        self._forces[..., 2] = _hover_thrust_per_rotor
        self._moments = torch.zeros(self.num_envs, len(self._falcon_idx), 3, device=self.device)
        self._thrust_cmds = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), device=self.device)
        self._thrust_cmds[:] = _hover_thrust_per_rotor
        # self.setpoint_delay_buffers = {}
        # for agent in self.cfg.possible_agents:
        #     self.setpoint_delay_buffers[agent] = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)
        #     self.setpoint_delay_buffers[agent].set_time_lag(cfg.constant_delay)
        self._setpoints = {}
        self.prev_actions = {}
        for agent in self.cfg.possible_agents:
            self._setpoints[agent] = {}
            if self._control_mode == "geometric":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 12, device=self.device)
            elif self._control_mode == "ACCBR":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 5, device=self.device)
            elif self._control_mode == "THRUST":
                self.prev_actions[agent] = torch.zeros(self.num_envs, 4, device=self.device)

        self.drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_orientations = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)
        self.drone_orientations[..., 0] = 1.0
        self.drone_rot_matrices = torch.zeros(self.num_envs, self._num_drones, 3, 3, device=self.device)
        self.drone_linear_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_linear_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_jerk = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_prev_acc = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)

        # pre-allocated one-hot agent ID buffers (avoids CPU alloc + H2D copy every obs step)
        self._onehot_falcon1 = torch.zeros(self.num_envs, 3, device=self.device)
        self._onehot_falcon1[:, 0] = 1.0
        self._onehot_falcon2 = torch.zeros(self.num_envs, 3, device=self.device)
        self._onehot_falcon2[:, 1] = 1.0

        # outer loop controller
        self.geo_controllers = {}
        for i in range(self._num_drones):
            self.geo_controllers[i] = GeometricController(self.num_envs, self._control_mode)
        self._ll_counter = 0
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)
        self._zeros = torch.zeros([self.num_envs, 3], device=self.device)

        # inner loop controller
        self._indi_controllers = {}
        for i in range(self._num_drones):
            self._indi_controllers[i] = IndiController(self.num_envs)

        # motor model
        # experimentally obtained
        self.motor_models = {}
        initial_rpms = [
            torch.tensor([[1441.5819, 1351.1626, 1341.0111, 1428.5597]], device=self.device).repeat(self.num_envs, 1),
            torch.tensor([[1377.9199, 1451.8428, 1408.9022, 1329.2014]], device=self.device).repeat(self.num_envs, 1),
            torch.tensor([[1281.3964, 1293.0708, 1361.7539, 1347.2434]], device=self.device).repeat(self.num_envs, 1),
        ]
        for i in range(self._num_drones):
            self.motor_models[i] = RotorMotor(self.num_envs, initial_rpms[i])
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # load buffers
        self.load_position = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_orientation = torch.zeros(self.num_envs, 4, device=self.device)
        self.load_orientation[:, 0] = 1.0
        self.current_load_matrix = torch.zeros(self.num_envs, 3, 3, device=self.device)
        self.load_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.load_length_x = torch.tensor([[0.275, 0, 0]] * self.num_envs, device=self.device)
        self.load_length_y = torch.tensor([[0, 0.275, 0]] * self.num_envs, device=self.device)

        # Goal terms
        # # goal buffers
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0

        self.goal_pos_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.drone_to_goal_error = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)

        self.goal_dist_counter = torch.zeros(self.num_envs, device=self.device)

        # progress-reward state: previous step's load-to-goal distance (per env)
        self._prev_load_dist = torch.zeros(self.num_envs, device=self.device)

        # reward logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "progress_reward",
                "distance_reward",
                "upright_reward",
                "action_smoothness",
                "body_rate_penalty",
                "force_penalty",
                "rope_vertical",
                "velocity_sync",
                "speed_limit",
                "crash_penalty",
                "cross_track_funnel",
                "time_penalty",
            ]
        }

        # Attachment offsets in rod local frame, aligned with resolved Falcon body order.
        # This avoids rewarding the wrong drone if find_bodies order differs from Falcon1/Falcon2 naming.
        attach_offsets = torch.zeros(self._num_drones, 3, device=self.device)
        for i, body_name in enumerate(self._falcon_body_names):
            if "Falcon1" in body_name:
                attach_offsets[i, 0] = +cfg.rod_half_length
            elif "Falcon2" in body_name:
                attach_offsets[i, 0] = -cfg.rod_half_length
            else:
                attach_offsets[i, 0] = +cfg.rod_half_length if i == 0 else -cfg.rod_half_length

        self.attach_offsets_local = attach_offsets.unsqueeze(0).expand(self.num_envs, self._num_drones, 3).contiguous()

        # -- metrics
        self.metrics = {}
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["drone_to_goal_distance"] = torch.zeros(self.num_envs, device=self.device)

        # termination buffers
        self.falcon_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.payload_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.falcon_fly_high = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.illegal_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.angle_limit_drone = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.angle_limit_load = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.cable_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.drone_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.body_pos_outside = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # debug vis
        self.set_debug_vis(cfg.debug_vis)

    def _resolve_rope_indices(self) -> list[int]:
        """Resolve rope body ids with a fallback for flyrod rope naming variants."""
        try:
            rope_ids, _ = self.robot.find_bodies(self.cfg.rope_name)
            return [int(i) for i in rope_ids]
        except ValueError:
            all_ids, all_names = self.robot.find_bodies(".*")
            fallback_ids = [
                int(all_ids[idx])
                for idx, name in enumerate(all_names)
                if name.startswith("rope_1_link_") or name.startswith("rope_2_link_")
            ]
            if len(fallback_ids) == 0:
                raise RuntimeError(
                    "Failed to resolve rope bodies. "
                    f"cfg.rope_name='{self.cfg.rope_name}' matched none and no fallback rope_*_link_* bodies were found."
                )
            return fallback_ids

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        contact_sensors = ContactSensor(self.cfg.contact_forces)
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)  # TODO: not sure what this does
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_forces"] = contact_sensors
        # LiDAR sensors are disabled for this variant.
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent in self.cfg.possible_agents:
            agent_name = str(agent)
            # terms used for smoothness reward, current and previously calculated actions
            self.prev_actions[agent_name][:] = self.actions[agent_name]
            self.actions[agent_name][:] = actions[agent_name]

            # introduce delay in the setpoints
            # actions[agent][:] = self.setpoint_delay_buffers[agent].compute(actions[agent])

        for drone, action in actions.items():
            if self._control_mode == "THRUST":
                action_clamped = torch.clamp(action, -1.0, 1.0)
                # Map from [-1, 1] to [thrust_min, thrust_max] using tensor arithmetic
                lower = torch.tensor(self.cfg.thrust_min, device=action_clamped.device, dtype=action_clamped.dtype)
                upper = torch.tensor(self.cfg.thrust_max, device=action_clamped.device, dtype=action_clamped.dtype)
                thrusts = (action_clamped + 1.0) * 0.5 * (upper - lower) + lower
                if drone == "falcon1":
                    self._thrust_cmds[:, self._falcon1_rotor_force_idx] = thrusts
                elif drone == "falcon2":
                    self._thrust_cmds[:, self._falcon2_rotor_force_idx] = thrusts
                continue

            if self._control_mode == "geometric":
                self._setpoints[drone]["pos"] = action[:, :3]
                self._setpoints[drone]["lin_vel"] = action[:, 3:6]
                self._setpoints[drone]["lin_acc"] = action[:, 6:9]
                self._setpoints[drone]["jerk"] = action[:, 9:12]

            elif self._control_mode == "ACCBR":
                self._setpoints[drone]["lin_acc"] = action[:, :3] * self.cfg.acc_scale
                self._setpoints[drone]["body_rates"] = torch.cat((action[:, 3:] * self.cfg.body_rate_scale, self._constant_yaw), dim=-1)

            self._setpoints[drone]["yaw"] = self._constant_yaw
            self._setpoints[drone]["yaw_rate"] = self._constant_yaw
            self._setpoints[drone]["yaw_acc"] = self._constant_yaw

    def _apply_action(self) -> None:
        if self._control_mode == "THRUST":
            self._forces[..., 2] = self._thrust_cmds
            self._moments.zero_()
        elif self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts = []
            all_moments = []

            drone_positions = self.robot.data.body_com_state_w[
                :, self._falcon_idx, :3
            ] - self.scene.env_origins.unsqueeze(1)
            drone_orientations = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
            drone_linear_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
            drone_angular_velocities = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]
            drone_linear_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, :3]
            drone_angular_accelerations = self.robot.data.body_acc_w[:, self._falcon_idx, 3:6]

            self.drone_positions[:] = drone_positions  # + torch.randn_like(drone_positions) * self.position_noise_std
            self.drone_orientations[:] = (
                drone_orientations  # + torch.randn_like(drone_orientations) * self.orientation_noise_std
            )
            self.drone_linear_velocities[:] = (
                drone_linear_velocities  # + torch.randn_like(drone_linear_velocities) * self.linear_velocity_noise_std
            )
            self.drone_angular_velocities[:] = (
                drone_angular_velocities  # + torch.randn_like(drone_angular_velocities) * self.angular_velocity_noise_std
            )
            self.drone_linear_accelerations[:] = (
                drone_linear_accelerations  # + torch.randn_like(drone_linear_accelerations) * self.linear_acceleration_noise_std
            )
            self.drone_angular_accelerations[:] = (
                drone_angular_accelerations  # + torch.randn_like(drone_angular_accelerations) * self.angular_acceleration_noise_std
            )

            for i in range(self._num_drones):
                drone_states: dict = {}  # dict of tensors
                drone_states["pos"] = self.drone_positions[:, i]
                drone_states["quat"] = self.drone_orientations[:, i]
                drone_states["lin_vel"] = self.drone_linear_velocities[:, i]
                drone_states["ang_vel"] = self.drone_angular_velocities[:, i]
                drone_states["lin_acc"] = self.drone_linear_accelerations[:, i]
                drone_states["ang_acc"] = self.drone_angular_accelerations[:, i]
                # calculate current jerk
                self._drone_jerk[:, i] = (drone_states["lin_acc"] - self._drone_prev_acc[:, i]) / (self.step_dt)
                drone_states["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = drone_states["lin_acc"]

                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[f"falcon{i+1}"]
                )

                target_rpm = self._indi_controllers[i].getCommand(
                    drone_states, self._forces[:, i * 4 : i * 4 + 4], alpha_cmd, acc_cmd, acc_load
                )

                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(target_rpm, self.sampling_time)
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            forces = torch.cat(all_thrusts, dim=-1)
            torques = torch.cat(all_moments, dim=-1)
            self._forces[..., 2] = forces
            self._moments[..., 2] = torques.view(self.num_envs, self._num_drones, 4).sum(-1)
            self._ll_counter = 0
        self._ll_counter += 1

        # apply torques induced by rotors to each body
        self.robot.set_external_force_and_torque(torch.zeros_like(self._moments), self._moments, body_ids=self._falcon_idx)
        # apply forces to each rotor
        self.robot.set_external_force_and_torque(self._forces, torch.zeros_like(self._forces), body_ids=self._falcon_rotor_idx)

    def _get_observations(self) -> dict[str, torch.Tensor]:

        # local observations include:
        # load state
        # ego-drone drone
        # other-drone_states
        # goal terms

        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )
        self.load_orientation[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 3:7].squeeze(1)
        self.current_load_matrix[:] = matrix_from_quat(self.load_orientation)
        self.load_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 7:10].squeeze(1)
        self.load_ang_vel[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 10:13].squeeze(1)

        self.drone_positions[:] = self.robot.data.body_com_state_w[
            :, self._falcon_idx, :3
        ] - self.scene.env_origins.unsqueeze(1)
        self.drone_orientations[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)
        self.drone_linear_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
        self.drone_angular_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]

        self.goal_pos_error[:] = self.pose_command_w[:, :3] - self.load_position
        self.drone_to_goal_error[:] = self.pose_command_w[:, :3].unsqueeze(1) - self.drone_positions

        # -- Noisy observation copies (Gaps 1 & 2)
        # Class buffers above stay clean -- rewards and terminations read them directly.
        # Only the tensors passed into obs construction are perturbed, simulating
        # IMU drift, MOCAP jitter, and 6-DOF payload pose estimation error.
        _pn = self.cfg.position_noise_std
        _vn = self.cfg.velocity_noise_std
        _on = self.cfg.orient_noise_std
        _an = self.cfg.ang_vel_noise_std
        load_pos_obs = self.load_position + torch.randn_like(self.load_position) * _pn
        load_vel_obs = self.load_vel + torch.randn_like(self.load_vel) * _vn
        load_ang_vel_obs = self.load_ang_vel + torch.randn_like(self.load_ang_vel) * _an
        drone_pos_obs = self.drone_positions + torch.randn_like(self.drone_positions) * _pn
        drone_vel_obs = self.drone_linear_velocities + torch.randn_like(self.drone_linear_velocities) * _vn
        drone_ang_vel_obs = self.drone_angular_velocities + torch.randn_like(self.drone_angular_velocities) * _an

        # SO(3)-preserving orientation noise: apply a small random rotation (axis-angle -> quat)
        # so the rotation matrix fed to the policy stays a valid element of SO(3).
        if _on > 0.0:
            # Load: random unit axis x small angle -> delta quaternion -> perturbed R
            ax_l = torch.randn(self.num_envs, 3, device=self.device)
            ax_l = ax_l / ax_l.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dq_l = quat_from_angle_axis(torch.randn(self.num_envs, device=self.device) * _on, ax_l)
            load_mat_obs = matrix_from_quat(quat_mul(dq_l, self.load_orientation))
            # Drones: flatten (num_envs, num_drones, 4) -> (N, 4), perturb, reshape
            nd = self.num_envs * self._num_drones
            ax_d = torch.randn(nd, 3, device=self.device)
            ax_d = ax_d / ax_d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dq_d = quat_from_angle_axis(torch.randn(nd, device=self.device) * _on, ax_d)
            drone_rot_obs = matrix_from_quat(
                quat_mul(dq_d, self.drone_orientations.view(-1, 4))
            ).view(self.num_envs, self._num_drones, 3, 3)
        else:
            load_mat_obs = self.current_load_matrix
            drone_rot_obs = self.drone_rot_matrices

        # Goal errors derived from noisy positions so they are internally consistent
        goal_error_obs = self.pose_command_w[:, :3] - load_pos_obs
        drone_to_goal_obs = self.pose_command_w[:, :3].unsqueeze(1) - drone_pos_obs

        lidar_f1_obs = ()
        lidar_f2_obs = ()

        # LiDAR observations are disabled for this variant.

        if self.cfg.partial_obs:
            obs_falcon1_t = torch.cat(
                (
                    load_pos_obs,
                    load_mat_obs.view(self.num_envs, -1),
                    # drone terms
                    self._onehot_falcon1,
                    drone_pos_obs[:, 0].view(self.num_envs, -1),
                    drone_rot_obs[:, 0].view(self.num_envs, -1),
                    drone_vel_obs[:, 0].view(self.num_envs, -1),
                    drone_ang_vel_obs[:, 0].view(self.num_envs, -1),
                    goal_error_obs,
                    drone_to_goal_obs[:, 0],
                    *lidar_f1_obs,
                ),
                dim=-1,
            )

            self._observation_buffers["falcon1"].append(obs_falcon1_t)

            obs_falcon2_t = torch.cat(
                (
                    load_pos_obs,
                    load_mat_obs.view(self.num_envs, -1),
                    # drone terms
                    self._onehot_falcon2,
                    drone_pos_obs[:, 1].view(self.num_envs, -1),
                    drone_rot_obs[:, 1].view(self.num_envs, -1),
                    drone_vel_obs[:, 1].view(self.num_envs, -1),
                    drone_ang_vel_obs[:, 1].view(self.num_envs, -1),
                    goal_error_obs,
                    drone_to_goal_obs[:, 1],
                    *lidar_f2_obs,
                ),
                dim=-1,
            )

            self._observation_buffers["falcon2"].append(obs_falcon2_t)

            obs_falcon1 = self._observation_buffers["falcon1"].buffer.reshape(self.num_envs, -1)
            obs_falcon2 = self._observation_buffers["falcon2"].buffer.reshape(self.num_envs, -1)
        else:
            obs_falcon1 = torch.cat(
                (
                    load_pos_obs,
                    load_mat_obs.view(self.num_envs, -1),
                    load_vel_obs,
                    load_ang_vel_obs,
                    # drone terms
                    self._onehot_falcon1,  # one-hot encoding
                    drone_pos_obs.view(self.num_envs, -1),
                    drone_rot_obs.view(self.num_envs, -1),
                    drone_vel_obs.view(self.num_envs, -1),
                    drone_ang_vel_obs.view(self.num_envs, -1),
                    goal_error_obs,
                    drone_to_goal_obs.view(self.num_envs, -1),
                    *lidar_f1_obs,
                ),
                dim=-1,
            )

            obs_falcon2 = torch.cat(
                (
                    load_pos_obs,
                    load_mat_obs.view(self.num_envs, -1),
                    load_vel_obs,
                    load_ang_vel_obs,
                    # drone terms
                    self._onehot_falcon2,  # one-hot encoding
                    drone_pos_obs.view(self.num_envs, -1),
                    drone_rot_obs.view(self.num_envs, -1),
                    drone_vel_obs.view(self.num_envs, -1),
                    drone_ang_vel_obs.view(self.num_envs, -1),
                    goal_error_obs,
                    drone_to_goal_obs.view(self.num_envs, -1),
                    *lidar_f2_obs,
                ),
                dim=-1,
            )

        observations = {
            "falcon1": obs_falcon1,
            "falcon2": obs_falcon2,
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # load terms
                self.load_position,
                self.current_load_matrix.view(self.num_envs, -1),
                self.load_vel,
                self.load_ang_vel,
                # drone terms
                self.drone_positions.view(self.num_envs, -1),
                self.drone_rot_matrices.view(self.num_envs, -1),
                self.drone_linear_velocities.view(self.num_envs, -1),
                self.drone_angular_velocities.view(self.num_envs, -1),
                # goal terms -- position error only (no orientation goal)
                self.goal_pos_error,
            ),
            dim=-1,
        )

        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Rod position tracking uses the same step-based shaping scale as the
        # manager-based flyrod task, without extra time normalization.
        # Progress (potential-based shaping) gives a useful gradient even when the
        # Gaussian term is saturated near zero far from the goal; the Gaussian gives
        # a smooth, flat-topped attractor at the goal so the agent stabilises.
        goal_pos_error = self.pose_command_w[:, :3] - self.load_position
        goal_pos_error_norm = torch.norm(goal_pos_error, dim=-1)

        # On the first step of an episode, prev_dist is undefined -- clamp to current
        # distance so the first-step progress is exactly zero (no spurious bonus).
        reset_mask = self.episode_length_buf == 0
        prev_dist = torch.where(reset_mask, goal_pos_error_norm, self._prev_load_dist)
        progress = prev_dist - goal_pos_error_norm
        reward_progress = self.cfg.progress_weight * progress

        # Scale each axis by its goal_range span so shaping matches the configured
        # task distribution. For fixed-axis goals (span=0), fall back to distance_std.
        goal_range_half_span = torch.tensor(
            [
                0.5 * (self.cfg.goal_range["pos_x"][1] - self.cfg.goal_range["pos_x"][0]),
                0.5 * (self.cfg.goal_range["pos_y"][1] - self.cfg.goal_range["pos_y"][0]),
                0.5 * (self.cfg.goal_range["pos_z"][1] - self.cfg.goal_range["pos_z"][0]),
            ],
            device=self.device,
            dtype=goal_pos_error.dtype,
        )
        distance_scales = torch.where(
            goal_range_half_span > 1.0e-6,
            goal_range_half_span,
            torch.full_like(goal_range_half_span, float(self.cfg.distance_std)),
        )
        normalized_goal_error = goal_pos_error / distance_scales
        normalized_goal_error_norm_sq = torch.sum(normalized_goal_error.square(), dim=-1)
        reward_distance = self.cfg.distance_weight * torch.exp(-normalized_goal_error_norm_sq)

        self._prev_load_dist = goal_pos_error_norm.detach().clone()

        # upright orientation: reward drone z-axis pointing in +world-z direction
        drone_z_up = self.drone_rot_matrices[:, :, 2, 2]  # (num_envs, num_drones): cos of tilt angle
        drone_tilt = 1.0 - drone_z_up  # 0 = level, 2 = upside-down
        reward_upright = self.cfg.upright_orient_weight * torch.exp(-drone_tilt.mean(dim=-1) * 3.0)

        # action smoothness reward
        current_actions = torch.cat([self.actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        action_prev = torch.cat([self.prev_actions[agent] for agent in self.cfg.possible_agents], dim=-1)
        diff_action = ((current_actions - action_prev).abs()) / self._num_drones
        reward_action_smoothness = (
            self.cfg.action_smoothness_weight * torch.exp(-torch.norm(diff_action, dim=-1).square()) * self.step_dt
        )

        # commanded body rate penalty (ACCBR only)
        if self._control_mode == "THRUST":
            reward_body_rate_penalty = torch.zeros(self.num_envs, device=self.device)
        else:
            commanded_body_rates = torch.cat([self.actions[agent][:, 3:] for agent in self.cfg.possible_agents], dim=-1)
            body_rate_penalty = torch.norm(commanded_body_rates / self._num_drones, dim=-1)
            reward_body_rate_penalty = self.cfg.body_rate_penalty_weight * torch.exp(-body_rate_penalty) * self.step_dt

        # force penalty
        normalized_forces = self._forces[..., 2] / self.cfg.max_thrust_pp
        effort_mean = torch.mean(normalized_forces, dim=-1)
        reward_effort = self.cfg.force_penalty_weight * torch.exp(-effort_mean) * self.step_dt

        # rope verticality: drones should stay directly above their rod attachment points
        # attach_world = load_pos + R(load_quat) @ offset_local, then penalise XY offset to drone
        load_quat_per_drone = self.load_orientation.repeat_interleave(self._num_drones, dim=0)
        attach_offsets_world = quat_rotate(
            load_quat_per_drone, self.attach_offsets_local.reshape(-1, 3)
        ).view(self.num_envs, self._num_drones, 3)
        attach_points_world = self.load_position.unsqueeze(1) + attach_offsets_world
        xy_err = self.drone_positions[..., :2] - attach_points_world[..., :2]
        xy_err_norm = torch.norm(xy_err, dim=-1).mean(dim=-1)
        reward_rope_vertical = (
            self.cfg.rope_vertical_weight * torch.exp(-xy_err_norm * 4.0) * self.step_dt
        )

        # velocity synchronisation: penalise differences between the two drones' linear velocities
        vel_diff = self.drone_linear_velocities[:, 0] - self.drone_linear_velocities[:, 1]
        vel_diff_norm = torch.norm(vel_diff, dim=-1)
        reward_velocity_sync = (
            self.cfg.velocity_sync_weight * torch.exp(-vel_diff_norm * 1.5) * self.step_dt
        )

        # payload speed soft-cap: only penalises speed above v_max so slow motion is uncosted
        load_speed = torch.norm(self.load_vel, dim=-1)
        speed_excess = torch.clamp(load_speed - self.cfg.payload_v_max, min=0.0)
        reward_speed_limit = (
            self.cfg.speed_limit_weight * torch.exp(-speed_excess * 2.0) * self.step_dt
        )

        # cross-track funnel: penalise Y deviation so drones aim for the door gap, not the walls
        cross_track_error = torch.abs(self.load_position[:, 1])
        reward_funnel = -self.cfg.cross_track_weight * cross_track_error * self.step_dt

        # constant time penalty: makes hovering costly so the agent actively seeks the goal
        reward_time_penalty = torch.full(
            (self.num_envs,), -self.cfg.time_penalty_weight * self.step_dt, device=self.device
        )

        # one-shot crash penalty: fires on the step a non-timeout termination triggers.
        # Computed inline because _get_dones runs after _get_rewards in this env.
        falcon_low_now = (self.drone_positions[:, :, 2] < self.cfg.fly_low_threshold).any(dim=-1)
        payload_low_now = self.load_position[:, 2] < self.cfg.fly_low_threshold
        bbox_now = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)
        payload_bbox_now = (self.load_position.abs() > self.cfg.bounding_box_threshold).any(dim=-1)
        bbox_now = bbox_now | payload_bbox_now

        contact_sensor = self.scene.sensors[self.cfg.sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        body_ids = self.cfg.sensor_cfg.body_ids
        if isinstance(body_ids, slice):
            start = 0 if body_ids.start is None else body_ids.start
            stop = net_contact_forces.shape[2] if body_ids.stop is None else body_ids.stop
            step = 1 if body_ids.step is None else body_ids.step
            contact_ids = list(range(start, stop, step))
        else:
            contact_ids = list(body_ids)
        # flatten falcon indices in case find_bodies returned nested lists
        def _flatten_ids_local(xs):
            out = []
            if xs is None:
                return out
            # single int/string id
            if isinstance(xs, (int, str)):
                try:
                    out.append(int(xs))
                except Exception:
                    pass
                return out
            for x in xs:
                if isinstance(x, (list, tuple)):
                    out.extend([int(i) for i in x])
                else:
                    out.append(int(x))
            return out

        extra_ids = _flatten_ids_local(self._payload_idx) + _flatten_ids_local(self._falcon_idx)
        for cid in extra_ids:
            if cid not in contact_ids:
                contact_ids.append(cid)
        contact_forces_selected = net_contact_forces[:, :, contact_ids]
        max_forces = torch.max(torch.norm(contact_forces_selected, dim=-1), dim=1)[0]
        illegal_contact_now = torch.any(max_forces > self.cfg.contact_sensor_threshold, dim=1)

        rpos = get_drone_rpos(self.drone_positions)
        pdist = get_drone_pdist(rpos)
        separation = pdist.min(dim=-1).values.min(dim=-1).values
        drone_collision_now = separation < self.cfg.drone_collision_threshold

        crash_now = falcon_low_now | payload_low_now | bbox_now | illegal_contact_now | drone_collision_now
        if self.cfg.enable_cable_terminations and self._rope_terms_enabled:
            rope_orientations_world = self.robot.data.body_com_state_w[:, self._rope_endpoint_idx, 3:7]
            drone_orientation_inv = quat_inv(self.drone_orientations)
            rope_orientations_drones = quat_mul(drone_orientation_inv, rope_orientations_world)
            roll_drone, pitch_drone, _ = euler_xyz_from_quat(rope_orientations_drones.view(-1, 4))
            mapped_angle_drone = torch.stack((torch.cos(roll_drone), torch.cos(pitch_drone)), dim=1).view(
                self.num_envs, self._num_drones, 2
            )
            angle_limit_drone_now = (mapped_angle_drone < self.cfg.cable_angle_limits_drone).any(dim=-1).any(dim=-1)

            payload_orientation_world = self.load_orientation.unsqueeze(1).expand(-1, self._num_drones, -1)
            payload_orientation_inv = quat_inv(payload_orientation_world)
            rope_orientations_payload = quat_mul(payload_orientation_inv, rope_orientations_world)
            roll_load, pitch_load, _ = euler_xyz_from_quat(rope_orientations_payload.view(-1, 4))
            mapped_angle_load = torch.stack((torch.cos(roll_load), torch.cos(pitch_load)), dim=1).view(
                self.num_envs, self._num_drones, 2
            )
            angle_limit_load_now = (mapped_angle_load < self.cfg.cable_angle_limits_payload).any(dim=-1).any(dim=-1)

            cable_collision_now = self._cable_collision(
                self.cfg.cable_collision_threshold, self.cfg.cable_collision_num_points
            )
            crash_now = crash_now | angle_limit_drone_now | angle_limit_load_now | cable_collision_now

        reward_crash = -self.cfg.crash_penalty * crash_now.float()

        rewards = {
            "progress_reward": reward_progress,
            "distance_reward": reward_distance,
            "upright_reward": reward_upright,
            "action_smoothness": reward_action_smoothness,
            "body_rate_penalty": reward_body_rate_penalty,
            "force_penalty": reward_effort,
            "rope_vertical": reward_rope_vertical,
            "velocity_sync": reward_velocity_sync,
            "speed_limit": reward_speed_limit,
            "crash_penalty": reward_crash,
            "cross_track_funnel": reward_funnel,
            "time_penalty": reward_time_penalty,
        }

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        shared_rewards = (
            reward_progress
            + reward_distance
            + reward_upright
            + reward_action_smoothness
            + reward_body_rate_penalty
            + reward_effort
            + reward_rope_vertical
            + reward_velocity_sync
            + reward_speed_limit
            + reward_crash
            + reward_funnel
            + reward_time_penalty
        )

        return {str(agent): shared_rewards for agent in self.cfg.possible_agents}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        The terminations for the environment. Since all of the agents are connected by the cables,
        if 1 agent terminates, terminate all agents.
        """
        self.load_position[:] = (
            self.robot.data.body_com_state_w[:, self._payload_idx, :3].squeeze(1) - self.scene.env_origins
        )

        # crashing into ground
        self.falcon_fly_low = (self.drone_positions[:, :, 2] < self.cfg.fly_low_threshold).any(dim=-1)
        self.payload_fly_low = self.load_position[:, 2] < self.cfg.fly_low_threshold
        self.falcon_fly_high = (self.drone_positions[:, :, 2] > self.cfg.fly_high_threshold).any(dim=-1)

        # illegal contact
        contact_sensor = self.scene.sensors[self.cfg.sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        body_ids = self.cfg.sensor_cfg.body_ids
        if isinstance(body_ids, slice):
            start = 0 if body_ids.start is None else body_ids.start
            stop = net_contact_forces.shape[2] if body_ids.stop is None else body_ids.stop
            step = 1 if body_ids.step is None else body_ids.step
            contact_ids = list(range(start, stop, step))
        else:
            contact_ids = list(body_ids)
        # flatten falcon indices in case find_bodies returned nested lists
        def _flatten_ids_local(xs):
            out = []
            if xs is None:
                return out
            # single int/string id
            if isinstance(xs, (int, str)):
                try:
                    out.append(int(xs))
                except Exception:
                    pass
                return out
            for x in xs:
                if isinstance(x, (list, tuple)):
                    out.extend([int(i) for i in x])
                else:
                    out.append(int(x))
            return out

        extra_ids = _flatten_ids_local(self._payload_idx) + _flatten_ids_local(self._falcon_idx)
        for cid in extra_ids:
            if cid not in contact_ids:
                contact_ids.append(cid)
        contact_forces_selected = net_contact_forces[:, :, contact_ids]
        max_forces = torch.max(torch.norm(contact_forces_selected, dim=-1), dim=1)[0]
        self.illegal_contact = torch.any(max_forces > self.cfg.contact_sensor_threshold, dim=1)

        # angle limits
        if self._rope_terms_enabled:
            rope_orientations_world = self.robot.data.body_com_state_w[:, self._rope_endpoint_idx, 3:7]
            drone_orientation_inv = quat_inv(self.drone_orientations)
            rope_orientations_drones = quat_mul(drone_orientation_inv, rope_orientations_world)
            roll_drone, pitch_drone, _ = euler_xyz_from_quat(rope_orientations_drones.view(-1, 4))
            mapped_angle_drone = torch.stack((torch.cos(roll_drone), torch.cos(pitch_drone)), dim=1).view(
                self.num_envs, self._num_drones, 2
            )
            self.angle_limit_drone = (mapped_angle_drone < self.cfg.cable_angle_limits_drone).any(dim=-1).any(dim=-1)

            self.load_orientation[:] = self.robot.data.body_com_state_w[:, self._payload_idx, 3:7].squeeze(1)
            payload_orientation_world = self.load_orientation.unsqueeze(1).expand(-1, self._num_drones, -1)
            payload_orientation_inv = quat_inv(payload_orientation_world)
            rope_orientations_payload = quat_mul(payload_orientation_inv, rope_orientations_world)
            roll_load, pitch_load, _ = euler_xyz_from_quat(rope_orientations_payload.view(-1, 4))
            mapped_angle_load = torch.stack((torch.cos(roll_load), torch.cos(pitch_load)), dim=1).view(
                self.num_envs, self._num_drones, 2
            )
            self.angle_limit_load = (mapped_angle_load < self.cfg.cable_angle_limits_payload).any(dim=-1).any(dim=-1)

            # cables colliding
            self.cable_collision = self._cable_collision(
                self.cfg.cable_collision_threshold, self.cfg.cable_collision_num_points
            )
        else:
            self.angle_limit_drone[:] = False
            self.angle_limit_load[:] = False
            self.cable_collision[:] = False

        # drones colliding
        rpos = get_drone_rpos(self.drone_positions)
        pdist = get_drone_pdist(rpos)
        separation = (
            pdist.min(dim=-1).values.min(dim=-1).values
        )  # get the smallest distance between drones in the swarm
        self.drone_collision = separation < self.cfg.drone_collision_threshold

        # bounding box
        self.body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)
        payload_outside = (self.load_position.abs() > self.cfg.bounding_box_threshold).any(dim=-1)
        self.body_pos_outside = self.body_pos_outside | payload_outside

        # update metrics
        self._update_metrics()

        # reset when episode ends
        terminations = (
            self.falcon_fly_low
            | self.payload_fly_low
            | self.falcon_fly_high
            | self.illegal_contact
            | self.drone_collision
            | self.body_pos_outside
        )
        if self.cfg.enable_cable_terminations:
            terminations = (
                terminations | self.angle_limit_drone | self.angle_limit_load | self.cable_collision
            )
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        timed_outs = self.time_out

        terminated = {str(agent): terminations for agent in self.cfg.possible_agents}
        time_outs = {str(agent): timed_outs for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_payload_com_from_reset_base(self, env_ids: torch.Tensor) -> None:
        """Apply reset_base translation on payload (rod) CoM instead of articulation root.

        The default event `mdp.reset_root_state_uniform` samples root translation. For flyrod,
        rewards and goals are defined on the payload CoM (`rod_link`), so we shift the sampled
        root pose by a delta that places payload CoM at the sampled reset position.
        """
        reset_base = getattr(self.cfg.events, "reset_base", None)
        if reset_base is None:
            return

        params = getattr(reset_base, "params", None)
        if not isinstance(params, dict):
            return

        pose_range = params.get("pose_range", {})
        if not isinstance(pose_range, dict):
            return

        num_ids = int(env_ids.shape[0])
        if num_ids == 0:
            return

        def _sample_axis(axis: str) -> torch.Tensor:
            low, high = pose_range.get(axis, (0.0, 0.0))
            return torch.empty(num_ids, device=self.device).uniform_(low, high)

        desired_payload_pos_env = torch.stack(
            [_sample_axis("x"), _sample_axis("y"), _sample_axis("z")], dim=-1
        )
        desired_payload_pos_w = desired_payload_pos_env + self.scene.env_origins[env_ids]

        current_payload_pos_w = self.robot.data.body_com_state_w[:, self._payload_idx, :3][env_ids].squeeze(1)
        root_state_w = self.robot.data.root_state_w[env_ids].clone()

        delta_w = desired_payload_pos_w - current_payload_pos_w
        root_pose = root_state_w[:, :7]
        root_pose[:, :3] = root_pose[:, :3] + delta_w

        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state_w[:, 7:13], env_ids=env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = env_ids
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids_tensor)
        self._reset_payload_com_from_reset_base(env_ids_tensor)
        self._reset_target_pose(env_ids_tensor)

        for agent in self.cfg.possible_agents:
            # self.setpoint_delay_buffers[agent].reset(env_ids)
            self._observation_buffers[agent].reset(env_ids_tensor)

        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["Episode_Termination/angle_drones_cable"] = torch.count_nonzero(
            self.angle_limit_drone[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/angle_load_cable"] = torch.count_nonzero(
            self.angle_limit_load[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/cables_collide"] = torch.count_nonzero(
            self.cable_collision[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/drones_collide"] = torch.count_nonzero(
            self.drone_collision[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/bounding_box"] = torch.count_nonzero(
            self.body_pos_outside[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_low"] = torch.count_nonzero(
            self.falcon_fly_low[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/falcon_fly_high"] = torch.count_nonzero(
            self.falcon_fly_high[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/payload_fly_low"] = torch.count_nonzero(
            self.payload_fly_low[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/illegal_contact"] = torch.count_nonzero(
            self.illegal_contact[env_ids_tensor]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.time_out[env_ids_tensor]).item()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids_tensor])
            self.extras["log"]["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids_tensor] = 0.0

        # log metrics
        for metric_name, metric_value in self.metrics.items():
            self.extras["log"][f"Metrics/pose_command/{metric_name}"] = metric_value.mean()

        # log reward weighting as runtime parameters for TensorBoard tracking
        self.extras["log"]["Params/progress_weight"] = float(self.cfg.progress_weight)
        self.extras["log"]["Params/distance_weight"] = float(self.cfg.distance_weight)
        self.extras["log"]["Params/distance_std"] = float(self.cfg.distance_std)

        # reset the action history
        for agent in self.cfg.possible_agents:
            self.prev_actions[agent][env_ids_tensor] = 0.0
            self.actions[agent][env_ids_tensor] = 0.0

        # reset controller and motor internal state so stale filter history and
        # post-crash RPMs from the previous episode do not contaminate the new one
        for i in range(self._num_drones):
            self.geo_controllers[i].reset(env_ids_tensor)
            self._indi_controllers[i].reset(env_ids_tensor)
            self.motor_models[i].reset(env_ids_tensor)

        # reinitialise force/moment/jerk buffers to a clean hover state so that
        # the first _apply_action of the new episode starts from a known baseline
        _hover_thrust_per_rotor = (2 * 0.6017 + self.cfg.rod_mass) * 9.8066 / 8
        self._forces[env_ids_tensor] = 0.0
        self._forces[env_ids_tensor, :, 2] = _hover_thrust_per_rotor
        self._moments[env_ids_tensor] = 0.0
        self._drone_prev_acc[env_ids_tensor] = 0.0
        self._thrust_cmds[env_ids_tensor] = _hover_thrust_per_rotor

        # progress reward: clear stale prev_dist; reset_mask in _get_rewards will
        # repopulate it from the new episode's first observed distance
        self._prev_load_dist[env_ids_tensor] = 0.0

    def _reset_target_pose(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = r.uniform_(*self.cfg.goal_range["pos_x"])
        self.pose_command_w[env_ids, 1] = r.uniform_(*self.cfg.goal_range["pos_y"])
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.goal_range["pos_z"])
        # orientation goal fixed at identity (position-only task)
        self.pose_command_w[env_ids, 3] = 1.0
        self.pose_command_w[env_ids, 4:] = 0.0

    def _update_metrics(self):
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.load_position,
            self.load_orientation,
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        self.metrics["drone_to_goal_distance"] = torch.norm(self.drone_to_goal_error, dim=-1).mean(dim=-1)

        # turn on for xyz debug falcon12
        if self.num_envs == 1:
            # Extract XYZ for environment 0, drone 0 (Falcon1) and drone 1 (Falcon2)
            # f1_pos = self.drone_positions[0, 0]
            # f2_pos = self.drone_positions[0, 1]

            # Extract Payload Roll, Pitch, Yaw from the quaternion for environment 0
            payload_quat = self.load_orientation[0].unsqueeze(0)  # Shape (1, 4)
            roll, pitch, yaw = euler_xyz_from_quat(payload_quat)
            
            # Convert from radians to degrees
            roll_deg = torch.rad2deg(roll[0])
            pitch_deg = torch.rad2deg(pitch[0])
            yaw_deg = torch.rad2deg(yaw[0])

            # Print to console formatted to 3 decimal places
            # print(f"[PLAY Debug] Falcon 1 XYZ: ({f1_pos[0]:.3f}, {f1_pos[1]:.3f}, {f1_pos[2]:.3f})  |  "
            #       f"Falcon 2 XYZ: ({f2_pos[0]:.3f}, {f2_pos[1]:.3f}, {f2_pos[2]:.3f})")
            print(f"[PLAY Debug] Payload RPY: (Roll: {roll_deg:.3f}°, Pitch: {pitch_deg:.3f}°, Yaw: {yaw_deg:.3f}°)")
            
        

    def _cable_collision(
        self,
        threshold: float = 0.0,
        num_points: int = 5,
    ) -> torch.Tensor:
        """Check for collisions between cables.

        A collision is detected if the minimum Euclidean distance between any two points
        on different cables is below the threshold.
        """
        cable_bottom_pos_env = self.robot.data.body_com_state_w[
            :, self._rope_endpoint_idx, :3
        ] - self.scene.env_origins.unsqueeze(1)
        cable_directions = self.drone_positions - cable_bottom_pos_env  # (num_envs, num_cables, 3)

        # Create linearly spaced points for interpolation (num_points,)
        linspace_points = torch.linspace(0, 1, num_points, device=self.device).view(
            1, 1, num_points, 1
        )  # (1, 1, num_points, 1)

        # Compute cable points (num_envs, num_cables, num_points, 3)
        cable_points = cable_bottom_pos_env.unsqueeze(2) + linspace_points * cable_directions.unsqueeze(
            2
        )  # (num_envs, num_cables, num_points, 3)

        # Flatten cable points for easier distance calculation (num_envs, num_cables * num_points, 3)
        cable_points_flat = cable_points.view(self.num_envs, -1, 3)

        # Pairwise distance calculation
        cable_points_a = cable_points_flat.unsqueeze(2)  # (num_envs, num_points_total, 1, 3)
        cable_points_b = cable_points_flat.unsqueeze(1)  # (num_envs, 1, num_points_total, 3)
        pairwise_diff = cable_points_a - cable_points_b  # (num_envs, num_points_total, num_points_total, 3)
        pairwise_distances = torch.norm(pairwise_diff, dim=-1)  # (num_envs, num_points_total, num_points_total)

        # Mask to ignore self-distances and distances within the same cable
        num_cables = cable_bottom_pos_env.shape[1]
        points_per_cable = num_points

        # Create mask to ignore points on the same cable
        cable_indices = torch.arange(num_cables, device=self.device).repeat_interleave(
            points_per_cable
        )  # (num_points_total,)
        same_cable_mask = cable_indices.unsqueeze(0) == cable_indices.unsqueeze(
            1
        )  # (num_points_total, num_points_total)
        same_cable_mask = same_cable_mask.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )  # (num_envs, num_points_total, num_points_total)

        # Apply mask: set ignored distances to a large value
        pairwise_distances[same_cable_mask] = 1000.0

        # Find the minimum distance across all points in each environment
        min_distances, _ = torch.min(pairwise_distances.view(self.num_envs, -1), dim=-1)  # Shape: (num_envs,)

        # Check if the minimum distance is below the threshold
        is_cable_collision = min_distances < threshold  # Shape: (num_envs,)

        assert is_cable_collision.shape == (self.num_envs,)
        return is_cable_collision

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not hasattr(self, "goal_pose_visualizer"):
            # -- goal pose
            self.goal_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_goal)
            # -- current body pose
            self.body_pose_visualizer = VisualizationMarkers(self.cfg.marker_cfg_body)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(
            self.pose_command_w[:, :3] + self.scene.env_origins, self.pose_command_w[:, 3:]
        )

        # -- current body pose
        self.body_pose_visualizer.visualize(self.load_position + self.scene.env_origins, self.load_orientation)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
