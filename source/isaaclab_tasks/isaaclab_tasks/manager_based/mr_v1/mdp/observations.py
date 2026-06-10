# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.jit

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from isaaclab_contrib.assets import Multirotor

if TYPE_CHECKING:
    pass

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape

"""
State.
"""


def base_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the base roll and pitch in the simulation world frame.

    Parameters:
        env: Manager-based environment providing the scene and tensors.
        asset_cfg: Scene entity config pointing to the target robot (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 2). Column 0 is roll, column 1 is pitch.
        Values are radians normalized to [-pi, pi], expressed in the world frame.

    Notes:
        - Euler angles are computed from asset.data.root_quat_w using XYZ convention.
        - Only roll and pitch are returned; yaw is omitted.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = math_utils.wrap_to_pi(roll)
    pitch = math_utils.wrap_to_pi(pitch)

    return torch.cat((roll.unsqueeze(-1), pitch.unsqueeze(-1)), dim=-1)


"""
Commands.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_drone_commands(
    env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Generate a body-frame direction and distance to the commanded position.

    This observation reads a command from env.command_manager identified by command_name,
    interprets its first three components as a target position in the world frame, and
    returns:
        [dir_x, dir_y, dir_z, distance]
    where dir_* is the unit vector from the current body origin to the target, expressed
    in the multirotor body (root link) frame, and distance is the Euclidean separation.

    Parameters:
        env: Manager-based RL environment providing scene and command manager.
        command_name: Name of the command term to query from the command manager.
        asset_cfg: Scene entity config for the multirotor asset (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 4) with body-frame unit direction (3) and distance (1).

    Frame conventions:
        - Current position is asset.data.root_pos_w relative to env.scene.env_origins (world frame).
        - Body orientation uses asset.data.root_link_quat_w to rotate world vectors into the body frame.

    Assumptions:
        - env.command_manager.get_command(command_name) returns at least three values
          representing a world-frame target position per environment.
        - A small epsilon (1e-8) is used to guard against zero-length direction vectors.
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    current_position_w = asset.data.root_pos_w - env.scene.env_origins
    command = cast(ManagerBasedRLEnv, env).command_manager.get_command(command_name)
    current_position_b = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, command[:, :3] - current_position_w)
    current_position_b_dir = current_position_b / (torch.norm(current_position_b, dim=-1, keepdim=True) + 1e-8)
    current_position_b_mag = torch.norm(current_position_b, dim=-1, keepdim=True)
    return torch.cat((current_position_b_dir, current_position_b_mag), dim=-1)


"""
Sensors.
"""


def lidar_distances(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Euclidean distances from a ray-caster sensor origin to each ray hit point.

    Missed rays (hit position = inf) and any NaN values are clamped to the
    sensor's configured max_distance so the observation is always finite.

    Args:
        sensor_cfg: SceneEntityCfg identifying the RayCaster sensor in the scene.

    Returns:
        torch.Tensor: Shape (num_envs, num_rays).
    """
    sensor = cast(RayCaster, env.scene.sensors[sensor_cfg.name])
    max_dist = float(sensor.cfg.max_distance)
    # pos_w: (N, 3), ray_hits_w: (N, B, 3)
    distances = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    # Missed rays return inf; uninitialized sensor pos can produce NaN — clamp both.
    return torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)


def goal_distance_heading(
    env: ManagerBasedEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the goal distance and heading error in the robot body frame."""
    asset: Multirotor = env.scene[asset_cfg.name]
    command = cast(ManagerBasedRLEnv, env).command_manager.get_command(command_name)

    current_position_w = asset.data.root_link_pos_w - env.scene.env_origins
    goal_vector_w = command[:, :3] - current_position_w
    goal_vector_b = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, goal_vector_w)

    distance = torch.norm(goal_vector_b, dim=-1, keepdim=True)
    heading = torch.atan2(goal_vector_b[:, 1], goal_vector_b[:, 0]).unsqueeze(-1)
    heading = math_utils.wrap_to_pi(heading)
    return torch.cat((distance, heading), dim=-1)


def base_lin_vel_body_frame(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the root linear velocity expressed in the body frame."""
    asset: Multirotor = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def base_ang_vel_body_frame(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the root angular velocity expressed in the body frame."""
    asset: Multirotor = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def normalized_lidar_distances(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return LiDAR distances normalized to [0, 1]."""
    sensor = cast(RayCaster, env.scene.sensors[sensor_cfg.name])
    max_dist = float(sensor.cfg.max_distance)
    distances = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    distances = torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)
    return torch.clamp(distances / max_dist, 0.0, 1.0)
