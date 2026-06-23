# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

"""
Drone control rewards.
"""


def distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward the distance to a goal position using an exponential kernel.

    This reward computes an exponential falloff of the squared Euclidean distance
    between the commanded target position and the asset (robot) root position.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; larger values
            produce a gentler falloff.
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment reward
        values in [0, 1], with 1.0 when the position error is zero.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    # compute the error
    position_error_square = torch.sum(torch.square(target_position_w - current_position), dim=1)
    return torch.exp(-position_error_square / std**2)


def distance_to_goal_falcon_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Falcon1_odometry_sensor_link"),
    std: float = 1.0,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward Falcon body distance to goal using the same kernel as distance_to_goal_exp."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()

    if isinstance(asset_cfg.body_ids, slice):
        raise ValueError("asset_cfg must resolve to exactly one body. Set body_names in env_cfg reward params.")
    if isinstance(asset_cfg.body_ids, list):
        body_id = int(asset_cfg.body_ids[0])
    else:
        body_id = int(asset_cfg.body_ids)

    body_position_w = asset.data.body_pos_w[:, body_id] - env.scene.env_origins

    position_error_square = torch.sum(torch.square(target_position_w - body_position_w), dim=1)

    return torch.exp(-position_error_square / std**2)


def progress_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
    progress_scale: float = 1.0,
) -> torch.Tensor:
    """Reward per-step progress toward the target position.

    Computes the reduction in Euclidean distance to the commanded target between
    consecutive steps. Positive values indicate moving closer to the goal.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3]
    current_position = asset.data.root_pos_w - env.scene.env_origins
    current_dist = torch.norm(target_position_w - current_position, p=2, dim=1)

    # ==========================================================
    # --- NEW: PLAY MODE DEBUG PRINT (SINGLE AGENT) ---
    # ==========================================================
    if env.num_envs == 1:
        try:
            # Find the exact body index dynamically for the single Falcon
            f_idx, _ = asset.find_bodies(".*Falcon.*")
            
            if len(f_idx) > 0:
                # Grab the world position and subtract the environment origin
                f_pos = asset.data.body_pos_w[0, f_idx[0]] - env.scene.env_origins[0]
                print(f"[PLAY Debug] Falcon XYZ: ({f_pos[0]:.3f}, {f_pos[1]:.3f}, {f_pos[2]:.3f})")
            else:
                # Fallback just in case the body names don't match
                root_pos = current_position[0]
                print(f"[PLAY Debug] Payload Root XYZ: ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")
        except Exception:
            pass
    # ==========================================================

    progress_state: dict[str, Any] = env.__dict__.setdefault("_simple_prog_state", {})
    prev_dist = progress_state.get("prev_dist")
    if prev_dist is None or prev_dist.shape[0] != env.num_envs:
        prev_dist = current_dist.clone()

    reset_mask = env.episode_length_buf == 0
    prev_dist = torch.where(reset_mask, current_dist, prev_dist)

    progress_reward = progress_scale * (prev_dist - current_dist)
    progress_state["prev_dist"] = current_dist.detach().clone()

    return progress_reward

def obstacle_penalty_lidar(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar_360"),
    warning_distance: float = 1.0,
    critical_distance: float = 0.3,
) -> torch.Tensor:
    """Soft obstacle-avoidance penalty from the closest LiDAR return.

    The penalty is zero when all rays are at or beyond `warning_distance`.
    It ramps up smoothly inside that radius and becomes much steeper once the
    closest return drops below `critical_distance`.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    max_dist = float(sensor.cfg.max_distance)

    distances = torch.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    distances = torch.nan_to_num(distances, nan=max_dist, posinf=max_dist, neginf=0.0)
    closest_distance = torch.min(distances, dim=1).values

    span = max(warning_distance - critical_distance, 1e-6)
    proximity = torch.clamp((warning_distance - closest_distance) / span, min=0.0, max=1.0)

    # Smooth, continuous shaping: gentle near 1 m, sharp as the ray approaches 0.3 m.
    penalty = proximity**2 + 2.0 * proximity**4

    return penalty


def distance_to_goal_switch_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
    door_position: tuple[float, float, float] = (2.0, 0.0, 1.0),
    switch_x_enter: float = 0.1,
    progress_scale: float = 1.0,
) -> torch.Tensor:
    """Distance reward with hard target switching from door to goal.

    Before crossing the door plane, the target is the fixed doorway point.
    After crossing ``door_x + switch_x_enter``, the target switches to
    ``target_pose``.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    door_pos = torch.tensor(
        door_position, device=current_position.device, dtype=current_position.dtype
    )
    door_x = door_pos[0]
    drone_x = current_position[:, 0]

    passed_door = drone_x >= (door_x + switch_x_enter)
    door_target = door_pos.unsqueeze(0).expand_as(target_position_w)
    active_target = torch.where(passed_door.unsqueeze(1), target_position_w, door_target)

    active_dist = torch.norm(active_target - current_position, p=2, dim=1)

    # Per-step progress keeps a usable signal even when the goal is far.
    progress_state: dict[str, Any] = env.__dict__.setdefault("_switch_prog_state", {})
    prev_dist = progress_state.get("prev_dist")
    if prev_dist is None or prev_dist.shape[0] != env.num_envs:
        prev_dist = active_dist.clone()

    reset_mask = env.episode_length_buf == 0
    prev_dist = torch.where(reset_mask, active_dist, prev_dist)

    progress_reward = progress_scale * (prev_dist - active_dist)
    progress_state["prev_dist"] = active_dist.detach().clone()

    # Exponential term improves precision near the active target.
    proximity_reward = torch.exp(-(active_dist**2) / std**2)

    return progress_reward + proximity_reward


def prog_distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
    command_name: str = "target_pose",
    door_position: tuple[float, float, float] = (2.0, 0.0, 1.0),
    switch_scale: float = 0.25,
    progress_scale: float = 1.0,
    door_half_width: float = 0.6,
    door_half_height: float = 0.6,
    door_pass_bonus: float = 2.0,
    goal_radius: float = 3.0,
) -> torch.Tensor:
    """Reward progress using a doorway-routed moving waypoint.

    This avoids far-goal saturation by using per-step distance improvement as
    the primary signal and only using exponential precision shaping near goal.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    door_pos = torch.tensor(
        door_position, device=current_position.device, dtype=current_position.dtype
    )
    door_x, door_y, door_z = door_pos[0], door_pos[1], door_pos[2]
    drone_x = current_position[:, 0]

    transition_scale = torch.clamp(
        torch.as_tensor(switch_scale, device=current_position.device, dtype=current_position.dtype),
        min=1e-3,
    )
    blend_to_door = torch.sigmoid((door_x - drone_x) / transition_scale).unsqueeze(1)

    # Dynamic waypoint: door before crossing, final goal after crossing.
    moving_waypoint = blend_to_door * door_pos.unsqueeze(0) + (1.0 - blend_to_door) * target_position_w

    # Primary signal: per-step progress to active waypoint.
    progress_state: dict[str, Any] = env.__dict__.setdefault("_mr_v1_prog_state", {})
    active_dist = torch.norm(moving_waypoint - current_position, p=2, dim=1)

    prev_dist = progress_state.get("prev_dist")
    if prev_dist is None or prev_dist.shape[0] != env.num_envs:
        prev_dist = active_dist.clone()

    reset_mask = env.episode_length_buf == 0
    prev_dist = torch.where(reset_mask, active_dist, prev_dist)
    progress_reward = progress_scale * (prev_dist - active_dist)
    progress_state["prev_dist"] = active_dist.detach().clone()

    # Bonus for passing through the doorway opening.
    in_door_y = (torch.abs(current_position[:, 1] - door_y) <= door_half_width).float()
    in_door_z = (torch.abs(current_position[:, 2] - door_z) <= door_half_height).float()
    passed_plane = (drone_x >= door_x).float()
    door_pass_reward = door_pass_bonus * passed_plane * in_door_y * in_door_z

    # Precision shaping only when close enough to final goal.
    goal_dist = torch.norm(target_position_w - current_position, p=2, dim=1)
    near_goal = (goal_dist < goal_radius).float()
    goal_precision_reward = near_goal * torch.exp(-(goal_dist**2) / std**2)

    return progress_reward + door_pass_reward + goal_precision_reward


def progress_to_active_target_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
    door_position: tuple[float, float, float] = (2.0, 0.0, 1.0),
    switch_x_enter: float = 0.1,
    progress_scale: float = 1.0,
) -> torch.Tensor:
    """Reward progress to an active target with hard if/else phase switching.

    Uses only x-position to switch from door target to final goal target.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    door_pos = torch.tensor(
        door_position, device=current_position.device, dtype=current_position.dtype
    )
    door_x = door_pos[0]

    drone_x = current_position[:, 0]
    passed_door = drone_x >= (door_x + switch_x_enter)

    door_target = door_pos.unsqueeze(0).expand_as(target_position_w)
    active_target = torch.where(passed_door.unsqueeze(1), target_position_w, door_target)
    active_dist = torch.norm(active_target - current_position, p=2, dim=1)

    progress_state: dict[str, Any] = env.__dict__.setdefault("_phase_prog_state", {})
    prev_dist = progress_state.get("prev_dist")
    if prev_dist is None or prev_dist.shape[0] != env.num_envs:
        prev_dist = active_dist.clone()

    reset_mask = env.episode_length_buf == 0
    prev_dist = torch.where(reset_mask, active_dist, prev_dist)

    progress_reward = progress_scale * (prev_dist - active_dist)
    proximity_reward = torch.exp(-(active_dist**2) / std**2)

    progress_state["prev_dist"] = active_dist.detach().clone()

    return progress_reward + proximity_reward

def geodesic_distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
    door_position: tuple[float, float, float] = (2.0, 0.0, 1.0),
    door_half_width: float = 0.5,
    door_threshold: float = 0.25,
) -> torch.Tensor:
    """Reward progress to goal with a smooth door-aware geodesic distance.
    The reward blends two path lengths:
    1) a path that goes via the doorway center, and
    2) direct distance to the goal.
    This avoids the hard phase switch that can create a local optimum where the
    drone hovers right before the door.
    Note:
        ``door_threshold`` is used as a sigmoid transition scale (in meters),
        not a hard distance cutoff.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    door_pos = torch.tensor(
        door_position, device=current_position.device, dtype=current_position.dtype
    )
    door_x = door_pos[0]
    drone_x = current_position[:, 0]

    dist_drone_to_door = torch.norm(door_pos - current_position, p=2, dim=1)
    dist_drone_to_goal = torch.norm(target_position_w - current_position, p=2, dim=1)
    dist_door_to_goal  = torch.norm(target_position_w - door_pos, p=2, dim=1)

    before_door = (drone_x < door_x)

    # Approximate shortest feasible path before the wall as going via the door.
    via_door_distance = dist_drone_to_door + dist_door_to_goal

    # Smooth transition around the door plane to avoid reward discontinuities.
    # door_threshold is the sigmoid scale: smaller => sharper handoff.
    transition_scale = torch.clamp(
        torch.as_tensor(door_threshold, device=current_position.device, dtype=current_position.dtype),
        min=1e-3,
    )
    blend_to_via_door = torch.sigmoid((door_x - drone_x) / transition_scale)

    # Squared-exponential blended reward (geodesic signal)
    via_door_reward    = torch.exp(-(via_door_distance**2) / std**2)
    direct_goal_reward = torch.exp(-(dist_drone_to_goal**2) / std**2)
    reward = blend_to_via_door * via_door_reward + (1.0 - blend_to_via_door) * direct_goal_reward

    # Linear-kernel convergence bonus — strong gradient near goal, only after door
    # helps drone precisely converge on goal independent of std tuning
    convergence_reward = torch.exp(-dist_drone_to_goal / std) * (1.0 - before_door.float())

    # Alignment bonus — only before wall, only if misaligned laterally
    door_lateral_offset = torch.norm(
        current_position[:, 1:3] - door_pos[1:3], p=2, dim=1
    )
    needs_alignment = (door_lateral_offset > door_half_width).float()
    alignment_bonus = (
        0.2 * before_door.float() * needs_alignment
        * torch.exp(-door_lateral_offset / door_half_width)
    )

    # # For debug
    # phase1_like = torch.mean(blend_to_via_door).item()
    # phase2_like = torch.mean(1.0 - blend_to_via_door).item()
    # transition_frac = torch.mean(
    #     ((blend_to_via_door > 0.1) & (blend_to_via_door < 0.9)).float()
    # ).item()
    # past_door_frac = torch.mean((drone_x > door_x).float()).item()
    # print(
    #     f"[geo] p1_like={phase1_like:.3f} p2_like={phase2_like:.3f} "
    #     f"trans={transition_frac:.3f} past_door={past_door_frac:.3f}"
    # )

    return reward + 0.5 * convergence_reward + alignment_bonus

def potential_field_distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
    door_position: tuple[float, float, float] = (2.0, 0.0, 1.0),
    forward_bonus_scale: float = 0.1,
    forward_bonus_window: float = 0.5,
) -> torch.Tensor:
    """Reward progress with a doorway-routed potential field.

    Uses a smooth blend between a via-door path and direct-to-goal distance.
    This preserves a continuous gradient while encouraging routing through the
    doorway before crossing the wall plane.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_position_w = command[:, :3].clone()
    current_position = asset.data.root_pos_w - env.scene.env_origins

    door_pos = torch.tensor(
        door_position, device=current_position.device, dtype=current_position.dtype
    )
    door_x = door_pos[0]
    drone_x = current_position[:, 0]

    dist_drone_to_door = torch.norm(door_pos - current_position, p=2, dim=1)
    dist_door_to_goal = torch.norm(target_position_w - door_pos, p=2, dim=1)
    dist_drone_to_goal = torch.norm(target_position_w - current_position, p=2, dim=1)

    # Approximate shortest feasible path before the wall as going via the door.
    via_door_distance = dist_drone_to_door + dist_door_to_goal

    # Smoothly hand off to direct goal distance near/after the door plane.
    # Fixed transition scale keeps this reward self-contained.
    transition_scale = torch.as_tensor(0.25, device=current_position.device, dtype=current_position.dtype)
    blend_to_via_door = torch.sigmoid((door_x - drone_x) / transition_scale)
    path_distance = blend_to_via_door * via_door_distance + (1.0 - blend_to_via_door) * dist_drone_to_goal

    # Small directional bonus to help cross the door plane consistently.
    drone_vel_x = asset.data.root_lin_vel_w[:, 0]
    forward_bonus = (
        forward_bonus_scale
        * torch.clamp(drone_vel_x, min=0.0)
        * (drone_x < door_x + forward_bonus_window).float()
    )

    return torch.exp(-path_distance**2 / std**2) + forward_bonus

def ang_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize angular velocity magnitude with an exponential kernel.

    This reward computes exp(-||omega||^2 / std^2) where omega is the body-frame
    angular velocity of the asset. It is useful for encouraging low rotational
    rates.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; controls
            sensitivity to angular velocity magnitude.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero angular velocity.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of angular velocity (all axes)
    ang_vel_squared = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)

    return torch.exp(-ang_vel_squared / std**2)


def lin_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize linear velocity magnitude with an exponential kernel.

    Computes exp(-||v||^2 / std^2) where v is the asset's linear velocity in
    world frame. Useful for encouraging the agent to reduce translational speed.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero linear velocity.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of linear velocity (all axes)
    lin_vel_squared = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)

    return torch.exp(-lin_vel_squared / std**2)


def yaw_aligned(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.5
) -> torch.Tensor:
    """Reward alignment of the vehicle's yaw to zero using an exponential kernel.

    The function extracts the yaw (rotation about Z) from the world-frame root
    quaternion and computes exp(-yaw^2 / std^2). This encourages heading
    alignment to a zero-yaw reference.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; smaller values
            make the reward more sensitive to yaw deviations.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        perfect yaw alignment (yaw == 0).
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # extract yaw from current orientation
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    # normalize yaw to [-pi, pi] (target is 0)
    yaw = math_utils.wrap_to_pi(yaw)

    # return exponential reward (1 when yaw=0, approaching 0 when rotated)
    return torch.exp(-(yaw**2) / std**2)
