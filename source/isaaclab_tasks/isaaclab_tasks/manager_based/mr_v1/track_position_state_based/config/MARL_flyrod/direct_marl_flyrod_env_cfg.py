# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from copy import deepcopy

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.envs import ViewerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Use the new aggressive multi-agent multirotor asset
from isaaclab_assets.robots.flyrodv2_2 import FLYRODV2_2_CFG
# from isaaclab_assets.robots.flyrod_MARL import FLYROD_MARL_CFG


@configclass
class EventCfg:
    """Events for the hovering task.

    Resetting states on resets, disturbances, etc.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-6.0, -6.0),
                "y": (-1.0, 1.0),
                "z": (1.5, 1.5),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (math.pi, math.pi),
                # "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore[arg-type]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["rod_link"]),
            "mass_distribution_params": (0.5, 0.6),
            "operation": "abs",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

@configclass
class EventCfg_PLAY(EventCfg):
    """Play-only events for the hovering task."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-6.0, -6.0),
                "y": (-1.0, 1.0),
                "z": (1.5, 1.5),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (math.pi, math.pi),
                # "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # Fix mass and CoM at nominal values so PLAY episodes are fully deterministic
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore[arg-type]
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["rod_link"]),
            "mass_distribution_params": (1.0, 1.0),  # scale by 1.0 → no change
            "operation": "scale",
        },
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a flying robot."""

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # walls
    wall_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Left",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 10.0, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -6.0, 5.0)),
    )

    wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Right",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 10.0, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),

        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 6.0, 5.0)),
    )

    wall_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Top",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 2.0, 6.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 0.0, 7.0)),
    )

    floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Floor",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 22.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 0.0, -0.05)),
    )

    wall_env_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Env_Left",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 0.1, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -11.0, 5.0)),
    )

    wall_env_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Env_Right",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 0.1, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 11.0, 5.0)),
    )

    # LiDAR disabled by request.
    # Keep these definitions commented for quick re-enable later if needed.
    lidar_falcon1: MultiMeshRayCasterCfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Flyrodv2/Falcon1_base_link",
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Wall_.*"],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=15.0,
        ),
        max_distance=10.0,
        debug_vis=False,
    )

    lidar_falcon2 = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Flyrodv2/Falcon2_base_link",
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Wall_.*"],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=15.0,
        ),
        max_distance=10.0,
        debug_vis=False,
    )


@configclass
class DirectMARLFlyrodEnvCfg(DirectMARLEnvCfg):
    """Direct MARL config for flyrodv2 using ACCBR control, matching marl_hover architecture."""

    # control mode
    control_mode = "ACCBR"  # ACCBR or geometric

    # env
    decimation = 3
    episode_length_s = 8.0

    # history of observations
    partial_obs = True
    history_len = 3

    possible_agents = ["falcon1", "falcon2"]
    num_drones = len(possible_agents)

    num_lidar_rays = 0

    if control_mode == "ACCBR":
        action_dim_accbr = 5  # [lin_acc_x, lin_acc_y, lin_acc_z, roll_rate, pitch_rate]
        action_spaces = {"falcon1": action_dim_accbr, "falcon2": action_dim_accbr}
        if partial_obs:
            # load_pos(3)+load_mat(9)+one_hot(3)+drone_pos(3)+drone_rot(9)+drone_vel(3)+drone_ang_vel(3)+goal_pos_err(3)+drone_to_goal(3)=39.
            obs_dim_accbr = 39 * history_len
        else:
            # load_pos(3)+load_mat(9)+load_vel(3)+load_ang_vel(3)+one_hot(3)+
            # drone_pos_all(6)+drone_rot_all(18)+drone_vel_all(6)+drone_ang_vel_all(6)+
            # goal_pos_err(3)+drone_to_goal_all(6)=66
            obs_dim_accbr = 66
        observation_spaces = {
            "falcon1": obs_dim_accbr,
            "falcon2": obs_dim_accbr,
        }
        # state: rod(3+9+3+3=18) + 2×falcon(2×(3+9+3+3)=36) + goal(3) = 57
        state_space = 57

    # simulation — 300 Hz physics, 100 Hz policy
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 300.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="multiply",
        #     restitution_combine_mode="multiply",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        # ),
        # physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )

    # robot (use new flyrodv2_2 multirotor-style asset)
    robot_cfg = FLYRODV2_2_CFG.replace(prim_path="/World/envs/env_.*/Flyrodv2")  # type: ignore[attr-defined]
    robot_cfg.spawn.activate_contact_sensors = True

    # contact sensor
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Flyrodv2/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/Wall_.*"],
    )
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*")
    contact_sensor_threshold = 0.1

    # scene
    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=25.0, replicate_physics=True)

    # body name patterns for find_bodies
    # Match only the actual base-link bodies, not the `_inertia` collision bodies.
    falcon_names = r"Falcon[12]_base_link$"
    # rotor names
    falcon_rotor_names = "Falcon[12]_rotor_.*"
    # payload name
    payload_name = "rod_link"
    # rope/cable name and properties
    rope_name = "rope_.*_link"
    rope_stiffness: float = 1000.0  # [N/m] — spring stiffness of cable joints
    rope_damping: float = 50.0  # [N*s/m] — damping of cable joints
    cable_angle_limits_drone = 0.0  # cos(angle) limits
    cable_angle_limits_payload = -math.sqrt(2) / 2  # cos(angle) limits
    cable_collision_threshold = 0.2
    cable_collision_num_points = 10

    # low level control
    low_level_decimation: int = 1
    max_thrust_pp: float = 10.0  # matches flyrod allocation matrix thrust_range max

    # action-to-physics scaling: affine map from raw policy outputs to physical setpoints
    # policy outputs are unbounded Gaussian (clip_actions=false); these multipliers convert
    # them into physically meaningful units so the controller receives reasonable commands.
    acc_scale: float = 1.5 * 9.8066  # [m/s²] — output ±1 → ±1.5 g lin_acc command
    body_rate_scale: float = 2.0 * math.pi  # [rad/s] — output ±1 → ±2π rad/s roll/pitch rate

    # rewards
    # Task reward weights are raised well above regularizers so that progress toward the
    # goal dominates the per-step return; otherwise the agents farm the exp(-cost) "survival"
    # bonuses by hovering. Regularizers are disabled until the agents learn to fly forward.
    progress_weight = 25.0
    distance_weight = 25.0
    distance_std = 1.5
    upright_orient_weight = 0.5 # 2.0
    action_smoothness_weight = 0.0
    body_rate_penalty_weight = 0.0
    force_penalty_weight = 0.0

    rope_vertical_weight = 0.0
    rod_half_length = 0.275  # attach offsets in rod local frame: drone1 at +x, drone2 at -x

    velocity_sync_weight = 0.0

    # payload speed soft-cap disabled during exploration so the agents can carry momentum
    speed_limit_weight = 0.0
    payload_v_max = 0.5  # [m/s]

    # cross-track funnel: penalise Y deviation from door centre (y=0)
    cross_track_weight: float = 0.0

    # constant time penalty: makes hovering painful so the agent seeks the goal
    time_penalty_weight: float = 1.0

    # one-shot penalty applied on the step a non-timeout termination fires
    crash_penalty: float = 5.0

    # goal terms: target position only for the rod (no orientation goal)
    goal_range = {
        "pos_x": (9.0, 9.0),
        "pos_y": (0.0, 0.0),
        "pos_z": (1.5, 1.5),
    }
    range_curriculum_steps = 7500

    make_quat_unique_command = False

    # ── sim-to-real: observation noise ──────────────────────────────────────
    # Mild additive Gaussian noise applied only to the observation tensors;
    # reward and termination computations always use the clean physics buffers.
    # Gap 1: position + velocity noise (IMU drift / MOCAP jitter)
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    ang_vel_noise_std: float = 0.0
    # Gap 2: rotation matrix noise — proxy for 6-DOF pose estimation error on payload
    orient_noise_std: float = 0.0

    # debug visualization
    debug_vis: bool = True
    if debug_vis:
        marker_cfg_goal = deepcopy(FRAME_MARKER_CFG)
        marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore[attr-defined]
        marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"

        marker_cfg_body = deepcopy(FRAME_MARKER_CFG)
        marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore[attr-defined]
        marker_cfg_body.prim_path = "/Visuals/Command/body_pose"

    # terminations
    fly_low_threshold: float = 0.0
    drone_collision_threshold: float = 0.6
    bounding_box_threshold: float = 20.0
    contact_sensor_threshold: float = 0.1
    # rope-angle / cable-collision terminations disabled during exploration
    enable_cable_terminations: bool = False

    events = EventCfg()


@configclass
class DirectMARLFlyrodEnvCfg_PLAY(DirectMARLFlyrodEnvCfg):
    """Play variant for Direct MARL flyrodv2 task."""

    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # No observation noise during evaluation — clean rollouts for analysis
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    orient_noise_std: float = 0.0
    ang_vel_noise_std: float = 0.0

    # No initial velocity perturbation at play-time reset
    events: EventCfg = EventCfg_PLAY()

    # Viewer option for PLAY: choose camera focused on the robot or an upper-right overview
    # Options: "default", "robot", "upper_right"
    viewer_option: str = "upper_right"
    if viewer_option == "robot":
        viewer: ViewerCfg = ViewerCfg(
            origin_type="asset_root",
            env_index=0,
            asset_name="robot",
            eye=(4.0, 4.0, 4.0),
            lookat=(0.0, 0.0, 0.0),
        )
    elif viewer_option == "upper_right":
        viewer: ViewerCfg = ViewerCfg(
            origin_type="world",
            env_index=0,
            eye=(2.0, 10.0, 15.0),
            lookat=(2.0, 0.0, 0.0),
        )
    else:
        viewer: ViewerCfg = ViewerCfg()