# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_contrib.assets import MultirotorCfg

import isaaclab_tasks.manager_based.mr_v1.mdp as mdp


##
# Scene definition
##
# @configclass
# class MySceneCfg(InteractiveSceneCfg):
#     """Configuration for the terrain scene with a flying robot."""

#     # robots
#     robot: MultirotorCfg = MISSING

#     # lights
#     sky_light = AssetBaseCfg(
#         prim_path="/World/skyLight",
#         spawn=sim_utils.DomeLightCfg(
#             intensity=750.0,
#             texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
#         ),
#     )

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a flying robot."""

    # robots
    robot: MultirotorCfg = MISSING

    # lights
    sky_light = AssetBaseCfg(
        # ... (your existing light code) ...
    )

    # Doorway walls taking up the full plane except for a 1.0m x 2.0m gap
    wall_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Left",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 10.0, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -6.0, 5.0)) 
    )

    wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Right",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 10.0, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 6.0, 5.0))
    )

    wall_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Top",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 2.0, 6.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5))
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 0.0, 7.0))
    )

    # Collision detection
    from isaaclab.sensors import ContactSensorCfg
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0, 
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Wall_.*"] 
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target_pose = mdp.DroneUniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.DroneUniformPoseCommandCfg.Ranges(
            pos_x=(3.0, 5.0),
            pos_y=(-3.0, 3.0),
            pos_z=(1.5, 1.5),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    thrust_command = mdp.ThrustActionCfg(
        asset_name="robot",
        scale=3.0,
        offset=3.0,
        preserve_order=False,
        use_default_offset=False,
        clip={
            "back_left_prop": (0.0, 6.0),
            "back_right_prop": (0.0, 6.0),
            "front_left_prop": (0.0, 6.0),
            "front_right_prop": (0.0, 6.0),
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_link_position = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_orientation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        last_action = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-2.5, 2.5),
                "z": (1.0, 1.5),
                "yaw": (-math.pi / 6.0, math.pi / 6.0),
                "roll": (-math.pi / 6.0, math.pi / 6.0),
                "pitch": (-math.pi / 6.0, math.pi / 6.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    distance_to_goal_exp = RewTerm(
        func=mdp.distance_to_goal_exp,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.35,
            "command_name": "target_pose",
        },
    )
    # geodesic_distance = RewTerm(
    #     func=mdp.geodesic_distance_to_goal_exp,
    #     weight=25.0,
    #     params={
    #         "std": 1.75,
    #         "door_position": (2.0, 0.0, 2.0)
    #     },
    # )
    # potential_field_distance = RewTerm(
    #     func=mdp.potential_field_distance_to_goal_exp,
    #     weight=25.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "std": 1.5,
    #         "command_name": "target_pose",
    #         "door_position": (2.0, 0.0, 2.0),
    #         "forward_bonus_scale": 0.1,
    #         "forward_bonus_window": 0.5,
    #     },
    # )
    # moving_waypoint_distance = RewTerm(
    #     func=mdp.progress_to_active_target_exp,
    #     weight=25.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "std": 1.5,
    #         "command_name": "target_pose",
    #         "door_position": (2.35, 0.0, 2.0),
    #         "switch_x_enter": 0.1,
    #         "progress_scale": 1.0,
    #     },
    # )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    yaw_aligned = RewTerm(
        func=mdp.yaw_aligned,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
    )
    lin_vel_xyz_exp = RewTerm(
        func=mdp.lin_vel_xyz_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 2.0},
    )
    ang_vel_xyz_exp = RewTerm(
        func=mdp.ang_vel_xyz_exp,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 10.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude_l2 = RewTerm(func=mdp.action_l2, weight=-0.05)

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -3.0})
    crash_wall = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    )

##
# Environment configuration
##


@configclass
class TrackPositionNoObstaclesEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the state-based drone pose-control environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=25.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
