# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg

import isaaclab_tasks.manager_based.mr_v1.mdp as mdp


# Build absolute path to the flyrod USD generated in this repository.
_REPO_ROOT = Path(__file__).resolve().parents[8]
_FLYROD_USD_PATH = _REPO_ROOT / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "mr_v1" / "usd_creations" / "flyrod" / "flyrod.usd"

FLYROD_THRUSTER_CFG = ThrusterCfg(
    # integration timestep used by this manager-based env (100 Hz control loop -> 0.01s)
    dt=0.01,
    thrust_range=(0.1, 10.0),
    thrust_const_range=(9.26312e-06, 1.826312e-05),
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.005, 0.005),
    torque_to_thrust_ratio=0.07,
    thruster_names_expr=[
        "Falcon1_rotor_0",
        "Falcon1_rotor_1",
        "Falcon1_rotor_2",
        "Falcon1_rotor_3",
        "Falcon2_rotor_0",
        "Falcon2_rotor_1",
        "Falcon2_rotor_2",
        "Falcon2_rotor_3",
    ],
)

FLYROD_CFG = MultirotorCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_FLYROD_USD_PATH.resolve()),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        rps={
            "Falcon1_rotor_0": 200.0,
            "Falcon1_rotor_1": 200.0,
            "Falcon1_rotor_2": 200.0,
            "Falcon1_rotor_3": 200.0,
            "Falcon2_rotor_0": 200.0,
            "Falcon2_rotor_1": 200.0,
            "Falcon2_rotor_2": 200.0,
            "Falcon2_rotor_3": 200.0,
        },
    ),
    actuators={"thrusters": FLYROD_THRUSTER_CFG},
    rotor_directions=[1, -1, 1, -1, 1, -1, 1, -1],
    # 6x8 wrench allocation matrix for two rigidly connected quadrotors.
    allocation_matrix=[
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.575, 0.425, 0.425, 0.575, -0.425, -0.575, -0.575, -0.425],
        [-0.075, -0.075, 0.075, 0.075, -0.075, -0.075, 0.075, 0.075],
        [-0.07, 0.07, -0.07, 0.07, -0.07, 0.07, -0.07, 0.07],
    ],
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a flying robot."""

    robot: ArticulationCfg = MISSING  # type: ignore[assignment]

    # Doorway walls taking up the full plane except for a 2.0m x 4.0m gap.
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
            size=(20.0, 24.0, 0.1),
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -12.0, 5.0)),
    )

    wall_env_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall_Env_Right",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 0.1, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 12.0, 5.0)),
    )

    # Collision detection against doorway walls.
    from isaaclab.sensors import ContactSensorCfg

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Wall_.*"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target_pose = mdp.DroneUniformPoseCommandCfg(
        asset_name="robot",
        body_name="rod_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.DroneUniformPoseCommandCfg.Ranges(
            pos_x=(9.0, 9.0),
            pos_y=(-1.0, 1.0),
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
        scale=2.0,
        offset=2.2,
        preserve_order=True,
        use_default_offset=False,
        clip={".*": (0.1, 10.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_link_position = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_orientation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        last_action = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-6.0, -6.0),
                "y": (-2.0, 2.0),
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

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (-6.0, -6.0),
    #             "y": (-2.0, 2.0),
    #             "z": (1.0, 1.5),
    #             "yaw": ((math.pi / 2.0) - 0.6, (math.pi / 2.0) + 0.6),
    #             "roll": (-math.pi / 5.0, math.pi / 5.0),
    #             "pitch": (-math.pi / 5.0, math.pi / 5.0),
    #         },
    #         "velocity_range": {
    #             "x": (-0.2, 0.2),
    #             "y": (-0.2, 0.2),
    #             "z": (-0.2, 0.2),
    #             "roll": (-0.3, 0.3),
    #             "pitch": (-0.3, 0.3),
    #             "yaw": (-0.5, 0.5),
    #         },
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "target_pose",
            "progress_scale": 1.0,
        },
    )
    distance_to_goal_exp = RewTerm(
        func=mdp.distance_to_goal_exp,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "target_pose",
        },
    )
    # falcon1_distance_to_goal_falcon_exp = RewTerm(
    #     func=mdp.distance_to_goal_falcon_exp,
    #     weight=15,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Falcon1_odometry_sensor_link"),
    #         "std": 1.5,
    #         "command_name": "target_pose",
    #     },
    # )
    # falcon2_distance_to_goal_falcon_exp = RewTerm(
    #     func=mdp.distance_to_goal_falcon_exp,
    #     weight=10,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Falcon2_odometry_sensor_link"),
    #         "std": 1.5,
    #         "command_name": "target_pose",
    #     },
    # )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # yaw_aligned = RewTerm(
    #     func=mdp.yaw_aligned,
    #     weight=2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
    # )
    lin_vel_xyz_exp = RewTerm(
        func=mdp.lin_vel_xyz_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 2.0},
    )
    ang_vel_xyz_exp = RewTerm(
        func=mdp.ang_vel_xyz_exp,
        weight=1.0,
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
    crash = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05})
    crash_wall = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    )


@configclass
class FlyrodTrackPositionNoObstaclesEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the state-based flyrod pose-control environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=25.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
