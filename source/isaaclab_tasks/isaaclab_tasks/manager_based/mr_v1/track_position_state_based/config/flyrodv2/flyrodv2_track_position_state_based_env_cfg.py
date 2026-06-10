# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from pxr import Gf, UsdGeom
import torch

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg

import isaaclab_tasks.manager_based.mr_v1.mdp as mdp


_REPO_ROOT = Path(__file__).resolve().parents[8]
# _FLYROD_USD_PATH = (
#     _REPO_ROOT
#     / "source"
#     / "isaaclab_tasks"
#     / "isaaclab_tasks"
#     / "manager_based"
#     / "mr_v1"
#     / "usd_creations"
#     / "flyrod"
#     / "flyrod.usd"
# )

_FALCON_USD_PATH = (
    _REPO_ROOT
    / "MARL_cooperative_aerial_manipulation_ext"
    / "exts"
    / "MARL_mav_carry_ext"
    / "MARL_mav_carry_ext"
    / "assets"
    / "data"
    / "AMR"
    / "falcon"
    / "falcon.usd"
)

FLYROD_THRUSTER_CFG = ThrusterCfg(
    # integration timestep used by this manager-based env (100 Hz control loop -> 0.01s)
    dt=0.01,
    thrust_range=(0.1, 10.0),
    thrust_const_range=(9.26312e-06, 1.826312e-05),
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.005, 0.005),
    torque_to_thrust_ratio=0.07,
    thruster_names_expr=[
        "Falcon_rotor_0",
        "Falcon_rotor_1",
        "Falcon_rotor_2",
        "Falcon_rotor_3",
    ],
)

FLYROD_CFG = MultirotorCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_FALCON_USD_PATH.resolve()),
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
            "Falcon_rotor_0": 200.0,
            "Falcon_rotor_1": 200.0,
            "Falcon_rotor_2": 200.0,
            "Falcon_rotor_3": 200.0,
        },
    ),
    actuators={"thrusters": FLYROD_THRUSTER_CFG},
    rotor_directions=[1, -1, 1, -1],
    allocation_matrix=None,
)


def _sample_uniform_scalar(low: float, high: float, device: torch.device) -> float:
    return float((torch.rand(1, device=device) * (high - low) + low).item())


# Module-level counter to track resets per environment and only randomize every N resets
_obstacle_reset_counts = {}


def randomize_obstacle_layout(
    env,
    env_ids: torch.Tensor,
    obstacle_cfgs: list[SceneEntityCfg],
    position_x_range: tuple[float, float],
    position_y_range: tuple[float, float],
    yaw_range: tuple[float, float],
    scale_range: tuple[float, float],
    base_z: float,
    reset_interval: int = 10,
):
    """Randomize obstacle layout every N resets (default: every 10 resets = ~10 episodes)."""
    if env_ids is None:
        return

    global _obstacle_reset_counts
    stage = get_current_stage()
    env_id_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # Track reset counts per env and only randomize every reset_interval resets
    env_ids_to_randomize = []
    for env_id in env_id_tensor.tolist():
        env_id_key = int(env_id)
        _obstacle_reset_counts[env_id_key] = _obstacle_reset_counts.get(env_id_key, 0) + 1
        
        # Only randomize if we've hit the interval
        if _obstacle_reset_counts[env_id_key] % reset_interval == 0:
            env_ids_to_randomize.append(env_id)

    if not env_ids_to_randomize:
        return

    for env_id in env_ids_to_randomize:
        env_origin = env.scene.env_origins[env_id]
        for obstacle_cfg in obstacle_cfgs:
            obstacle = env.scene[obstacle_cfg.name]
            matching_paths = sim_utils.find_matching_prim_paths(obstacle.cfg.prim_path)
            prim_path = matching_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)

            x_pos = _sample_uniform_scalar(*position_x_range, device=env.device)
            y_pos = _sample_uniform_scalar(*position_y_range, device=env.device)
            scale_value = _sample_uniform_scalar(*scale_range, device=env.device)
            # keep obstacles upright; only randomize yaw
            roll = 0.0
            pitch = 0.0
            yaw = _sample_uniform_scalar(*yaw_range, device=env.device)

            # place obstacle on the floor at fixed base_z (do not scale z)
            root_position = torch.tensor(
                [[x_pos, y_pos, base_z]], device=env.device, dtype=env_origin.dtype
            )
            root_position = root_position + env_origin.unsqueeze(0)
            root_orientation = math_utils.quat_from_euler_xyz(
                torch.tensor([roll], device=env.device),
                torch.tensor([pitch], device=env.device),
                torch.tensor([yaw], device=env.device),
            )
            obstacle.write_root_pose_to_sim(
                torch.cat((root_position, root_orientation), dim=-1),
                env_ids=torch.tensor([env_id], device=env.device),
            )
            obstacle.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([env_id], device=env.device)
            )


def root_height_above_maximum(
    env,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height.

    Note:
        This is currently only supported for flat terrains, i.e. the maximum height is in the world frame.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene with doorway walls identical to the flyrod v1 setup."""

    robot: ArticulationCfg = MISSING  # type: ignore[assignment]

    # Inverted layout:
    # wall_left/right/top are empty.
    # wall_doorway blocks the opening at (2.0, 0.0, 2.0).

    # wall_left = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Wall_Left",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.1, 10.0, 10.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    #         visible=False,
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -6.0, 5.0)),
    # )

    # wall_right = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Wall_Right",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.1, 10.0, 10.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    #         visible=False,
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 6.0, 5.0)),
    # )

    # wall_top = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Wall_Top",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.1, 2.0, 6.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    #         visible=False,
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 0.0, 7.0)),
    # )

    # wall_doorway = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Wall_Doorway",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.1, 10.0, 4.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 0.0, 2.0)),
    # )

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

    obstacle_01: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_01",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(1.4, 1.4, 3.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.35, 0.35)),
                ),
                sim_utils.CylinderCfg(
                    radius=0.88,
                    height=3.4,
                    axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.75, 0.85)),
                ),
                sim_utils.CuboidCfg(
                    size=(4.8, 0.48, 2.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.25)),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.6)),
    )

    obstacle_02: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_02",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(1.0, 3.6, 2.4),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.55, 0.85)),
                ),
                sim_utils.CylinderCfg(
                    radius=1.12,
                    height=2.8,
                    axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.55, 0.3)),
                ),
                sim_utils.CuboidCfg(
                    size=(5.4, 0.4, 1.8),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.8, 0.4)),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.2)),
    )

    obstacle_03: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_03",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(1.8, 1.8, 4.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.4, 0.8)),
                ),
                sim_utils.CylinderCfg(
                    radius=0.72,
                    height=4.4,
                    axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.8, 0.65)),
                ),
                sim_utils.CuboidCfg(
                    size=(4.0, 0.56, 2.8),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.55, 0.35)),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 2.0)),
    )

    obstacle_04: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_04",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(1.2, 1.2, 2.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.45, 0.45)),
                ),
                sim_utils.CylinderCfg(
                    radius=0.96,
                    height=2.4,
                    axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.65, 0.9)),
                ),
                sim_utils.CuboidCfg(
                    size=(6.4, 0.4, 1.8),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.35)),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 1.2)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        # sample contact less frequently to reduce CPU work
        update_period=0.02,
        history_length=1,
        track_air_time=False,
        # leave filter empty to avoid strict per-pattern count mismatch across assets
        filter_prim_paths_expr=[],
    )

    lidar_360 = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Falcon_base_link",
        # reduce raycast frequency to lower cost
        update_period=0.02,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        mesh_prim_paths=["{ENV_REGEX_NS}/Wall_Floor", "{ENV_REGEX_NS}/Wall_Env_Left", "{ENV_REGEX_NS}/Wall_Env_Right", "{ENV_REGEX_NS}/Obstacle_01", "{ENV_REGEX_NS}/Obstacle_02", "{ENV_REGEX_NS}/Obstacle_03", "{ENV_REGEX_NS}/Obstacle_04"],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            # halve the number of rays (360/45 = 8 rays)
            horizontal_res=45.0,
        ),
        # reduce max distance to cut down on hit checks
        max_distance=3.0,
        debug_vis=False,
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
        body_name="Falcon_base_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.DroneUniformPoseCommandCfg.Ranges(
            pos_x=(9.0, 9.0),
            pos_y=(-10.5, 10.5),
            pos_z=(1.5, 1.5),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Single agent outputs 8 independent thrust commands, one per rotor.
    With allocation_matrix=None each thrust is applied directly to its
    own rotor body so Falcon1 and Falcon2 are controlled independently
    through the rope dynamics.
    """

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
    """Observation specifications for the MDP.

    Tracks the rod_link (body 0 / articulation root) state plus a compact
    16-ray 360-degree LiDAR mounted at the center of the system.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        goal_distance_heading = ObsTerm(
            func=mdp.goal_distance_heading,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "target_pose"},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel_body_frame, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_body_frame, noise=Unoise(n_min=-0.05, n_max=0.05))
        lidar_distances = ObsTerm(
            func=mdp.normalized_lidar_distances,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("lidar_360")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_obstacles = EventTerm(
        func=randomize_obstacle_layout,
        mode="reset",
        params={
            "obstacle_cfgs": [
                SceneEntityCfg("obstacle_01"),
                SceneEntityCfg("obstacle_02"),
                SceneEntityCfg("obstacle_03"),
                SceneEntityCfg("obstacle_04"),
            ],
            "position_x_range": (0.0, 4.0),
            "position_y_range": (-11.0, 11.0),
            "yaw_range": (-math.pi, math.pi),
            # sizes are increased directly in the spawn cfgs; keep scale neutral
            "scale_range": (1.0, 1.0),
            # approximate center height so obstacles sit on the floor
            "base_z": 1.2,
            # only randomize obstacle positions every 10 resets (~10 episodes)
            "reset_interval": 10,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-6.0, -6.0),
                "y": (-1.0, 1.0),
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
    """Reward terms for the MDP.

    All positional rewards track the rod_link (body 0) because that is the
    payload we want to navigate through the doorway.
    """

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
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
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
    obstacle_penalty_lidar = RewTerm(
        func=mdp.obstacle_penalty_lidar,
        weight=-25.0,
        params={"sensor_cfg": SceneEntityCfg("lidar_360")},
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05})
    crash_high = DoneTerm(func=root_height_above_maximum, params={"maximum_height": 3.0})
    crash_wall = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    )


@configclass
class FlyrodV2TrackPositionEnvCfg(ManagerBasedRLEnvCfg):
    """Single-agent env cfg for rope-connected dual-drone payload transport.

    Two Falcon quadrotors carry a rod payload via compliant rope joints.
    Unlike flyrod v1 (rigid rod + allocation matrix), this config uses
    allocation_matrix=None so each rotor applies force to its own body
    and the rope dynamics handle the rest.
    """

    # default to letting the runner/CLI choose parallelism; enable physics replication
    # Set `num_envs` via CLI (e.g. `--num_envs 512`) or experiment config to avoid
    # hardcoded mismatches with the runtime replication count used by PhysX.
    scene: MySceneCfg = MySceneCfg(num_envs=MISSING, env_spacing=25.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
