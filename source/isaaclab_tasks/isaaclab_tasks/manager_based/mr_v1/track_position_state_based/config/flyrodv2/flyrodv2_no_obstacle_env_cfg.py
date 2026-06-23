# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from .flyrodv2_track_position_state_based_env_cfg import FLYROD_CFG, FlyrodV2TrackPositionEnvCfg


@configclass
class FlyrodV2NoObstacleEnvCfg(FlyrodV2TrackPositionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FLYROD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt


@configclass
class FlyrodV2NoObstacleEnvCfg_PLAY(FlyrodV2NoObstacleEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False

        # =====================================================================
        # --- OVERRIDE OBSTACLES FOR MATPLOTLIB 2D REPRESENTATION ---
        # =====================================================================
        
        # 1. Turn off the randomizer so they stay exactly where we put them
        self.events.randomize_obstacles = None

        # 1.5. NEW: Update LiDAR to stop looking for deleted obstacles!
        # Because Setups 2 and 3 only use ONE obstacle, we must remove Obstacle_02/03/04
        self.scene.lidar_360.mesh_prim_paths = [
            "{ENV_REGEX_NS}/Wall_Floor", 
            "{ENV_REGEX_NS}/Wall_Env_Left", 
            "{ENV_REGEX_NS}/Wall_Env_Right", 
            "{ENV_REGEX_NS}/Obstacle_01"
        ]

        # ---------------------------------------------------------------------
        # --- SETUP 1: THE 4 PILLARS (COMMENTED OUT) ---
        # ---------------------------------------------------------------------
        # self.scene.obstacle_01 = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle_01",
        #     spawn=sim_utils.CuboidCfg(
        #         size=(1.5, 1.5, 3.0),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.35, 0.35)),
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, -3.0, 1.5)),
        # )

        # self.scene.obstacle_02 = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle_02",
        #     spawn=sim_utils.CuboidCfg(
        #         size=(1.5, 1.5, 3.0),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.55, 0.85)),
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.0, 3.0, 1.5)),
        # )

        # self.scene.obstacle_03 = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle_03",
        #     spawn=sim_utils.CuboidCfg(
        #         size=(1.5, 1.5, 3.0),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.4, 0.8)),
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, -3.0, 1.5)),
        # )

        # self.scene.obstacle_04 = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle_04",
        #     spawn=sim_utils.CuboidCfg(
        #         size=(1.5, 1.5, 3.0),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.45, 0.45)),
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, 3.0, 1.5)),
        # )

        # ---------------------------------------------------------------------
        # --- SETUP 2: THE RIGHT GAP (ACTIVE) ---
        # 60% Wall on the Left, 40% Gap on the Right.
        # Room is 22m wide. Wall is 13.2m wide.
        # Centered at Y=-4.4, it blocks from the left wall (Y=-11.0) to Y=2.2.
        # Gap is wide open from Y=2.2 to the right wall (Y=11.0).
        # ---------------------------------------------------------------------
        # self.scene.obstacle_01 = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Obstacle_01",
        #     spawn=sim_utils.CuboidCfg(
        #         size=(1.5, 13.2, 3.0),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.35, 0.35)),
        #     ),
        #     # CHANGED X FROM -2.0 TO 2.0 (Closer to the X=9.0 finish line)
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, -4.4, 1.5)),
        # )

        # # Ensure no other obstacles spawn and crash the LiDAR
        # self.scene.obstacle_02 = None
        # self.scene.obstacle_03 = None
        # self.scene.obstacle_04 = None


        # ---------------------------------------------------------------------
        # --- SETUP 3: THE LEFT GAP (COMMENTED OUT) ---
        # 60% Wall on the Right, 40% Gap on the Left.
        # Centered at Y=4.4, it blocks from the right wall (Y=11.0) to Y=-2.2.
        # Gap is wide open from the left wall (Y=-11.0) to Y=-2.2.
        # ---------------------------------------------------------------------
        self.scene.obstacle_01 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle_01",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, 13.2, 3.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.35, 0.35)),
            ),
            # CHANGED X FROM -2.0 TO 2.0 (Closer to the X=9.0 finish line)
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 4.4, 1.5)),
        )
        
        self.scene.obstacle_02 = None
        self.scene.obstacle_03 = None
        self.scene.obstacle_04 = None