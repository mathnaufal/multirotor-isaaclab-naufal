# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

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
