# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the flyrod platform."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

repo_root = Path(__file__).resolve().parents[4]
usd_path = (
    repo_root
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "mr_v1"
    / "usd_creations"
    / "flyrod"
    / "flyrod.usd"
)

##
# Configuration
##

FLYROD_MARL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Flyrod",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path.resolve()),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "falcon_odom_joints": ImplicitActuatorCfg(
            joint_names_expr=["Falcon[12]_odometry_sensor_joint"],
            stiffness=None,
            damping=None,
        ),
        # keep same patterns as flyrodv2 to allow multi-agent setups (MARL)
        "falcon_imu_joints": ImplicitActuatorCfg(
            joint_names_expr=["Falcon[12]_imu_joint"],
            stiffness=None,
            damping=None,
        ),
        "falcon_rotor_joints": ImplicitActuatorCfg(
            joint_names_expr=["Falcon[12]_rotor_.*_joint"],
            stiffness=None,
            damping=None,
        ),
        # "rope_joints": ImplicitActuatorCfg(
        #     joint_names_expr=["rope_[12]_sphere_joint_.*_joint_.*"],
        #     stiffness=None,
        #     damping=0.005,
        # ),
    },
)
"""Configuration for the flyrod articulation."""
