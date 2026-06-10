# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-TrackPositionNoObstacles-Flyrod-DirectMARL-v3",
    entry_point=f"{__name__}.direct_marl_flyrod_env_v3:DirectMARLFlyrodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_marl_flyrod_env_cfg_v3:DirectMARLFlyrodEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TrackPositionNoObstacles-Flyrod-DirectMARL-Play-v3",
    entry_point=f"{__name__}.direct_marl_flyrod_env_v3:DirectMARLFlyrodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_marl_flyrod_env_cfg_v3:DirectMARLFlyrodEnvCfg_PLAY",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
