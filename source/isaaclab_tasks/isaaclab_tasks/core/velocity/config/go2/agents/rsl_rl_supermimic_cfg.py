# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic settings for flat UnitreeGo2 velocity with the state_obs preset.

The per-step positive reward bound is 0.02 * (1.5 + 0.75 + 0.25 * 4 * (20 - 0.5)).
"""

from isaaclab.utils import configclass, replace

from ...rsl_rl_supermimic_cfg import FlatVelocitySuperMimicRunnerCfg


@configclass
class UnitreeGo2FlatSuperMimicRunnerCfg(FlatVelocitySuperMimicRunnerCfg):
    """Use the shared learner with the robot's physical reward bound."""

    experiment_name = "go2_flat_supermimic"
    algorithm = replace(FlatVelocitySuperMimicRunnerCfg().algorithm, max_step_reward=0.435)
