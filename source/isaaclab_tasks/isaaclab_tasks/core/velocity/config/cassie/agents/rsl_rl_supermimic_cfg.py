# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic settings for flat Cassie velocity with the state_obs preset.

The per-step positive reward bound is 0.02 * (2 + 1 + 5 * 0.3).
"""

from isaaclab.utils import configclass, replace

from ...rsl_rl_supermimic_cfg import FlatVelocitySuperMimicRunnerCfg


@configclass
class CassieFlatSuperMimicRunnerCfg(FlatVelocitySuperMimicRunnerCfg):
    """Use the shared learner with the robot's physical reward bound."""

    experiment_name = "cassie_flat_supermimic"
    algorithm = replace(FlatVelocitySuperMimicRunnerCfg().algorithm, max_step_reward=0.09)
