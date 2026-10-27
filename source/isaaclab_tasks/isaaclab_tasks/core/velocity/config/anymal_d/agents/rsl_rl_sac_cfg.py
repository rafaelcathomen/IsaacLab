# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlSacRunnerCfg


@configclass
class AnymalDFlatSacRunnerCfg(RslRlSacRunnerCfg):
    # about 2 h on one L40S GPU with 4096 environments
    max_iterations = 19000
    experiment_name = "anymal_d_flat_sac"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
