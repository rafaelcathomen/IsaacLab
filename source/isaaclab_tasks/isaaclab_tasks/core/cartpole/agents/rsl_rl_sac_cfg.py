# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small conventional SAC configuration for Cartpole integration checks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlQMLPModelCfg, RslRlSacAlgorithmCfg, RslRlSacRunnerCfg


@configclass
class CartpoleSACRunnerCfg(RslRlSacRunnerCfg):
    """small training configuration, rather than a locomotion learning benchmark."""

    num_steps_per_env = 8
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_sac"
    clip_actions = None
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.SquashedGaussianDistributionCfg(action_range=(-1.0, 1.0)),
    )
    critic = RslRlQMLPModelCfg(hidden_dims=[32, 32], activation="elu")
    algorithm = RslRlSacAlgorithmCfg(
        num_updates=8,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_starts=0,
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        alpha_learning_rate=3.0e-4,
    )
