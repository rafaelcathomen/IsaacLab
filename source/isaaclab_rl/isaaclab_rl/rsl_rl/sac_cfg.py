# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import field
from typing import Literal

from isaaclab.utils import configclass

from .rl_cfg import RslRlOffPolicyRunnerCfg


@configclass
class RslRlSacRunnerCfg(RslRlOffPolicyRunnerCfg):
    """Configuration of RSL-RL's SAC with the settings benchmarked across Isaac Lab tasks.

    RSL-RL's SAC defaults are FlashSAC-style: batch-normalized residual networks, a categorical critic with reward
    normalization, held exploration noise, 3-step returns, a 10M-transition replay buffer, :math:`\\gamma = 0.98`,
    learning rates of 1e-4, and an action bound of 2. On top of these, this configuration collects 8 steps per
    iteration and runs 64 gradient steps per iteration (8 per environment step), captures the gradient step in a CUDA
    graph, and maps the actions of joint-position-to-limits terms onto their valid range
    (:attr:`action_mapping`). Use 4096 environments.
    """

    num_steps_per_env: int = 8
    algorithm: dict = field(
        default_factory=lambda: {"class_name": "SAC", "num_updates": 64, "use_cuda_graph": True},
        metadata={"allow_new_keys": True},
    )
    save_interval: int = 500

    action_mapping: Literal["none", "to_limits"] = "to_limits"
    """Per-dimension action bounds resolved from the environment's action terms before the runner is created.

    * ``"to_limits"``: the actions of :class:`~isaaclab.envs.mdp.actions.JointPositionToLimitsAction` terms with
      ``rescale_to_limits`` get the bound ``1 / scale``, so the squashed action sweeps exactly the joint limits; all
      other dimensions keep the scalar ``algorithm["action_bound"]``. Environments without an action manager keep the
      scalar bound.
    * ``"none"``: the scalar ``algorithm["action_bound"]`` for every dimension.
    """
