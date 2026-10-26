# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native runner selection through Isaac Lab's configuration boundary."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import (
    RslRlOffPolicyRunnerCfg,
    RslRlSuperMimicRunnerCfg,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("runner_name", ["OffPolicyRunner", "SuperMimicRunner"])
def test_native_runner_configuration_constructs_the_selected_learner(runner_name):
    """Native defaults and task options survive Isaac Lab config conversion."""
    import rsl_rl.runners as runners

    if not hasattr(runners, runner_name):
        pytest.skip(f"Installed RSL-RL build does not include {runner_name}")
    task = SimpleNamespace(
        representation=SimpleNamespace(feature_dim=1, increment_dim=1, error_floor=torch.ones(1)),
        name="native_factory",
        history_length=2,
        num_actions=1,
        obs_dim=1,
        goal_dim=1,
        xi_dim=1,
        critic_dim=1,
        max_step_reward=1.0,
        label_names=(),
        encoder_hidden=None,
        action_joints=None,
    )
    env = SimpleNamespace(
        task=task,
        device="cpu",
        num_envs=4,
        num_actions=1,
        cfg={},
        get_observations=lambda: TensorDict({"policy": torch.zeros(4, 1)}, batch_size=[4]),
    )
    common = dict(device="cpu", max_iterations=1, save_interval=1, experiment_name="native_factory")
    if runner_name == "SuperMimicRunner":
        cfg = RslRlSuperMimicRunnerCfg(
            **common,
            algorithm={
                "compile": False,
                "horizon": 3,
                "world_model_hidden": (8,),
                "encoder_hidden": (8,),
                "decoder_hidden": (8,),
                "critic_hidden": (8,),
            },
        )
    else:
        cfg = RslRlOffPolicyRunnerCfg(
            **common,
            obs_groups={"actor": ["policy"], "critic": ["policy"]},
            algorithm={
                "class_name": "SAC",
                "replay_buffer_size": 64,
                "learning_starts": 0,
                "num_updates": 1,
                "batch_size": 8,
            },
        )
    cfg = handle_deprecated_rsl_rl_cfg(cfg, "5.5.1")
    runner = create_rsl_rl_runner(env, cfg)
    assert isinstance(runner, getattr(runners, runner_name))
    if runner_name == "SuperMimicRunner":
        assert runner.alg.cfg.horizon == 3 and runner.alg.obs_history.shape == (4, 2, 1)
    else:
        assert runner.alg.batch_size == 8 and runner.alg.num_updates == 1
        assert runner.alg.actor.distribution.action_bound.item() == 2
