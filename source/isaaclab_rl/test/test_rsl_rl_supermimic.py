# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic task adapters and native configuration at the Isaac Lab boundary."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_rl.rsl_rl import RslRlSuperMimicRunnerCfg, create_rsl_rl_runner, wrap_rsl_rl_env

pytestmark = pytest.mark.unit


def test_task_adapter_constructs_native_supermimic():
    """The task-owned adapter and learner options reach the selected native runner."""
    import rsl_rl.runners as runners

    if not hasattr(runners, "SuperMimicRunner"):
        pytest.skip("The installed RSL-RL does not include SuperMimicRunner")
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
    model_env = SimpleNamespace(task=task, device="cpu", num_envs=4, cfg={}, state=lambda: (torch.zeros(4, 1),))
    raw_env = object()

    def adapter(env):
        assert env is raw_env
        return model_env

    cfg = RslRlSuperMimicRunnerCfg(
        device="cpu",
        max_iterations=1,
        save_interval=1,
        experiment_name="native_factory",
        env_adapter=adapter,
        algorithm={
            "compile": False,
            "horizon": 3,
            "world_model_hidden": (8,),
            "encoder_hidden": (8,),
            "decoder_hidden": (8,),
            "critic_hidden": (8,),
        },
    )
    runner = create_rsl_rl_runner(wrap_rsl_rl_env(raw_env, cfg), cfg)
    assert isinstance(runner, runners.SuperMimicRunner)
    assert runner.env is model_env
    assert runner.alg.cfg.horizon == 3 and runner.alg.obs_history.shape == (4, 2, 1)
    cfg.env_adapter = None
    assert wrap_rsl_rl_env(model_env, cfg) is model_env
    with pytest.raises(ValueError, match="task-owned env_adapter"):
        wrap_rsl_rl_env(raw_env, cfg)


def test_native_options_added_through_hydra_survive_config_restoration(monkeypatch):
    """Task-specific options survive Hydra while the runner schema stays strict."""
    import gymnasium as gym

    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.hydra import resolve_task_config

    @configclass
    class EnvCfg:
        seed: int = 42

    cfg = RslRlSuperMimicRunnerCfg(device="cpu", max_iterations=1, save_interval=1, experiment_name="native_cli")
    cfg.algorithm["max_replay_memory_fraction"] = None
    task_name = "Isaac-Native-SuperMimic-Options-Test"
    gym.register(id=task_name, entry_point="unused:Env")
    monkeypatch.setattr(
        "isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry",
        lambda task, key: EnvCfg() if key == "env_cfg_entry_point" else cfg.copy(),
    )
    try:
        _, resolved = resolve_task_config(
            task_name,
            "rsl_rl_cfg_entry_point",
            overrides=["+agent.algorithm.horizon=16", "++agent.algorithm.max_replay_memory_fraction=0.5"],
        )
        assert resolved.algorithm["horizon"] == 16
        assert resolved.algorithm["max_replay_memory_fraction"] == 0.5
        with pytest.raises(KeyError, match="unknown_runner_field"):
            resolved.from_dict({"unknown_runner_field": True})
    finally:
        del gym.registry[task_name]
