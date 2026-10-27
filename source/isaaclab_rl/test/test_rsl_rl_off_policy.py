# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Off-policy (SAC) runner creation through Isaac Lab's configuration boundary."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import (
    RslRlOffPolicyRunnerCfg,
    RslRlSacRunnerCfg,
    create_rsl_rl_runner,
    resolve_sac_action_bounds,
)

pytestmark = pytest.mark.unit

rsl_rl_runners = pytest.importorskip("rsl_rl.runners")
if not hasattr(rsl_rl_runners, "OffPolicyRunner"):
    pytest.skip("The installed RSL-RL does not include the OffPolicyRunner.", allow_module_level=True)

COMMON = dict(device="cpu", max_iterations=1, save_interval=1, experiment_name="off_policy_test")
SMALL_SAC = {"replay_buffer_size": 64, "learning_starts": 0, "num_updates": 1, "batch_size": 8, "use_cuda_graph": False}


def _env(num_actions: int = 1, action_manager: object = None) -> SimpleNamespace:
    """Return the parts of a wrapped environment that runner construction reads."""
    unwrapped = SimpleNamespace(action_manager=action_manager)
    return SimpleNamespace(
        device="cpu",
        num_envs=4,
        num_actions=num_actions,
        max_episode_length=10,
        cfg={},
        unwrapped=unwrapped,
        get_observations=lambda: TensorDict({"policy": torch.zeros(4, 3)}, batch_size=[4]),
    )


def _action_manager(scales: list[float | None], dims: list[int]) -> SimpleNamespace:
    """Return an action manager with to-limits terms (scale given) and other terms (scale None)."""
    from isaaclab.envs.mdp.actions import JointPositionToLimitsAction

    terms = {}
    for index, (scale, dim) in enumerate(zip(scales, dims)):
        if scale is None:
            terms[f"other_{index}"] = SimpleNamespace(action_dim=dim)
        else:
            term = JointPositionToLimitsAction.__new__(JointPositionToLimitsAction)
            term.cfg = SimpleNamespace(rescale_to_limits=True, clip=None)
            term._scale = scale
            term.__dict__["action_dim"] = dim
            terms[f"to_limits_{index}"] = term
    return SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__)


@pytest.fixture(autouse=True)
def _plain_action_dim(monkeypatch):
    """Let the stand-in action terms report their dimension without a simulation."""
    from isaaclab.envs.mdp.actions import JointPositionToLimitsAction

    monkeypatch.setattr(JointPositionToLimitsAction, "action_dim", property(lambda self: self.__dict__["action_dim"]))


def test_off_policy_configuration_constructs_sac():
    """Native SAC options survive the configuration conversion; omitted options keep the RSL-RL defaults."""
    cfg = RslRlOffPolicyRunnerCfg(
        **COMMON, obs_groups={"actor": ["policy"], "critic": ["policy"]}, algorithm={"class_name": "SAC", **SMALL_SAC}
    )
    runner = create_rsl_rl_runner(_env(), cfg)
    assert isinstance(runner, rsl_rl_runners.OffPolicyRunner)
    assert runner.alg.batch_size == 8 and runner.alg.num_updates == 1
    assert runner.alg.actor.distribution.action_bound.item() == 2


def test_sac_configuration_defaults():
    """The SAC configuration collects 8 steps and runs 64 gradient steps per iteration."""
    cfg = RslRlSacRunnerCfg(**COMMON)
    assert cfg.num_steps_per_env == 8
    assert cfg.algorithm == {"class_name": "SAC", "num_updates": 64, "use_cuda_graph": True}
    assert cfg.action_mapping == "to_limits"


def test_to_limits_mapping_sets_per_dimension_bounds():
    """To-limits bounds cover the joint range regardless of scale direction, in action-manager order."""
    manager = _action_manager([None, -0.5, torch.tensor([[0.25, -1.0]])], [2, 1, 2])
    cfg = RslRlSacRunnerCfg(**COMMON, obs_groups={"actor": ["policy"], "critic": ["policy"]})
    cfg.algorithm.update(SMALL_SAC)
    env = _env(num_actions=5, action_manager=manager)
    runner = create_rsl_rl_runner(env, cfg)
    assert cfg.algorithm["action_bound"] == [2.0, 2.0, 2.0, 4.0, 1.0]
    assert cfg.algorithm["action_offset"] == [0.0] * 5
    torch.testing.assert_close(runner.alg.actor.distribution.action_bound, torch.tensor([2.0, 2.0, 2.0, 4.0, 1.0]))


def test_to_limits_mapping_without_to_limits_terms_keeps_the_scalar_bound():
    """Tasks without to-limits terms (and environments without an action manager) keep the scalar bound."""
    for manager in (_action_manager([None], [3]), None):
        cfg = RslRlSacRunnerCfg(**COMMON)
        resolve_sac_action_bounds(_env(num_actions=3, action_manager=manager), cfg)
        assert "action_bound" not in cfg.algorithm and "action_offset" not in cfg.algorithm


def test_mapping_rejects_unsupported_settings():
    """Clipped to-limits terms and action clipping narrower than the actor's range are rejected."""
    manager = _action_manager([0.5], [1])
    manager.get_term("to_limits_0").cfg.clip = {".*": (-1.0, 1.0)}
    with pytest.raises(ValueError, match="clipped action term"):
        resolve_sac_action_bounds(_env(action_manager=manager), RslRlSacRunnerCfg(**COMMON))
    with pytest.raises(ValueError, match="truncate"):
        resolve_sac_action_bounds(_env(), RslRlSacRunnerCfg(**COMMON, clip_actions=1.0))
    manager = _action_manager([0.0], [1])
    with pytest.raises(ValueError, match="finite, nonzero"):
        resolve_sac_action_bounds(_env(action_manager=manager), RslRlSacRunnerCfg(**COMMON))


@pytest.mark.parametrize(
    ("algorithm", "clip_actions", "rejected"),
    [
        ({"action_bound": [1.0], "action_offset": [2.0]}, 2.0, True),
        ({"action_bound": (1.0,), "action_offset": (-2.0,)}, 3.0, False),
        ({"action_bound": 1.0, "final_action_bound": 3.0}, 2.0, True),
    ],
)
def test_action_clipping_covers_affine_and_scheduled_support(algorithm, clip_actions, rejected):
    """Clipping must contain the full mapped range, including offsets and scheduled bounds."""
    cfg = RslRlSacRunnerCfg(
        **COMMON,
        action_mapping="none",
        clip_actions=clip_actions,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
    )
    cfg.algorithm.update({**SMALL_SAC, **algorithm})
    if rejected:
        with pytest.raises(ValueError, match="truncate"):
            create_rsl_rl_runner(_env(), cfg)
    else:
        runner = create_rsl_rl_runner(_env(), cfg)
        torch.testing.assert_close(runner.alg.actor.distribution.action_offset, torch.tensor([-2.0]))


def test_inference_runner_allocates_a_minimal_replay_buffer():
    """CPU playback and export omit the training replay allocation and CUDA graphs."""
    cfg = RslRlSacRunnerCfg(**COMMON, obs_groups={"actor": ["policy"], "critic": ["policy"]})
    cfg.algorithm["replay_buffer_size"] = 10_000_000
    runner = create_rsl_rl_runner(_env(), cfg, inference=True)
    assert runner.alg.storage.num_slots == 3
    assert not runner.alg.use_cuda_graph
    assert cfg.algorithm["use_cuda_graph"]


def test_native_options_added_through_hydra_survive_config_restoration(monkeypatch):
    """Hydra additions reach the native algorithm dictionary while ordinary config fields remain strict."""
    import gymnasium as gym

    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.hydra import resolve_task_config

    @configclass
    class EnvCfg:
        seed: int = 42

    cfg = RslRlSacRunnerCfg(**COMMON)
    task_name = "Isaac-Native-OffPolicy-Options-Test"
    gym.register(id=task_name, entry_point="unused:Env")
    monkeypatch.setattr(
        "isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry",
        lambda task, key: EnvCfg() if key == "env_cfg_entry_point" else cfg.copy(),
    )
    try:
        _, resolved = resolve_task_config(
            task_name,
            "rsl_rl_cfg_entry_point",
            overrides=["+agent.algorithm.batch_size=16", "agent.algorithm.num_updates=8"],
        )
        assert resolved.algorithm["batch_size"] == 16 and resolved.algorithm["num_updates"] == 8
        assert resolved.algorithm["class_name"] == "SAC"
        with pytest.raises(KeyError, match="unknown_runner_field"):
            resolved.from_dict({"unknown_runner_field": True})
    finally:
        del gym.registry[task_name]
