# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner creation shared by the RSL-RL train, play, and export entry points."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import to_dict

from .rl_cfg import RslRlBaseRunnerCfg
from .vecenv_wrapper import RslRlVecEnvWrapper

if TYPE_CHECKING:
    from rsl_rl.runners import DistillationRunner, OffPolicyRunner, OnPolicyRunner

OFF_POLICY_RUNNERS = ("OffPolicyRunner",)
"""Runner classes trained from replayed transitions, which need the true terminal observations."""


def wrap_rsl_rl_env(env: Any, agent_cfg: RslRlBaseRunnerCfg) -> RslRlVecEnvWrapper:
    """Wrap an Isaac Lab environment for the configured RSL-RL runner.

    Off-policy runners record the terminal observations of episodes that end (``extras["final_obs"]``).
    """
    return RslRlVecEnvWrapper(
        env, clip_actions=agent_cfg.clip_actions, record_final_obs=agent_cfg.class_name in OFF_POLICY_RUNNERS
    )


def create_rsl_rl_runner(
    env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg, log_dir: str | None = None, inference: bool = False
) -> OnPolicyRunner | DistillationRunner | OffPolicyRunner:
    """Instantiate the RSL-RL runner selected by ``agent_cfg.class_name``.

    For SAC, the per-dimension action bounds are resolved from the environment first (see
    :func:`resolve_sac_action_bounds`) and stored in ``agent_cfg``, so training, playback, and export use the same
    mapping.

    Args:
        env: The wrapped environment.
        agent_cfg: The runner configuration.
        log_dir: The log directory of training runs.
        inference: Whether the runner only loads and runs a policy (playback, export). Off-policy runners then
            allocate the smallest replay buffer instead of the training one.

    Raises:
        ValueError: If the configured runner class is not supported.
        ImportError: If the installed RSL-RL does not include the selected runner.
    """
    if getattr(agent_cfg, "action_mapping", None) is not None:
        resolve_sac_action_bounds(env, agent_cfg)
    cfg = to_dict(agent_cfg)
    cfg.pop("action_mapping", None)
    if inference and agent_cfg.class_name in OFF_POLICY_RUNNERS:
        cfg["algorithm"]["replay_buffer_size"] = env.num_envs * cfg["algorithm"].get("n_steps", 3)
    runners = importlib.import_module("rsl_rl.runners")
    if agent_cfg.class_name not in ("OnPolicyRunner", "DistillationRunner", *OFF_POLICY_RUNNERS):
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    try:
        runner_class = getattr(runners, agent_cfg.class_name)
    except AttributeError as exc:
        raise ImportError(f"The installed RSL-RL does not include the {agent_cfg.class_name}.") from exc
    return runner_class(env, cfg, log_dir=log_dir, device=agent_cfg.device)


def resolve_sac_action_bounds(env: RslRlVecEnvWrapper, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Set SAC's per-dimension action bound and offset from the environment's action terms (in place).

    The bounds are in the raw action coordinates accepted by the environment. With ``action_mapping="to_limits"``,
    the dimensions of :class:`~isaaclab.envs.mdp.actions.JointPositionToLimitsAction` terms with
    ``rescale_to_limits`` get the bound ``1 / abs(scale)``. The term scales, clips to [-1, 1], and rescales to the joint
    limits, so the squashed action sweeps exactly the joint limits. All other dimensions keep the scalar
    ``algorithm["action_bound"]``. Environments without to-limits terms, or without an action manager, keep the scalar
    bound unchanged.

    Raises:
        ValueError: If the mapping is unknown, a to-limits term also clips its actions, or ``clip_actions`` would
            truncate the actor's action range.
    """
    mapping = agent_cfg.action_mapping
    if mapping not in ("none", "to_limits"):
        raise ValueError(f"Unknown SAC action mapping: {mapping!r}")
    algorithm = agent_cfg.algorithm
    bound = algorithm.get("action_bound", 2.0)
    manager = getattr(env.unwrapped, "action_manager", None)
    if mapping == "to_limits" and manager is not None and not isinstance(bound, (list, tuple)):
        from isaaclab.envs.mdp.actions import JointPositionToLimitsAction

        bounds, has_to_limits = [], False
        for name in manager.active_terms:
            term = manager.get_term(name)
            if isinstance(term, JointPositionToLimitsAction) and term.cfg.rescale_to_limits:
                if term.cfg.clip is not None:
                    raise ValueError(f"The to-limits mapping does not support the clipped action term '{name}'.")
                scale = term._scale[0] if isinstance(term._scale, torch.Tensor) else term._scale
                scale = torch.as_tensor(scale, dtype=torch.float).expand(term.action_dim)
                if not torch.isfinite(scale).all() or (scale == 0).any():
                    raise ValueError(f"The to-limits action term '{name}' requires finite, nonzero scales.")
                bounds += scale.abs().reciprocal().tolist()
                has_to_limits = True
            else:
                bounds += [float(bound)] * term.action_dim
        if has_to_limits:
            algorithm["action_bound"] = bounds
            algorithm["action_offset"] = [0.0] * len(bounds)
    if agent_cfg.clip_actions is not None:
        bounds = torch.as_tensor(algorithm.get("action_bound", 2.0))
        final_bound = algorithm.get("final_action_bound")
        if final_bound is not None:
            bounds = torch.maximum(bounds, torch.as_tensor(final_bound))
        offsets = algorithm.get("action_offset")
        offsets = torch.as_tensor(0.0 if offsets is None else offsets)
        widest = (bounds + offsets.abs()).max().item()
        if agent_cfg.clip_actions < widest:
            raise ValueError(
                f"clip_actions={agent_cfg.clip_actions} would truncate SAC's action range "
                f"(maximum magnitude {widest}); unset it."
            )
