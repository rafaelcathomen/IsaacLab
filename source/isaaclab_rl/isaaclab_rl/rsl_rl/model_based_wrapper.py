# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical transition collection for a task-owned differentiable SuperMimic model."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..utils.env_types import check_env_type

if TYPE_CHECKING:
    from rsl_rl.env.model_based_env import DifferentiableTask, State


class RslRlModelBasedEnvWrapper:
    """Expose an Isaac Lab environment through RSL-RL's ``ModelBasedEnv`` contract.

    ``capture(env, successor)`` belongs to the task package. It returns owned or
    borrowed tensors named ``state`` (flat physical state), ``xi`` (privileged
    context) and ``goal_state``. When ``successor=True``, return the true successor
    goal, including task-owned goal advancement, before reset. Optional ``labels``
    describe that successor; optional ``disturbance`` describes the start state.
    This wrapper copies every capture before resets or subsequent steps can reuse
    its storage. Geometry, goal semantics and differentiable rewards stay in ``task``.

    Capture callbacks must only read simulation state. They must not recompute
    policy observations, advance observation history or draw random goals. Tasks
    with stochastic goal advancement must expose the already chosen successor.
    """

    _callback_name = "rsl_rl_model_based_transition"

    def __init__(
        self,
        env: Any,
        task: DifferentiableTask,
        capture: Callable[[Any, bool], dict[str, torch.Tensor]],
    ) -> None:
        """Reset the environment and attach a collector to its pre-reset step phase."""
        check_env_type(env, allow_warp=False)
        self.env, self.task, self.capture = env, task, capture
        self.num_envs, self.device = self.unwrapped.num_envs, self.unwrapped.device
        if not hasattr(self.unwrapped, "register_pre_reset_callback"):
            raise TypeError("Model-based collection requires Isaac Lab's pre-reset callback interface")
        num_actions = (
            self.unwrapped.action_manager.total_action_dim
            if hasattr(self.unwrapped, "action_manager")
            else self.unwrapped.single_action_space.shape[0]
        )
        if task.num_actions != num_actions:
            raise ValueError("Task actions must match the environment's input action width")
        self.prev_action = torch.zeros(self.num_envs, num_actions, device=self.device)
        self.prev_prev_action = torch.zeros_like(self.prev_action)
        self._closed = False
        self._successor: dict[str, torch.Tensor] | None = None
        self.unwrapped.register_pre_reset_callback(self._callback_name, self._capture_successor)
        try:
            self.reset()
        except Exception:
            self.unwrapped.unregister_pre_reset_callback(self._callback_name)
            raise

    @property
    def unwrapped(self) -> Any:
        """Return the physical environment."""
        return self.env.unwrapped

    @property
    def cfg(self) -> object:
        """Return the simulator configuration for logging."""
        return self.unwrapped.cfg

    @property
    def episode_step(self) -> torch.Tensor:
        """Return the current episode counters, including autoreset boundaries."""
        return self.unwrapped.episode_length_buf

    @property
    def xi(self) -> torch.Tensor:
        """Return privileged physical context without adding it to actor observations."""
        return self._current["xi"]

    @property
    def goal_state(self) -> torch.Tensor:
        """Return the task's current exogenous goal state."""
        return self._current["goal_state"]

    def state(self) -> State:
        """Decode the task's physical state without a robot-specific layout."""
        return self.task.representation.unflatten(self._current["state"])

    def _snapshot(self, successor: bool) -> dict[str, torch.Tensor]:
        snapshot = self.capture(self.unwrapped, successor)
        missing = {"state", "xi", "goal_state"} - snapshot.keys()
        if missing:
            raise ValueError(f"Model-based capture is missing {sorted(missing)}")
        return {key: value.detach().clone() for key, value in snapshot.items()}

    def _capture_successor(self, env: Any) -> None:
        self._successor = self._snapshot(True)

    @torch.no_grad()
    def reset(self, **kwargs: Any) -> tuple:
        """Reset all environments, physical snapshots and previous-action context."""
        result = self.env.reset(**kwargs)
        self.prev_action.zero_()
        self.prev_prev_action.zero_()
        self._current = self._snapshot(False)
        self._successor = None
        return result

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> dict:
        """Step once and retain both the physical successor and the new episode state."""
        if self._closed:
            raise RuntimeError("The model-based environment wrapper is closed")
        start, episode_step = self._current, self.episode_step.clone()
        action = actions.detach().clone()
        self._successor = None
        _, reward, terminated, truncated, extras = self.env.step(actions)
        if self._successor is None:
            raise RuntimeError("The environment did not invoke the pre-reset transition callback")
        done = terminated | truncated
        transition = {
            "state": start["state"],
            "next_state": self._successor["state"],
            "xi": start["xi"],
            "goal_state": start["goal_state"],
            "next_goal_state": self._successor["goal_state"],
            "action": action,
            "prev_action": self.prev_action,
            "prev_prev_action": self.prev_prev_action,
            "episode_step": episode_step,
            "reward": reward.clone(),
            "done": done,
            "terminated": terminated.clone(),
        }
        if "labels" in self._successor:
            transition["labels"] = self._successor["labels"]
        if "disturbance" in start:
            transition["disturbance"] = start["disturbance"]
        for key in ("metrics", "episode_length", "episode_return", "episode_terms"):
            if key in extras:
                transition[key] = copy.deepcopy(extras[key])
        if extras.get("log"):
            transition["metrics"] = {
                key: torch.as_tensor(value, device=self.device).clone()
                for key, value in {**extras["log"], **transition.get("metrics", {})}.items()
            }
        self.prev_prev_action = self.prev_action.masked_fill(done[:, None], 0)
        self.prev_action = action.masked_fill(done[:, None], 0)
        self._current = self._snapshot(False)
        return transition

    def close(self) -> None:
        """Detach the collector before closing the simulator."""
        if not self._closed:
            self.unwrapped.unregister_pre_reset_callback(self._callback_name)
            self._closed = True
            self.env.close()
