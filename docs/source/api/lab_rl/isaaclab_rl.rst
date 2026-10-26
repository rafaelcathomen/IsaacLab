.. _api-isaaclab-rl:

isaaclab_rl
===========

.. automodule:: isaaclab_rl

Unified Entrypoints
-------------------

.. automodule:: isaaclab_rl.entrypoints

.. autoclass:: isaaclab_rl.entrypoints.TrainingRequest
   :members:
   :show-inheritance:

.. autoclass:: isaaclab_rl.entrypoints.PlaybackRequest
   :members:
   :show-inheritance:

.. autoclass:: isaaclab_rl.entrypoints.SimpleAgentRequest
   :members:
   :show-inheritance:

.. autofunction:: isaaclab_rl.entrypoints.train

.. autofunction:: isaaclab_rl.entrypoints.play

.. autofunction:: isaaclab_rl.entrypoints.zero_agent

.. autofunction:: isaaclab_rl.entrypoints.random_agent

.. autofunction:: isaaclab_rl.entrypoints.run_train_cli

.. autofunction:: isaaclab_rl.entrypoints.run_play_cli

.. autofunction:: isaaclab_rl.entrypoints.run_zero_agent_cli

.. autofunction:: isaaclab_rl.entrypoints.run_random_agent_cli

RL Utilities
------------

.. automodule:: isaaclab_rl.utils.wandb
   :members:
   :show-inheritance:

RL-Games Wrapper
----------------

.. automodule:: isaaclab_rl.rl_games
   :members:
   :show-inheritance:

RSL-RL Wrapper
--------------

Native off-policy and model-based runners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runner factory supports PPO, distillation, native SAC/FlashSAC and SuperMimic.
The selected runner must be present in the installed RSL-RL build. The native
algorithm implementations live in the corresponding branches of
`rafaelcathomen/rsl_rl <https://github.com/rafaelcathomen/rsl_rl>`_. Isaac Lab
forwards algorithm dictionaries so omitted options retain the library defaults.

Use ``RslRlOffPolicyRunnerCfg`` for SAC and select actor and privileged critic
observation groups with ``obs_groups``. ``wrap_rsl_rl_env(env, agent_cfg)`` enables
owned final observations for that runner. True terminations take precedence over
simultaneous timeouts. Terminal history previews include the latest physical
frame without advancing live delay or history buffers twice.
Off-policy and model-based collection require the Torch frontend. Warp frontends
are rejected because they do not supply pre-reset terminal observations or physical
transition callbacks. Ordinary on-policy Warp collection remains available.

Native model and algorithm dictionaries accept Hydra ``+`` and ``++`` additions.
Their consumer in RSL-RL validates these options. Fixed-schema configuration fields
retain their existing validation.

Use ``RslRlSuperMimicRunnerCfg`` with a task-owned ``env_adapter`` factory for
SuperMimic. The factory receives the Gym environment and returns a
``ModelBasedEnv``. ``RslRlModelBasedEnvWrapper`` supplies physical collection for
manager-based and direct RL environments. Its task-owned capture function returns
flat physical state, privileged context and exogenous goals. Successor captures
run after physics and reward evaluation, before autoreset, command updates and
interval events. Goal advancement belongs to the task; stochastic successors
must already be chosen when captured. Physical context does not automatically
become a policy input.

Class-based observation terms receive terminal reads through ``preview()`` with
``_is_observation_preview`` set for the call. Terms with private history must honor
this flag. Direct environments expose the same flag during terminal observation
computation. Built-in camera stacks preview the latest frame without advancing
their live history.

The unified training and playback entrypoints select these wrappers from the
runner configuration. SuperMimic playback maintains observation history and
passes goals separately to the policy. Playback exports its two-input policy to
TorchScript. The generic observation-only LEAPP export path does not support
SuperMimic; task-specific exporters remain in the task package.
When resuming SuperMimic, ``--no-resume_optimizer_state`` retains all learned models,
task training state and iteration while keeping fresh optimizers. It requires an
RSL-RL build with ``SuperMimicRunner.load(..., load_optimizer=False)`` support.

.. automodule:: isaaclab_rl.rsl_rl
   :members:
   :imported-members:
   :show-inheritance:

SKRL Wrapper
------------

.. automodule:: isaaclab_rl.skrl
   :members:
   :show-inheritance:

Stable-Baselines3 Wrapper
-------------------------

.. automodule:: isaaclab_rl.sb3
   :members:
   :show-inheritance:

TorchRL Wrapper
---------------

.. automodule:: isaaclab_rl.torchrl
   :members:
   :imported-members:
   :show-inheritance:
