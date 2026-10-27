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

Off-policy runners (SAC)
~~~~~~~~~~~~~~~~~~~~~~~~

The runner factory (:func:`~isaaclab_rl.rsl_rl.create_rsl_rl_runner`) supports PPO, distillation, and RSL-RL's
off-policy runner with SAC; the selected runner must be present in the installed RSL-RL. Isaac Lab forwards the
algorithm and model dictionaries of :class:`~isaaclab_rl.rsl_rl.RslRlOffPolicyRunnerCfg`, so omitted options keep the
library defaults. :class:`~isaaclab_rl.rsl_rl.RslRlSacRunnerCfg` adds the settings benchmarked across Isaac Lab tasks.
Tasks with a SAC configuration register it as ``rsl_rl_sac_cfg_entry_point``:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Reorient-Cube-Allegro --agent rsl_rl_sac_cfg_entry_point

For SAC, :func:`~isaaclab_rl.rsl_rl.resolve_sac_action_bounds` sets per-dimension action bounds from the environment's
action terms before the runner is created, in the raw action coordinates accepted by the environment: actions of
joint-position-to-limits terms sweep exactly the joint limits, all others keep the scalar bound. The resolved bounds
are stored in the runner configuration, so playback and export use the same mapping.

:func:`~isaaclab_rl.rsl_rl.wrap_rsl_rl_env` enables owned terminal observations (``extras["final_obs"]``) for
off-policy runners, so episodes that time out bootstrap from their true final observation. True terminations take
precedence over simultaneous time-outs. Terminal capture requires the Torch frontend; Warp frontends are rejected
because they do not supply pre-reset terminal observations.

Terminal observations are computed as a preview that includes the current sample without advancing live delay,
history, or tensor processor state. Class-based observation terms receive terminal reads through ``preview()`` with
``_is_observation_preview`` set; terms with private history must honor this flag, and direct environments expose the
same flag while computing terminal observations. Built-in camera stacks, integrators, and digital filters support
previews; custom class-based tensor processors must implement
:meth:`~isaaclab.utils.tensor_processors.TensorProcessor.preview`.

Native model and algorithm dictionaries accept Hydra ``+`` and ``++`` additions, validated by their consumer in
RSL-RL. Fixed-schema configuration fields keep their existing validation.

Model-based runners (SuperMimic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``RslRlSuperMimicRunnerCfg`` with a task-owned ``env_adapter`` factory for
SuperMimic. The factory receives the Gym environment and returns a
``ModelBasedEnv``. ``RslRlModelBasedEnvWrapper`` supplies physical collection for
manager-based and direct RL environments. Its task-owned capture function returns
flat physical state, privileged context and exogenous goals. Successor captures
run after physics and reward evaluation, before autoreset, command updates and
interval events. Goal advancement belongs to the task; stochastic successors
must already be chosen when captured. Physical context does not automatically
become a policy input.

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
