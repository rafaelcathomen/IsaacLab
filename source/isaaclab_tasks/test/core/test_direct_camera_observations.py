# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminal observation ownership in the direct camera task, without rendering."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.images import CameraFrameStack

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv

pytestmark = pytest.mark.unit


def test_direct_terminal_camera_read_does_not_advance_survivor_history():
    """Use the real task observation path with sensor and joint tensor stand-ins."""
    env = CartpoleCameraEnv.__new__(CartpoleCameraEnv)
    env._is_closed = True
    env._is_observation_preview = False
    env.cfg = SimpleNamespace(
        scene=SimpleNamespace(tiled_camera=SimpleNamespace(data_types=["rgb"])),
        write_image_to_file=False,
        observation_noise_model=None,
    )
    env.scene = SimpleNamespace(num_envs=2)
    camera = torch.randint(0, 255, (2, 4, 4, 3), dtype=torch.uint8)
    env._tiled_camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=camera)}))
    env._frames = CameraFrameStack(2, "cpu", 2, channel_first=True)
    env.joint_pos = torch.zeros(2, 2)
    env.joint_vel = torch.zeros(2, 2)
    env.cartpole = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=env.joint_pos.clone()),
            default_joint_vel=SimpleNamespace(torch=env.joint_vel.clone()),
        )
    )
    env._cart_dof_idx, env._pole_dof_idx = [0], [1]
    previous = camera.float() / 255
    previous = (previous - previous.mean((1, 2), keepdim=True)).permute(0, 3, 1, 2)
    env._compute_observations()
    camera.copy_(torch.randint_like(camera, 0, 255))
    successor = camera.float() / 255
    successor = (successor - successor.mean((1, 2), keepdim=True)).permute(0, 3, 1, 2)
    terminal = env._compute_observations(preview=True)["policy"]
    env._frames.reset([0])
    live = env._compute_observations()["policy"]
    expected = torch.cat((previous, successor), dim=1)
    torch.testing.assert_close(terminal, expected)
    torch.testing.assert_close(live[1], expected[1])
    torch.testing.assert_close(live[0], successor[0].repeat(2, 1, 1))
