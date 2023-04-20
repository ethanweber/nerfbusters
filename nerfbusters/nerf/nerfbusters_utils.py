"""
Utils for Nerfbusters project..
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Type, Union

import torch
from rich.progress import Console

from nerfstudio.cameras.cameras import Cameras

CONSOLE = Console(width=120)


def random_train_pose(
    size: int,
    resolution: int,
    device: Union[torch.device, str],
    radius_mean: float = 1.0,
    radius_std: float = 0.1,
    central_rotation_range: Tuple[float, float] = (0, 360),
    vertical_rotation_range: Tuple[float, float] = (-90, 0),
    focal_range: Tuple[float, float] = (0.75, 1.35),
    jitter_std: float = 0.01,
    center: Tuple[float, float, float] = (0, 0, 0),
):
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius_mean: mean radius of the orbit camera.
        radius_std: standard deviation of the radius of the orbit camera.
        central_rotation_range: amount that we rotate around the center of the object
        vertical_rotation_range: amount that we allow the cameras to pan up and down from horizontal
        focal_range: focal length range
        jitter_std: standard deviation of the jitter added to the camera position
        center: center of the object
    Return:
        poses: [size, 4, 4]
    """

    assert (
        vertical_rotation_range[0] >= -90 and vertical_rotation_range[1] <= 90
    ), "vertical_rotation_range must be in [-90, 90]"

    vertical_rotation_range = [vertical_rotation_range[0] + 90, vertical_rotation_range[1] + 90]
    # This is the uniform sample on the part of the sphere we care about where 0 = 0 degrees and 1 = 360 degrees
    sampled_uniform = (
        torch.rand(size) * (vertical_rotation_range[1] - vertical_rotation_range[0]) + vertical_rotation_range[0]
    ) / 180
    vertical_rotation = torch.arccos(1 - 2 * sampled_uniform)
    central_rotation = torch.deg2rad(
        torch.rand(size) * (central_rotation_range[1] - central_rotation_range[0]) + central_rotation_range[0]
    )

    c_cos = torch.cos(central_rotation)
    c_sin = torch.sin(central_rotation)
    v_cos = torch.cos(vertical_rotation)
    v_sin = torch.sin(vertical_rotation)
    zeros = torch.zeros_like(central_rotation)
    ones = torch.ones_like(central_rotation)

    rot_z = torch.stack(
        [
            torch.stack([c_cos, -c_sin, zeros], dim=-1),
            torch.stack([c_sin, c_cos, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    rot_x = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, v_cos, -v_sin], dim=-1),
            torch.stack([zeros, v_sin, v_cos], dim=-1),
        ],
        dim=-2,
    )

    # Default directions are facing in the -z direction, so origins should face opposite way
    origins = torch.stack([torch.tensor([0, 0, 1])] * size, dim=0)
    origins = (origins * radius_mean) + (origins * (torch.randn((origins.shape)) * radius_std))
    R = torch.bmm(rot_z, rot_x)  # Want to have Rx @ Ry @ origin
    t = (
        torch.bmm(R, origins.unsqueeze(-1))
        + torch.randn((size, 3, 1)) * jitter_std
        + torch.tensor(center)[None, :, None]
    )
    camera_to_worlds = torch.cat([R, t], dim=-1)

    focals = torch.rand(size) * (focal_range[1] - focal_range[0]) + focal_range[0]

    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=focals * resolution,
        fy=focals * resolution,
        cx=resolution / 2,
        cy=resolution / 2,
    ).to(device)

    return cameras, torch.rad2deg(vertical_rotation), torch.rad2deg(central_rotation)
