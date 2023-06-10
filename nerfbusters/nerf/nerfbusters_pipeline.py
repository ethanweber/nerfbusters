# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The Nerfbusters pipeline.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Type

import mediapy as media
import numpy as np
import torch
import yaml
from dotmap import DotMap
from nerfbusters.cubes.visualize3D import get_image_grid
from nerfbusters.lightning.nerfbusters_trainer import NerfbustersTrainer
from nerfbusters.utils.visualizations import get_3Dimage_fast
from nerfbusters.nerf.nerfbusters_utils import random_train_pose
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.fields.visibility_field import VisibilityField
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler, writer

print_stats = lambda x: print(f"x mean {x.mean():.3f}, std {x.std():.3f}, min {x.min():.3f}, max {x.max():.3f}")


def sample_cubes(
    min_x,
    min_y,
    min_z,
    max_x,
    max_y,
    max_z,
    res,
    spr_min,
    spr_max,
    num_cubes,
    cube_centers_x=None,
    cube_centers_y=None,
    cube_centers_z=None,
    device=None,
):
    assert device is not None, "device must be specified"

    # create the cubes
    scales = torch.rand(num_cubes, device=device) * (spr_max - spr_min) + spr_min
    cube_len = (max_x - min_x) * scales
    half_cube_len = cube_len / 2
    if cube_centers_x is None:
        cube_centers_x = (
            torch.rand(num_cubes, device=device) * (max_x - min_x - 2.0 * half_cube_len) + min_x + half_cube_len
        )
        cube_centers_y = (
            torch.rand(num_cubes, device=device) * (max_y - min_y - 2.0 * half_cube_len) + min_y + half_cube_len
        )
        cube_centers_z = (
            torch.rand(num_cubes, device=device) * (max_z - min_z - 2.0 * half_cube_len) + min_z + half_cube_len
        )
    else:
        assert cube_centers_x.shape == (num_cubes,)
        cube_centers_x = cube_centers_x * (max_x - min_x - 2.0 * half_cube_len) + min_x + half_cube_len
        cube_centers_y = cube_centers_y * (max_y - min_y - 2.0 * half_cube_len) + min_y + half_cube_len
        cube_centers_z = cube_centers_z * (max_z - min_z - 2.0 * half_cube_len) + min_z + half_cube_len
    cube_start_x = cube_centers_x - half_cube_len
    cube_start_y = cube_centers_y - half_cube_len
    cube_start_z = cube_centers_z - half_cube_len
    cube_start_xyz = torch.stack([cube_start_x, cube_start_y, cube_start_z], dim=-1).reshape(num_cubes, 1, 1, 1, 3)
    cube_end_x = cube_centers_x + half_cube_len
    cube_end_y = cube_centers_y + half_cube_len
    cube_end_z = cube_centers_z + half_cube_len
    cube_end_xyz = torch.stack([cube_end_x, cube_end_y, cube_end_z], dim=-1).reshape(num_cubes, 1, 1, 1, 3)
    l = torch.linspace(0, 1, res, device=device)
    xyz = torch.stack(torch.meshgrid([l, l, l], indexing="ij"), dim=-1)  # (res, res, res, 3)
    xyz = xyz[None, ...] * (cube_end_xyz - cube_start_xyz) + cube_start_xyz
    return xyz, cube_start_xyz, cube_end_xyz, scales


class WeightGrid(torch.nn.Module):
    """Keep track of the weights."""

    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution
        self.register_buffer("_weights", torch.ones(self.resolution, self.resolution, self.resolution))

    def update(self, xyz: TensorType["num_points", 3], weights: TensorType["num_points", 1], ema_decay: float = 0.5):
        """Updates the weights of the grid with EMA."""

        # xyz points should be in range [0, 1]
        assert xyz.min() >= 0, f"xyz min {xyz.min()}"
        assert xyz.max() < 1, f"xyz max {xyz.max()}"

        # verify the shapes are correct
        assert len(xyz.shape) == 2
        assert xyz.shape[0] == weights.shape[0]
        assert xyz.shape[1] == 3
        assert weights.shape[1] == 1

        # update the weights
        # indices = (xyz * self.resolution).long()
        # self._weights[indices[:, 0], indices[:, 1], indices[:, 2]] = torch.maximum(
        #     self._weights[indices[:, 0], indices[:, 1], indices[:, 2]] * ema_decay, weights.squeeze(-1)
        # )
        self._weights = ema_decay * self._weights
        indices = (xyz * self.resolution).long()
        self._weights[indices[:, 0], indices[:, 1], indices[:, 2]] += weights.squeeze(-1)

    def sample(self, num_points: int, randomize: bool = True) -> TensorType["num_points", 3]:
        """Samples points from the grid where the value is above the threshold."""

        device = self._weights.device

        probs = self._weights.view(-1) / self._weights.view(-1).sum()
        dist = torch.distributions.categorical.Categorical(probs)
        sample = dist.sample((num_points,))

        h = torch.div(sample, self.resolution**2, rounding_mode="floor")
        d = sample % self.resolution
        w = torch.div(sample, self.resolution, rounding_mode="floor") % self.resolution

        idx = torch.stack([h, w, d], dim=1).float()
        if randomize:
            return (idx + torch.rand_like(idx).to(device)) / self.resolution
        else:
            return idx / self.resolution


@dataclass
class NerfbustersPipelineConfig(VanillaPipelineConfig):
    """Nerfbusters Pipeline Config"""

    _target: Type = field(default_factory=lambda: NerfbustersPipeline)

    # some default overrides

    # NeRF checkpoint
    nerf_checkpoint_path: Optional[Path] = None

    # 3D diffusion model
    diffusioncube_config_path: Optional[Path] = Path("config/shapenet.yaml")
    diffusioncube_ckpt_path: Optional[Path] = Path("data/nerfbusters-diffusion-cube-weights.ckpt")

    # visualize options
    # what to visualize
    visualize_weights: bool = False
    visualize_cubes: bool = False
    visualize_patches: bool = False
    # how often to visualize
    steps_per_visualize_weights: int = 100
    steps_per_visualize_cubes: int = 1
    steps_per_visualize_patches: int = 10

    # cube sampling
    num_cubes: int = 40
    """Number of cubes per batch for training"""
    cube_resolution: int = 32
    cube_start_step: int = 0
    cube_scale_perc_range: Tuple[float, ...] = (0.01, 0.10)  # percentage of the scene box
    steps_per_draw_cubes: int = 20
    sample_method: Literal["uniform", "importance", "random", "fixed"] = "random"
    fix_samples_for_steps: int = 1
    num_views_per_cube: int = 3
    max_num_cubes_to_visualize: int = 6
    """If we should fix the batch of samples for a couple of steps."""
    fixed_cubes_center: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, -0.15),  # picnic - vase
        # (-0.85, 0.85, -0.5),  # plant - vase
        # (0.03, 0.28, -0.23), # table - chair
        # (-0.07, 0.02, -0.36),  # table - table
    )
    fixed_cubes_scale_perc: Tuple[Tuple[float], ...] = ((0.02),)
    aabb_scalar: float = 1.5

    # weight grid settings
    weight_grid_resolution: int = 100
    weight_grid_quantity: Literal["weights", "densities", "visibility"] = "weights"
    weight_grid_quantity_idx: int = -1
    weight_grid_update_per_step: int = 10

    # density to x
    density_to_x_crossing: float = 0.01
    density_to_x_max: float = 500.0
    density_to_x_activation: Literal[
        "sigmoid",
        "clamp",
        "sigmoid_complex",
        "rescale_clamp",
        "piecewise_linear",
        "batchnorm",
        "meannorm",
        "piecewise_loglinear",
        "piecewise_loglinear_sigmoid",
        "binarize",
        "piecewise_loglinear_threshold",
    ] = "binarize"
    piecewise_loglinear_threshold: float = 1e-3

    # TODO: add noise to densities (from original nerf paper)

    # patch sampling
    num_patches: int = 10
    """Number of patches per batch for training"""
    patch_resolution: int = 32
    """Patch resolution, where DiffRF used 48x48 and RegNeRF used 8x8"""
    focal_range: Tuple[float, float] = (3.0, 3.0)
    """Range of focal length"""
    central_rotation_range: Tuple[float, float] = (-180, 180)
    """Range of central rotation"""
    vertical_rotation_range: Tuple[float, float] = (-90, 20)
    """Range of vertical rotation"""
    jitter_std: float = 0.05
    """Std of camera direction jitter, so we don't just point the cameras towards the center every time"""
    center: Tuple[float, float, float] = (0, 0, 0)
    """Center coordinate of the camera sphere"""

    # -------------------------------------------------------------------------
    # 2D losses

    # regnerf loss
    use_regnerf_loss: bool = False
    regnerf_loss_mult: float = 1e-7

    # -------------------------------------------------------------------------
    # 3D losses

    # cube loss
    use_cube_loss: bool = False
    cube_loss_mult: float = 1e-1
    cube_loss_trange: Tuple[int, ...] = (9, 10)

    # multistep cube loss
    use_multistep_cube_loss: bool = False
    multistep_cube_loss_mult: float = 1e-2
    multistep_range: Tuple[int, ...] = (100, 700)
    num_multistep: int = 10

    # singlestep cube loss
    use_singlestep_cube_loss: bool = True
    singlestep_cube_loss_mult: float = 1e-1
    singlestep_target: float = 0.0
    singlestep_starting_t: int = 10

    # sparsity loss
    use_sparsity_loss: bool = False
    sparsity_loss_mult: float = 1e-2
    sparsity_length: float = 0.05

    # TV loss
    use_total_variation_loss: bool = False
    total_variation_loss_mult: float = 1e-7

    # visibility loss
    use_visibility_loss: bool = True
    visibility_loss_quantity: Literal["weights", "densities"] = "densities"
    visibility_loss_mult: float = 1e-6
    visibility_min_views: int = 1
    """Minimum number of training views that must be seen."""
    visibility_num_rays: int = 10

    # threshold loss
    use_threshold_loss: bool = False
    threshold_loss_mult: float = 1e-1


class NerfbustersPipeline(VanillaPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    # pylint: disable=abstract-method

    config: NerfbustersPipelineConfig

    def __init__(
        self,
        config: NerfbustersPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # initialize visibility "field"
        cameras = self.datamanager.train_dataparser_outputs.cameras.to(self.device)
        if "eval_frame_index_0_metadata" in self.datamanager.train_dataparser_outputs.metadata:
            # this block of code is for computing visibility from only the train images
            # we set this metadata when we are training with train+eval images, but we
            # only want to compute visibility from the train images
            eval_frame_index_0_metadata = self.datamanager.train_dataparser_outputs.metadata[
                "eval_frame_index_0_metadata"
            ]
            cameras = cameras[eval_frame_index_0_metadata == 1]
        self.model.visibility_field = VisibilityField(cameras).to(self.device)

        # initialize weight grid
        self.weight_grid = WeightGrid(resolution=self.config.weight_grid_resolution).to(self.device)

        # keep track of these to visualize cubes
        self.cube_start_xyz = None
        self.cube_end_xyz = None

        # bounding box of the scene
        self.aabb = self.datamanager.train_dataset.scene_box.aabb.to(self.device) * self.config.aabb_scalar

        # loading checkpoints needs to be last...

        # load checkpoint for NeRF pipeline if specified
        if self.config.nerf_checkpoint_path is not None:
            loaded_state = torch.load(self.config.nerf_checkpoint_path, map_location="cpu")
            # remove any keys with diffusion
            for key in list(loaded_state["pipeline"].keys()):
                if "diffusion" in key:
                    del loaded_state["pipeline"][key]
                if "weight_grid" in key:
                    del loaded_state["pipeline"][key]
            self.load_state_dict(loaded_state["pipeline"], strict=False)
            print("Loaded NeRF checkpoint from", self.config.nerf_checkpoint_path)

        # load 3D diffusion model
        self.diffusioncube_model = self.load_diffusion_model(
            self.config.diffusioncube_config_path, self.config.diffusioncube_ckpt_path
        )

        # because these are not registered as parameters, we not to convert to device manually
        self.diffusioncube_model.noise_scheduler.betas = self.diffusioncube_model.noise_scheduler.betas.to(self.device)
        self.diffusioncube_model.noise_scheduler.alphas = self.diffusioncube_model.noise_scheduler.alphas.to(
            self.device
        )
        self.diffusioncube_model.noise_scheduler.alphas_cumprod = (
            self.diffusioncube_model.noise_scheduler.alphas_cumprod.to(self.device)
        )

    def load_diffusion_model(self, diffusion_config_path, diffusion_ckpt_path):
        config = yaml.load(open(diffusion_config_path, "r"), Loader=yaml.Loader)
        config = DotMap(config)
        model = NerfbustersTrainer(config)
        ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model = model.to(self.device)
        model.noise_scheduler.alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(self.device)
        model.dsds_loss.alphas = model.dsds_loss.alphas.to(self.device)
        print("Loaded diffusion config from", diffusion_config_path)
        print("Loaded diffusion checkpoint from", diffusion_ckpt_path)
        return model

    def query_field(self, xyz, method="density"):
        if method == "density":
            # the fine network
            return self.model.field.density_fn(positions=xyz)
            # TODO(ethan): add support for the proposal density networks
            # look at the following line for reference
            # return self.model.density_fns[0](positions=xyz)
        else:
            raise NotImplementedError("Only density is supported for now.")

    def density_to_x(self, density):
        """Converts density to x for diffusion model."""
        if self.config.density_to_x_activation == "sigmoid":
            x = 2 * torch.sigmoid(1000.0 * (density - self.config.density_to_x_crossing)) - 1.0
        elif self.config.density_to_x_activation == "clamp":
            x = torch.clamp(self.config.density_to_x_crossing * density - 1, -1, 1)
        elif self.config.density_to_x_activation == "sigmoid_complex":
            density_to_x_temperature = -1.0 * math.log(1.0 / 3.0) / self.config.density_to_x_crossing
            x = ((torch.sigmoid(density_to_x_temperature * density)) - 0.5) * 4.0 - 1.0
        elif self.config.density_to_x_activation == "binarize":
            x = torch.where(density.detach() < self.config.density_to_x_crossing, -1.0, 1.0)
        elif self.config.density_to_x_activation == "rescale_clamp":
            x = torch.clamp(density / self.config.density_to_x_crossing - 1.0, -1.0, 1.0)
        elif self.config.density_to_x_activation == "piecewise_linear":
            x_fir = density / self.config.density_to_x_crossing - 1.0
            x_sec = (
                1.0
                / (self.config.density_to_x_max - self.config.density_to_x_crossing)
                * (density - self.config.density_to_x_crossing)
            )
            x = torch.where(density < self.config.density_to_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)
        elif self.config.density_to_x_activation == "piecewise_loglinear":
            x_fir = density / self.config.density_to_x_crossing - 1.0
            x_sec = torch.log(1 + density - self.config.density_to_x_crossing) / torch.log(
                torch.tensor(self.config.density_to_x_max)
            )
            x = torch.where(density < self.config.density_to_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)
        elif self.config.density_to_x_activation == "piecewise_loglinear_threshold":
            x_fir = density / self.config.density_to_x_crossing - 1.0
            x_sec = torch.log(1 + density - self.config.density_to_x_crossing) / torch.log(
                torch.tensor(self.config.density_to_x_max)
            )
            x = torch.where(density < self.config.density_to_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)
            x = torch.where(density < self.config.piecewise_loglinear_threshold, -1.0 * torch.ones_like(x), x)
        elif self.config.density_to_x_activation == "piecewise_loglinear_sigmoid":
            density_to_x_temperature = 1500
            x_fir = 2 / (1 + torch.exp(-(density_to_x_temperature * (density - self.config.density_to_x_crossing)))) - 1
            x_sec = torch.log(1 + density - self.config.density_to_x_crossing) / torch.log(
                torch.tensor(self.config.density_to_x_max)
            )
            x = torch.where(density < self.config.density_to_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)
        elif self.config.density_to_x_activation == "piecewise_exp":
            x_fir = density / self.config.density_to_x_crossing - 1.0
            x_sec = (
                1.0
                / (self.config.density_to_x_max - self.config.density_to_x_crossing)
                * (density - self.config.density_to_x_crossing)
            )
            x = torch.where(density < self.config.density_to_x_crossing, x_fir, x_sec).clamp(-1.0, 1.0)
        elif self.config.density_to_x_activation == "batchnorm":
            with torch.no_grad():
                self.counter = 0 if not hasattr(self, "counter") else self.counter + 1
                self.running_mean = (
                    density.mean()
                    if not hasattr(self, "running_mean")
                    else self.running_mean + density.mean() / (self.counter + 1)
                )
                self.running_var = (
                    density.var()
                    if not hasattr(self, "running_var")
                    else self.running_var + density.var() / (self.counter + 1)
                )

            mu = self.running_mean
            sigma = self.running_var.sqrt()
            x = (density - mu) / (sigma + 1e-7)
            x = torch.clamp(x, -1.0, 1.0)
        elif self.config.density_to_x_activation == "meannorm":
            with torch.no_grad():
                self.counter = 0 if not hasattr(self, "counter") else self.counter + 1
                self.running_mean = density.mean()
                # (
                #     density.mean()
                #     if not hasattr(self, "running_mean")
                #     else self.running_mean + density.mean() / (self.counter + 1)
                # )
            x = torch.log(density)  # - torch.mean(torch.log(density))  # / (density.std() + 1e-7)
            x = torch.clamp(x, -1.0, 1.0)

        # piecewise
        # density_crossing = 100.0
        # x_before = density / density_crossing - 1.0
        # x = torch.where(density < density_crossing, x_before, 1.0)
        # print(x.mean())

        # TODO: tanh
        # TODO: rescale and clamp
        return x

    def apply_singlestep_cube_loss(self, step, x, density, res, scales):
        """Returns the 3D cube gradient."""
        num_cubes = x.shape[0]
        x = x.reshape(num_cubes, 1, res, res, res)

        with torch.no_grad():
            xhat, w = self.diffusioncube_model.single_step_reverse_process(
                sample=x,
                starting_t=self.config.singlestep_starting_t,
                scale=scales,
            )

        # binarize
        xhat = torch.where(xhat < 0, -1, 1)
        mask_empty = xhat == -1
        mask_full = xhat == 1
        density = density.unsqueeze(1)
        loss = (density * mask_empty).sum()
        loss += (torch.clamp(self.config.singlestep_target - density, 0) * mask_full).sum()
        loss = loss / math.prod(density.shape)  # average loss
        return self.config.singlestep_cube_loss_mult * loss

    def apply_threshold_loss(self, step, x, density, res, scales):
        target = 0.0

        # binarize
        xhat = torch.where(x < 0, -1, 1)
        mask_empty = xhat == -1
        mask_full = xhat == 1
        density = density.unsqueeze(1)
        loss = (density.abs() * mask_empty).sum()
        loss += (torch.clamp(target - density.abs(), 0) * mask_full).sum()
        loss = loss / math.prod(density.shape)  # average loss
        return self.config.threshold_loss_mult * loss

    def apply_multistep_cube_loss(self, step: int, x, res, scales):
        """Returns the 3D cube gradient."""
        num_cubes = x.shape[0]
        x = x.reshape(num_cubes, 1, res, res, res)

        min_step, max_step = self.config.multistep_range
        num_multistep = self.config.num_multistep
        assert min_step <= max_step
        assert min_step > num_multistep

        # currently denoising starts with the same t for all cubes
        # i think it would be better if the started at different timesteps
        starting_t = torch.randint(min_step, max_step, (1,)).to(self.device).long().item()

        with torch.no_grad():
            xhat = self.diffusioncube_model.reverse_process(
                sample=x,
                scale=scales,
                bs=None,
                num_inference_steps=num_multistep,  # keep rather low for speed
                starting_t=starting_t,
            )

        return self.config.multistep_cube_loss_mult * (x - xhat).pow(2).mean()

    def apply_cube_loss(self, step: int, x, res, scales):
        """Returns the 3D cube gradient."""
        num_cubes = x.shape[0]
        x = x.reshape(num_cubes, 1, res, res, res)

        min_t, max_t = self.config.cube_loss_trange

        # currently denoising starts with the same t for all cubes
        # i think it would be better if the started at different timesteps
        timesteps = torch.randint(min_t, max_t, (1,)).to(self.device).long().item()

        model = self.diffusioncube_model.model
        dsds_loss = self.diffusioncube_model.dsds_loss
        grad = dsds_loss.grad_sds_unconditional(x, model, timesteps, scales, mult=self.config.cube_loss_mult)
        grad_mag = torch.mean(grad**2) ** 0.5
        return grad_mag

    def apply_total_variation_loss(self, step: int, density: TensorType["num_cubes", "res", "res", "res"], res):
        # x = self.density_to_x(density)
        x = density
        delta_x = x[..., 1:, :-1, :-1] - x[..., :-1, :-1, :-1]
        delta_y = x[..., :-1, 1:, :-1] - x[..., :-1, :-1, :-1]
        delta_z = x[..., :-1, :-1, 1:] - x[..., :-1, :-1, :-1]
        loss = torch.mean(delta_x**2 + delta_y**2 + delta_z**2)
        # loss = torch.mean(torch.sqrt(delta_x**2 + delta_y**2 + delta_z**2))
        # loss = torch.mean(torch.abs(delta_x + delta_y + delta_z))
        return loss

    def apply_sparsity_loss(self, step: int, density):
        """Returns the sparsity loss proposed in Plenoxels.
        Args:
            density: (N, 1)
        """

        l = self.config.sparsity_length
        v = torch.abs(1.0 - torch.exp(-l * density))
        v = torch.mean(v)
        return v

    def get_standard_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        # POPULATE THE WEIGHT GRID
        # TODO: make this faster by using less samples?
        if self.config.sample_method in ["importance", "random"] and self.config.weight_grid_quantity in [
            "weights",
            "densities",
        ]:
            if step % self.config.weight_grid_update_per_step == 0:
                with torch.no_grad():
                    ray_samples = model_outputs["ray_samples_list"][self.config.weight_grid_quantity_idx]
                    quantities = model_outputs[f"{self.config.weight_grid_quantity}_list"][
                        self.config.weight_grid_quantity_idx
                    ]
                    # get positions
                    positions = ray_samples.frustums.get_positions()
                    normalized_positions = SceneBox.get_normalized_positions(positions, self.aabb)
                    # UPDATE THE WEIGHT GRID
                    normalized_positions = normalized_positions.view(-1, 3)
                    quant = quantities.view(-1, 1)
                    # only use positions between 0 inclusive and 1 exclusive
                    mask = (normalized_positions >= 0.0) & (normalized_positions < 1.0)
                    mask = mask[:, 0] & mask[:, 1] & mask[:, 2]
                    normalized_positions = normalized_positions[mask]
                    quant = quant[mask]
                    if normalized_positions.numel() != 0:
                        if self.config.weight_grid_quantity == "densities":
                            self.weight_grid.update(normalized_positions, torch.clamp(quant, 0, 1))
                        elif self.config.weight_grid_quantity == "weights":
                            self.weight_grid.update(normalized_positions, quant)
                        else:
                            raise ValueError(f"Unknown weight grid quantity: {self.config.weight_grid_quantity}")

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            # Report the camera optimization metrics
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        # the loss dict
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def apply_regnerf_loss(self, step: int, patches_density: TensorType["num_patches", "res", "res"]):
        pd = patches_density
        delta_x = pd[..., :-1, 1:] - pd[..., :-1, :-1]
        delta_y = pd[..., 1:, :-1] - pd[..., :-1, :-1]
        loss = torch.mean(delta_x**2 + delta_y**2)
        return loss

    def visualize_patches(
        self,
        step: int,
        patches_rgb: TensorType["num_patches", "res", "res", 3],
        patches_density: TensorType["num_patches", "res", "res"],
    ):
        return None

    def visualize_weights(self, step: int):
        import plotly.graph_objects as go

        points = self.weight_grid.sample(num_points=10000)
        points = points.cpu().numpy()
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(color="black", size=3),
                )
            ]
        )
        fig.write_image("weights.png")
        fig.write_html("weights.html")

    def visualize_cubes(self, step: int, x: TensorType["num_cubes", "res", "res", "res"], max_num_cubes: int = 10):
        """
        Args:
            x: The input to the SDS model.
        """
        # TODO: Run SDS optimization on NeRF density cubes and visualize the results
        num_views = self.config.num_views_per_cube
        num_cubes = min(max_num_cubes, x.shape[0])
        with torch.no_grad():
            x_input = x[:num_cubes][:, None]

            with torch.no_grad():
                xhat_input, _ = self.diffusioncube_model.single_step_reverse_process(
                    sample=x_input,
                    starting_t=self.config.singlestep_starting_t,
                    scale=self.scales,
                )
                xhat_input = torch.where(xhat_input < 0, -1, 1).float()
            x_diff = torch.where(torch.abs(xhat_input - x_input) > 0, 1, -1).float()

            def get_image_grid_from_x(xin):
                depth_maps, normal_maps = get_3Dimage_fast(xin, num_views=num_views, format=False, th=0.0)
                depth_maps = depth_maps.reshape(num_views, num_cubes, *depth_maps.shape[-3:])
                normal_maps = normal_maps.reshape(num_views, num_cubes, *normal_maps.shape[-3:])
                image_grid = []
                for i in range(num_cubes):
                    image_row = []
                    for j in range(num_views):
                        image_row.append(normal_maps[j, i])
                    image_row = np.hstack(image_row)
                    image_grid.append(image_row)
                image_grid = np.vstack(image_grid)
                return image_grid

            image_grid_x = get_image_grid_from_x(x_input)
            image_grid_xhat = get_image_grid_from_x(xhat_input)
            image_grid_xdiff = get_image_grid_from_x(x_diff)
            writer.put_image(name="visualize_cubes/image_grid_x", image=torch.tensor(image_grid_x), step=step)
            writer.put_image(name="visualize_cubes/image_grid_xhat", image=torch.tensor(image_grid_xhat), step=step)
            writer.put_image(name="visualize_cubes/image_grid_xdiff", image=torch.tensor(image_grid_xdiff), step=step)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        model_outputs, loss_dict, metrics_dict = self.get_standard_loss_dict(step)

        # --------------------- 2D losses ---------------------
        activate_patch_sampling = self.config.use_regnerf_loss

        # TODO: debug why patch sampling decreases model performance
        if activate_patch_sampling:
            cameras, vertical_rotation, central_rotation = random_train_pose(
                size=self.config.num_patches,
                resolution=self.config.patch_resolution,
                device=self.device,
                radius_mean=self.config.aabb_scalar,  # no sqrt(3) here
                radius_std=0.0,
                central_rotation_range=self.config.central_rotation_range,
                vertical_rotation_range=self.config.vertical_rotation_range,
                focal_range=self.config.focal_range,
                jitter_std=self.config.jitter_std,
                center=self.config.center,
            )
            # TODO(ethan): fix indices
            camera_indices = torch.tensor(list(range(self.config.num_patches))).unsqueeze(-1)
            ray_bundle_patches = cameras.generate_rays(
                camera_indices
            )  # (patch_resolution, patch_resolution, num_patches)
            ray_bundle_patches = ray_bundle_patches.flatten()
            # TODO: deal with appearance embeddings
            model_outputs_patches = self.model(ray_bundle_patches)
            rgb_patches = (
                model_outputs_patches["rgb"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, 3)
                .permute(2, 0, 1, 3)
            )  # (num_patches, patch_resolution, patch_resolution, 3)
            depth_patches = (
                model_outputs_patches["depth"]
                .reshape(self.config.patch_resolution, self.config.patch_resolution, self.config.num_patches, 1)
                .permute(2, 0, 1, 3)[..., 0]
            )  # (num_patches, patch_resolution, patch_resolution)

        if self.config.use_visibility_loss:
            # randomly sample a center within the aabb
            center = (torch.rand(3, device=self.device) * (self.aabb[1] - self.aabb[0]) + self.aabb[0]).tolist()
            cameras, _, _ = random_train_pose(
                size=self.config.visibility_num_rays,
                resolution=1,
                device=self.device,
                radius_mean=self.config.aabb_scalar * math.sqrt(3.0),
                radius_std=0.0,
                central_rotation_range=(0, 360),
                vertical_rotation_range=(-90, 90),
                focal_range=self.config.focal_range,
                jitter_std=0,
                center=center,
            )
            # We only use the first camera index, but it doesn't matter since we don't care about appearance embeddings
            # when using the visibility loss.
            camera_indices = torch.tensor([0] * self.config.visibility_num_rays).unsqueeze(-1)
            ray_bundle_rays = cameras.generate_rays(camera_indices)
            # ray_bundle is (1, 1, self.config.visibility_num_rays)
            ray_bundle_rays = ray_bundle_rays.flatten()
            model_outputs_rays = self.model(ray_bundle_rays)
            quantity_list = model_outputs_rays[f"{self.config.visibility_loss_quantity}_list"]  # weights or densities
            visibility_loss = 0.0
            for i in range(len(quantity_list)):
                quantity_samples = quantity_list[i]
                ray_samples = model_outputs_rays["ray_samples_list"][i]
                visibility_samples = self.model.visibility_field(ray_samples)

                with torch.no_grad():
                    if (
                        self.config.sample_method in ["importance", "random"]
                        and self.config.weight_grid_quantity == "visibility"
                    ):
                        if step % self.config.weight_grid_update_per_step == 0:
                            if i - 1 == len(quantity_list):
                                # update the weight grid
                                # get positions
                                positions = ray_samples.frustums.get_positions()
                                print(positions.shape)
                                normalized_positions = SceneBox.get_normalized_positions(positions, self.aabb)
                                # UPDATE THE WEIGHT GRID
                                normalized_positions = normalized_positions.view(-1, 3)
                                quant = visibility_samples.view(-1, 1)
                                # only use positions between 0 inclusive and 1 exclusive
                                mask = (normalized_positions >= 0.0) & (normalized_positions < 1.0)
                                mask = mask[:, 0] & mask[:, 1] & mask[:, 2]
                                normalized_positions = normalized_positions[mask]
                                quant = quant[mask]
                                if normalized_positions.numel() != 0:
                                    self.weight_grid.update(normalized_positions, quant)

                quantity_samples_masked = quantity_samples[visibility_samples < self.config.visibility_min_views]
                if quantity_samples_masked.numel() != 0:
                    visibility_loss_i = self.config.visibility_loss_mult * quantity_samples_masked.mean()
                    visibility_loss += visibility_loss_i
            loss_dict["visibility_loss"] = visibility_loss

        if self.config.use_regnerf_loss:
            regnerf_loss = self.apply_regnerf_loss(step, depth_patches)
            loss_dict["regnerf_loss"] = self.config.regnerf_loss_mult * regnerf_loss

        # --------------------- 2D visualization -----------------------------------
        if self.config.visualize_patches and step % self.config.steps_per_visualize_patches == 0:
            assert activate_patch_sampling, "Cannot visualize patches without patch sampling."
            patches = []
            for idx in range(self.config.num_patches):
                patch = rgb_patches[idx].detach().cpu().numpy()
                patch = (patch * 255).astype("uint8")
                patches.append(patch)
            cols = 10

            rows = math.ceil(self.config.num_patches / cols)
            patches_im = get_image_grid(patches, rows=rows, cols=cols)
            media.write_image(f"patches.png", patches_im)

        # --------------------- 3D losses ---------------------
        activate_cube_sampling = (
            (self.config.use_cube_loss and step >= self.config.cube_start_step)
            or self.config.use_total_variation_loss
            or self.config.use_sparsity_loss
            or self.config.use_threshold_loss
            or (self.config.use_multistep_cube_loss and step >= self.config.cube_start_step)
            or (self.config.use_singlestep_cube_loss and step >= self.config.cube_start_step)
        )

        if activate_cube_sampling:
            # sample the cubes
            min_x, min_y, min_z = self.aabb[0]
            max_x, max_y, max_z = self.aabb[1]
            res = self.config.cube_resolution
            spr_min, spr_max = self.config.cube_scale_perc_range

            if step % self.config.fix_samples_for_steps == 0:
                # get centers from weight grid
                sample_method = self.config.sample_method
                if sample_method == "random":
                    sample_method = "importance" if float(torch.rand(1)) < 0.5 else "uniform"
                if sample_method == "importance":
                    centers = self.weight_grid.sample(num_points=self.config.num_cubes)
                    self.xyz, self.cube_start_xyz, self.cube_end_xyz, self.scales = sample_cubes(
                        min_x,
                        min_y,
                        min_z,
                        max_x,
                        max_y,
                        max_z,
                        res,
                        spr_min,
                        spr_max,
                        self.config.num_cubes,
                        cube_centers_x=centers[:, 0],
                        cube_centers_y=centers[:, 1],
                        cube_centers_z=centers[:, 2],
                        device=self.device,
                    )
                elif sample_method == "uniform":
                    self.xyz, self.cube_start_xyz, self.cube_end_xyz, self.scales = sample_cubes(
                        min_x,
                        min_y,
                        min_z,
                        max_x,
                        max_y,
                        max_z,
                        res,
                        spr_min,
                        spr_max,
                        self.config.num_cubes,
                        device=self.device,
                    )
                elif sample_method == "fixed":
                    # use a fixed set of cubes
                    fixed_cubes_center = torch.tensor(self.config.fixed_cubes_center, device=self.device)  # (N, 3)
                    num_cubes = fixed_cubes_center.shape[0]
                    scales = (
                        torch.tensor(self.config.fixed_cubes_scale_perc, device=self.device) / self.config.aabb_scalar
                    )  # (N)
                    cube_len = (max_x - min_x) * scales.unsqueeze(-1)
                    self.cube_start_xyz = (fixed_cubes_center - cube_len / 2).reshape(num_cubes, 1, 1, 1, 3)
                    self.cube_end_xyz = (fixed_cubes_center + cube_len / 2).reshape(num_cubes, 1, 1, 1, 3)
                    # TODO: move this part to a function
                    l = torch.linspace(0, 1, res, device=self.device)
                    xyz = torch.stack(torch.meshgrid([l, l, l], indexing="ij"), dim=-1)  # (res, res, res, 3)
                    xyz = xyz[None, ...] * (self.cube_end_xyz - self.cube_start_xyz) + self.cube_start_xyz
                    self.xyz = xyz
                    self.scales = scales

                    # TODO: add tiling params
                else:
                    raise ValueError(f"Unknown sample method {self.config.sample_method}")

            # sample the nerf density with the cubes
            self.xyz = self.xyz.to(self.device)
            density = self.query_field(self.xyz).squeeze(-1)  # (num_cubes, res, res, res)

            # some random point in the scene, xyz
            # a = torch.rand(3, device=self.device)

            # a = torch.tensor([-1.0, .78, 0.0], device=self.device)[None] + torch.rand(3, device=self.device)[None] * 0.0002
            # a = torch.tensor([-.5, .5, 0], device=self.device)[None]
            # b = self.query_field(a)[0]
            # c = self.density_to_x(b)
            # print("a, b, c", a, b, c)

            assert density.min() >= 0.0, f"density.min()={density.min()}"

            # move density to valid range for diffusion model
            x = self.density_to_x(density)
            # x_shifted = self.density_to_x(density_shifted)

            # stats for the sampled cubes
            metrics_dict["cube_density_mean"] = torch.mean(density)
            metrics_dict["cube_density_std"] = torch.std(density)
            metrics_dict["cube_density_min"] = torch.min(density)
            metrics_dict["cube_density_max"] = torch.max(density)
            metrics_dict["cube_density_median"] = torch.median(density)
            metrics_dict["cube_x_mean"] = torch.mean(x)
            metrics_dict["cube_x_std"] = torch.std(x)
            metrics_dict["cube_x_min"] = torch.min(x)
            metrics_dict["cube_x_max"] = torch.max(x)
            metrics_dict["cube_x_median"] = torch.median(x)

        # --------------------- 3D visualization ---------------------
        # visualize the weights grid
        if self.config.visualize_weights and step % self.config.steps_per_visualize_weights == 0:
            self.visualize_weights(step)

        # visualize the sds cubes
        if self.config.visualize_cubes and step % self.config.steps_per_visualize_cubes == 0:
            self.visualize_cubes(step, x, max_num_cubes=self.config.max_num_cubes_to_visualize)

        # multistep cube loss
        if self.config.use_multistep_cube_loss:
            assert x is not None
            assert x.min() >= -1.0 and x.max() <= 1.0, f"x.min()={x.min()}, x.max()={x.max()}"

            loss = self.apply_multistep_cube_loss(step=step, x=x, res=res, scales=self.scales)
            loss_dict["ms_cube_loss"] = loss

        # single step cube loss
        if self.config.use_singlestep_cube_loss:
            assert x is not None
            assert x.min() >= -1.0 and x.max() <= 1.0, f"x.min()={x.min()}, x.max()={x.max()}"

            loss = self.apply_singlestep_cube_loss(step=step, x=x, density=density, res=res, scales=self.scales)
            loss_dict["ss_cube_loss"] = loss

        # threshold loss
        if self.config.use_threshold_loss:
            assert x is not None
            assert x.min() >= -1.0 and x.max() <= 1.0, f"x.min()={x.min()}, x.max()={x.max()}"

            loss = self.apply_threshold_loss(step=step, x=x, density=density, res=res, scales=self.scales)
            loss_dict["threshold_loss"] = loss

        # cube loss
        if self.config.use_cube_loss:
            # sets the gradient with the SDS loss

            assert x is not None
            assert x.min() >= -1.0 and x.max() <= 1.0, f"x.min()={x.min()}, x.max()={x.max()}"

            grad_mag = self.apply_cube_loss(step=step, x=x, res=res, scales=self.scales)
            metrics_dict["cube_loss"] = grad_mag

            # stats for the sampled cubes
            metrics_dict["cube_density_mean"] = torch.mean(density)
            metrics_dict["cube_density_std"] = torch.std(density)
            metrics_dict["cube_density_min"] = torch.min(density)
            metrics_dict["cube_density_max"] = torch.max(density)
            metrics_dict["cube_x_mean"] = torch.mean(x)
            metrics_dict["cube_x_std"] = torch.std(x)
            metrics_dict["cube_x_min"] = torch.min(x)
            metrics_dict["cube_x_max"] = torch.max(x)

        if self.config.use_total_variation_loss:
            total_variation_loss = self.apply_total_variation_loss(step=step, density=density, res=res)
            loss_dict["total_variation_loss"] = total_variation_loss * self.config.total_variation_loss_mult

        if self.config.use_sparsity_loss:
            # use the sampled cubes
            sparsity_loss = ((x + 1.0) ** 2).mean()
            loss_dict["sparsity_loss"] = sparsity_loss * self.config.sparsity_loss_mult

        return model_outputs, loss_dict, metrics_dict

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks

        def draw_cubes(step):
            training_callback_attributes.viewer_state.vis["sceneState/cubes"].delete()

            if self.cube_start_xyz is None or self.cube_end_xyz is None:
                return

            for idx in range(self.cube_start_xyz.shape[0]):
                json_ = {
                    "type": "aabb",
                    "min_point": [float(v) for v in self.cube_start_xyz[idx].flatten()],
                    "max_point": [float(v) for v in self.cube_end_xyz[idx].flatten()],
                }
                training_callback_attributes.viewer_state.vis[f"sceneState/cubes/{idx:06d}"].write(json_)

        # if the visualizer is enabled
        if training_callback_attributes.viewer_state:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=self.config.steps_per_draw_cubes,
                    func=draw_cubes,
                )
            )
        return callbacks
