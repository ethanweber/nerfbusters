"""
Datasets for 3D crops.
"""

from __future__ import annotations

import contextlib
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
# import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from nerfbusters.cubes.utils import get_random_rotation_matrix, read_binvox
from pysdf import SDF

# from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_fill_holes
from torch.utils.data import IterableDataset

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils.eval_utils import eval_setup


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        list_ = []
        for m in scene_or_mesh.geometry.values():
            if not isinstance(m, trimesh.path.Path3D) and not isinstance(m, trimesh.points.PointCloud):
                list_.append(trimesh.Trimesh(vertices=m.vertices, faces=m.faces))
        mesh = trimesh.util.concatenate(list_)
    else:
        mesh = scene_or_mesh
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


@dataclass
class Crop:
    """A 3D crop from a scene.
    The shape of the tensors is [..., X_res, Y_res, Z_res, :]
    """

    # shape is [X, Y, Z, ...]
    # TODO: rename origins to xyz
    origins: Optional[torch.Tensor] = None
    rgb: Optional[torch.Tensor] = None
    density: Optional[torch.Tensor] = None
    occupancy: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None

    def __str__(self) -> str:
        origins_shape = self.origins.shape if self.origins is not None else None
        rgb_shape = self.rgb.shape if self.rgb is not None else None
        density_shape = self.density.shape if self.density is not None else None
        occupancy_shape = self.occupancy.shape if self.occupancy is not None else None
        return f"Crop class. origins: {origins_shape}, rgb: {rgb_shape}, density: {density_shape}, occupancy: {occupancy_shape}"


class NerfstudioDataset(IterableDataset):
    """Dataset for crops from NeRFs trained with Nerfstudio."""

    def __init__(
        self,
        load_config: Path,
        num_crops: int = 100,
        max_crop_size=0.5,
        min_crop_size=0.4,
        resolution=32,
        device="cpu",
    ) -> None:
        super().__init__()

        self.load_config = load_config
        self.num_crops = num_crops
        self.max_crop_size = max_crop_size
        self.min_crop_size = min_crop_size
        self.resolution = resolution
        self.device = device

        self.count = 0

        # load the pipeline
        _, self.pipeline, _ = eval_setup(Path(self.load_config))

        scene_box = self.pipeline.datamanager.train_dataparser_outputs.scene_box
        self.min_coord = scene_box.aabb[0]
        self.max_coord = scene_box.aabb[1]

    def get_crop(self, min_coord, max_coord, max_crop_size, min_crop_size, resolution):
        """Choose a 3D crop from the scene."""
        random_value = torch.rand(1)
        crop_scale = min_crop_size * random_value + max_crop_size * (1 - random_value)
        # range where we can sample boxes
        valid_range = max_coord - min_coord - crop_scale  # [0, scene box size - crop size]
        # define the box
        box_min = min_coord + valid_range * torch.rand(3)
        box_max = box_min + crop_scale
        # find the samples within the box
        x = torch.linspace(box_min[0], box_max[0], resolution)
        y = torch.linspace(box_min[1], box_max[1], resolution)
        z = torch.linspace(box_min[2], box_max[2], resolution)
        # make a grid
        grid = torch.stack(torch.meshgrid([x, y, z], indexing="ij"), dim=-1).to(self.device)

        origins = grid
        dirs = torch.zeros_like(origins)
        starts = torch.zeros_like(origins[..., :1])
        ends = torch.ones_like(origins[..., :1])
        zeros = torch.zeros_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts,
                ends=ends,
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
        )

        with torch.no_grad():
            field_outputs = self.pipeline.model.field(ray_samples, compute_normals=False)

        rgb = field_outputs[FieldHeadNames.RGB]
        density = field_outputs[FieldHeadNames.DENSITY]

        # compute visiblity with the training views
        crop = Crop(origins, rgb, density, scale=crop_scale)
        return crop

    def __next__(self):
        if self.count < self.num_crops:
            self.count += 1
            return self.get_crop(
                self.min_coord, self.max_coord, self.max_crop_size, self.min_crop_size, self.resolution
            )
        else:
            self.count = 0
            raise StopIteration

    def __iter__(self):
        return self


class SyntheticMeshDataset(IterableDataset):
    """Dataset to create crops from a single mesh."""

    def __init__(
        self,
        mesh_filename: Optional[str] = None,
        process_directory=None,
        voxel_method: Literal["sdf", "binvox"] = "sdf",
        binvox_path="bins/binvox",
        binvox_resolution=1024,
        binvox_voxels_surface_filename: Optional[str] = None,
        binvox_voxels_filename: Optional[str] = None,
        device="cpu",
    ) -> None:
        """Initialize the dataset.

        Args:
            mesh_filename: path to the mesh file
            process_directory: directory where we can store temporary files
        """
        super().__init__()
        self.mesh_filename = mesh_filename
        self.process_directory = process_directory
        self.voxel_method = voxel_method
        self.binvox_path = binvox_path
        self.binvox_resolution = binvox_resolution
        self.binvox_voxels_surface_filename = binvox_voxels_surface_filename
        self.binvox_voxels_filename = binvox_voxels_filename
        self.device = device

        if self.mesh_filename:
            # process the mesh
            assert self.binvox_voxels_surface_filename is None, "Cannot specify both mesh and voxel surface."
            assert self.binvox_voxels_filename is None, "Cannot specify both mesh and voxel."
            self.initialize()

        if self.binvox_voxels_surface_filename:
            # load the binvox voxels (both the surface and infilled voxels)
            assert self.binvox_voxels_filename is not None, "Must specify both voxel surface and voxel."
            self.voxels_surface = torch.from_numpy(np.load(self.binvox_voxels_surface_filename)).to(self.device).bool()
            self.voxels = torch.from_numpy(np.load(self.binvox_voxels_filename)).to(self.device).bool()

    def initialize(self):

        # load the mesh without processing
        scene_or_mesh = trimesh.load(self.mesh_filename, process=True)
        self.mesh = as_mesh(scene_or_mesh)
        self.is_watertight = trimesh.repair.fill_holes(self.mesh)

        # center and scale between [-1, 1]
        min_ = self.mesh.vertices.min(axis=0)
        max_ = self.mesh.vertices.max(axis=0)
        max_dist = max(max_ - min_)
        mid = (max_ + min_) / 2.0
        self.mesh.vertices = (self.mesh.vertices - mid) / (0.5 * max_dist)
        assert (self.mesh.vertices.max(axis=0) <= 1.0).all()
        assert (self.mesh.vertices.min(axis=0) >= -1.0).all()

        # self.mesh.vertices = 2.0 * self.mesh.vertices - 1.0

        self.vertices = torch.from_numpy(self.mesh.vertices)
        self.faces = torch.from_numpy(self.mesh.faces)

        # save the SDF already
        # TODO: maybe move this somewhere else
        if self.voxel_method == "sdf":
            self.sdf = SDF(self.vertices, self.faces)
        elif self.voxel_method == "binvox":
            assert self.process_directory is not None, "Must provide a process directory to use binvox"
            mesh_filename = os.path.join(self.process_directory, "mesh.obj")
            binvox_filename = os.path.join(self.process_directory, "mesh.binvox")
            if os.path.exists(mesh_filename):
                os.remove(mesh_filename)
            if os.path.exists(binvox_filename):
                os.remove(binvox_filename)
            self.export_mesh(mesh_filename)
            minx, miny, minz = -1, -1, -1
            maxx, maxy, maxz = 1, 1, 1
            print("Running binvox...")
            binvox_cmd = f"{self.binvox_path} -pb -e -t binvox -d {self.binvox_resolution} -bb {minx} {miny} {minz} {maxx} {maxy} {maxz} {mesh_filename}"
            _ = subprocess.run(binvox_cmd, capture_output=True, shell=True)
            print(binvox_cmd)

            voxels, self.xyz_min, self.xyz_max = read_binvox(binvox_filename)
            self.voxels_surface = voxels.to(self.device).bool()
            print("Running binary fill holes...")
            self.voxels = torch.from_numpy(binary_fill_holes(voxels.int())).to(self.device).bool()

    def export_mesh(self, filename):
        """Export the mesh to a file."""
        # make directory is needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.mesh.export(filename)

    def crop_from_binvox(self, origins, trilinear=True):
        # voxelize the mesh with binvox

        if trilinear:
            voxels = self.voxels.unsqueeze(0).unsqueeze(0).float()
            h, w, d, _ = origins.shape
            origins = origins.view(-1, 3).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            occupancy = F.grid_sample(voxels, origins, mode="bilinear", padding_mode="reflection")
            occupancy = occupancy.view(h, w, d)
        else:
            # [-1, 1] -> [0, 1] - > [0, binvox_resolution]
            indices = ((origins + 1) / 2 * self.binvox_resolution).long().clamp(0, self.binvox_resolution - 1)
            x_indices = indices[..., 0]
            y_indices = indices[..., 1]
            z_indices = indices[..., 2]
            occupancy = self.voxels[x_indices, y_indices, z_indices]

        crop = Crop(origins=origins.cpu(), occupancy=occupancy.cpu())

        return crop

    def get_crop(
        self,
        resolution: int = 32,
        crop_percent: float = 0.1,
        apply_random_rotation=True,
        center_crop=False,
        trilinear=True,
        crop_offset: bool = True,
    ) -> Crop:
        """Choose a 3D crop from the scene.
        Args:
            resolution: resolution of the voxel grid that we want to create
            crop_percent: percent of the rotated OBJ file (in the diagonal direction) that we want to crop
            apply_random_rotation: if True, apply a random rotation to the crop
        """

        xyz_min = torch.tensor([-1.0, -1.0, -1.0]).to(self.device)
        xyz_max = torch.tensor([1.0, 1.0, 1.0]).to(self.device)

        # get the length of the diagonal of the bounding box
        diag = torch.norm(xyz_max - xyz_min)
        crop_size = float(crop_percent * diag / torch.sqrt(torch.tensor(3.0)))
        half_crop_size = crop_size / 2.0

        if center_crop:
            # choose a center for the crop
            center = (xyz_min + xyz_max) / 2
            center = center.float().to(self.device)
        else:

            # TODO: allow for random location
            # random location
            # r = torch.rand(3)
            # center = r * (xyz_max - half_crop_size) + (1 - r) * (xyz_min + half_crop_size)

            if self.voxel_method == "binvox":
                # random location on the surface if binvox
                indices = self.voxels_surface.nonzero(as_tuple=False)  # (N, 3)
                index = indices[random.randint(0, len(indices) - 1)]  # (3)
                center = (index / self.binvox_resolution) * 2.0 - 1.0
                center = torch.clamp(center, min=-1 + half_crop_size, max=1 - half_crop_size)
                center = center.float().to(self.device)
            elif self.voxel_method == "sdf":
                # sample random location on the surface of the mesh
                samples, face_index = trimesh.sample.sample_surface(self.mesh, count=1)
                center = torch.from_numpy(samples[0]).float().to(self.device)
                print(samples)

            # maybe apply an offset
            if crop_offset:
                center += (torch.rand(3) - 0.5).to(self.device) * crop_size

        # define the bounds of the crop
        minx = center[0] - half_crop_size
        miny = center[1] - half_crop_size
        minz = center[2] - half_crop_size
        maxx = center[0] + half_crop_size
        maxy = center[1] + half_crop_size
        maxz = center[2] + half_crop_size

        x = torch.linspace(minx, maxx, resolution)
        y = torch.linspace(miny, maxy, resolution)
        z = torch.linspace(minz, maxz, resolution)
        # (X_res, Y_res, Z_res, 3)

        origins = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).float().to(self.device)

        # possibly apply a random rotation here
        if apply_random_rotation:
            R = get_random_rotation_matrix().float().to(self.device)
            origins = (origins - center) @ R.T + center

        if trilinear:
            origins = origins @ torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).float().to(self.device)

        if self.voxel_method == "sdf":
            # CODE THAT USES SDF: INSIDE OUTSIDE CHECK

            distances = torch.from_numpy(self.sdf(origins.reshape(-1, 3).cpu().numpy())).to(self.device)
            distances = distances.reshape(resolution, resolution, resolution)
            occupancy = distances >= 0  # inside and on the surface
            crop = Crop(origins=origins.cpu(), occupancy=occupancy.cpu())
        elif self.voxel_method == "binvox":
            # CODE THAT USES BINVOX
            crop = self.crop_from_binvox(origins, trilinear=trilinear)

        crop.scale = crop_percent

        return crop
