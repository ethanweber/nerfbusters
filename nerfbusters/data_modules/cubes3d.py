import os
import random
import string
import subprocess
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from nerfbusters.cubes.datasets3D import SyntheticMeshDataset
from nerfbusters.cubes.utils import read_binvox
from nerfbusters.utils.utils import get_gaussian_kernel1d
from scipy import ndimage as ndi
from torch.utils import data
from tqdm import tqdm


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def export_mesh(result, filename):
    """Export the mesh to a file."""
    # make directory is needed
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    result.export(filename)


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


class Cubes3D(data.Dataset):
    """Dataset that loads 3D cubes."""

    def __init__(
        self,
        path,
        train=True,
        percentage_of_empty_cubes=0.6,
        dilation_iterations=(0, 5),
        percent_of_scene=(0.01, 0.1),
        val_len=None,
        train_len=None,
        binvox_path="bins/binvox",
    ) -> None:
        super().__init__()

        self.path = path

        self.train = train
        self.percentage_of_empty_cubes = percentage_of_empty_cubes
        self.dilation_iterations = dilation_iterations
        self.percent_of_scene = percent_of_scene
        self.val_len = val_len
        self.train_len = train_len
        self.binvox_path = binvox_path
        self.gpu_id = torch.cuda.current_device()

        self.num_meshes = 3
        self.num_cubes_per_meshes = 1000

        # load the cubes to the CPU
        self.filenames = []
        with open(os.path.join(path, "train_labels.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(" ")
                self.filenames.append(line)

        # create preprocessing directory
        os.makedirs("tmp", exist_ok=True)

        if not self.train:
            torch.manual_seed(0)

            self.noise = torch.randn((1, 32, 32, 32))
            self.noise_levels = 0.5
            self.test_angle = 0
            self.test_percent_of_scene = 0.05
            self.test_dilation_iterations = 3

    def __len__(self) -> int:
        if self.train:
            if self.train_len is not None:
                assert (
                    len(self.filenames) >= self.train_len
                ), f"train_len={self.train_len} but len(self.cubes)={len(self.filenames)}"
                return self.train_len
            else:
                return self.num_meshes * self.num_cubes_per_meshes
        else:
            # same logic for eval
            if self.val_len is not None:
                assert (
                    len(self.filenames) >= self.val_len
                ), f"val_len={self.val_len} but len(self.cubes)={len(self.filenames)}"
                return self.val_len
            else:
                return len(self.filenames)

    def reset(self):
        self.cubes = []
        self.scales = []

        mesh_idx = np.random.randint(0, len(self.filenames), size=self.num_meshes)
        for i in mesh_idx:
            self.process_mesh(i)

    def load_mesh(self, mesh_path):

        # load mesh
        scene_or_mesh = trimesh.load(mesh_path, process=False)
        mesh = as_mesh(scene_or_mesh)

        # normalize mesh
        min_ = mesh.vertices.min(axis=0)
        max_ = mesh.vertices.max(axis=0)
        max_dist = max(max_ - min_)
        mid = (max_ + min_) / 2.0
        mesh.vertices = (mesh.vertices - mid) / (0.5 * max_dist)

        return mesh

    def process_mesh(self, index):

        mesh_path = os.path.join(self.path, self.filenames[index][0], "models", "model_normalized.obj")
        mesh = self.load_mesh(mesh_path)

        for i in tqdm(range(self.num_cubes_per_meshes)):
            cube, scale = self.create_cube(mesh, cube_name=f"tmp/cube")
            self.cubes.append(cube)
            self.scales.append(scale)

    def create_cube(self, mesh, cube_name):

        # pick random scale
        percent_of_scene = (
            np.random.uniform(self.percent_of_scene[0], self.percent_of_scene[1])
            if self.train
            else self.test_percent_of_scene
        )

        # pick random rotation
        angle = np.random.uniform(0, 2 * np.pi) if self.train else self.test_angle

        # pick random dilation
        iterations = (
            np.random.randint(self.dilation_iterations[0], self.dilation_iterations[1])
            if self.train
            else self.test_dilation_iterations
        )

        # pick random kernel size and sigma
        # sigma = np.random.uniform(0.1, 7) if self.train else 2

        # choose random center
        idx = np.random.randint(0, mesh.vertices.shape[0])
        center = mesh.vertices[idx]
        mesh.vertices = mesh.vertices - center[None, :]

        # rotate mesh
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [1, 1, 1]))

        # crop mesh
        box = trimesh.creation.box(extents=[percent_of_scene, percent_of_scene, percent_of_scene])
        mesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)

        # renormalize
        min_ = mesh.vertices.min(axis=0)
        max_ = mesh.vertices.max(axis=0)
        max_dist = max(max_ - min_)
        mid = (max_ + min_) / 2.0
        mesh.vertices = (mesh.vertices - mid) / (0.5 * max_dist)

        # save mesh
        export_mesh(mesh, cube_name + ".obj")

        # run binvox
        binvox_resolution = 32

        minx, miny, minz = -1, -1, -1
        maxx, maxy, maxz = 1, 1, 1

        binvox_cmd = f"{self.binvox_path} -pb -e -t binvox -d {binvox_resolution} -bb {minx} {miny} {minz} {maxx} {maxy} {maxz} {cube_name + '.obj'}"
        _ = subprocess.run(binvox_cmd, shell=True, check=True, stdout=subprocess.PIPE)

        # load binvox
        voxel_surface, xyz_min, xyz_max = read_binvox(cube_name + ".binvox")

        # dilate
        diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        cube = ndi.binary_dilation(voxel_surface, diamond, iterations=iterations)

        # gaussian blur
        # cube = ndi.gaussian_filter(cube.astype(np.float32), sigma=sigma)

        if os.path.exists(cube_name + ".obj"):
            os.remove(cube_name + ".obj")
        if os.path.exists(cube_name + ".binvox"):
            os.remove(cube_name + ".binvox")

        return cube, percent_of_scene

    def __getitem__(self, index: int):

        # randomly choose to return an empty cube
        if self.train and (random.random() < self.percentage_of_empty_cubes):
            # k = torch.rand(0, 1)
            percent_of_scene = np.random.uniform(self.percent_of_scene[0], self.percent_of_scene[1])
            return {"input": torch.ones(1, 32, 32, 32) * 2.0 - 1.0, "scale": percent_of_scene}

        worker_id = torch.utils.data.get_worker_info().id
        letters = string.ascii_lowercase
        tmp_name = letters[worker_id]
        tmp_name += letters[self.gpu_id]

        name = self.filenames[index][0]
        path = os.path.join(self.path, name, "models", "model_normalized.obj")
        mesh = self.load_mesh(path)
        cube, percent_of_scene = self.create_cube(mesh, cube_name=f"tmp/{tmp_name}")

        cube = torch.from_numpy(cube).float().unsqueeze(0)

        if self.train:

            return {"input": cube * 2.0 - 1.0, "scale": percent_of_scene}
        else:
            cube = cube * 2.0 - 1.0
            noise_level = self.noise_levels

            # add low frequency noise
            k = get_gaussian_kernel1d(31, 3)
            k3d = torch.einsum("i,j,k->ijk", k, k, k)
            k3d = k3d / k3d.sum()
            x_smooth = F.conv3d(
                cube.reshape(1, *cube.shape), k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2
            )
            low_freq_noise = cube - x_smooth.squeeze(0)
            noise = self.noise + low_freq_noise

            corrupted_cube = noise_level * noise + (1 - noise_level) * cube
            corrupted_cube = torch.clamp(corrupted_cube, -1.0, 1.0)

            return {"corrupted_input": corrupted_cube, "input": cube, "scale": percent_of_scene}
