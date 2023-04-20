import torch
from scipy.spatial.transform import Rotation as R
from .binvox_rw import read_as_3d_array
import numpy as np


def get_random_rotation_matrix():
    # return a random rotation matrix
    v = torch.rand(3)
    length = torch.sqrt(torch.sum(v**2))
    v = v / length
    v *= torch.rand(1) * 2 * torch.pi
    rotation_matrix = R.from_rotvec(v).as_matrix()
    rotation_matrix = torch.from_numpy(rotation_matrix)
    return rotation_matrix


def read_binvox(fpath, verbose=False):
    # read and return binvox data
    with open(fpath, "rb") as f:
        voxels = read_as_3d_array(f)
    if verbose:
        print(f"Read 3d voxel grid sized {list(voxels.data.shape)} from {fpath}")
    zero_5 = np.array([0.5, 0.5, 0.5])
    xyz_min = zero_5 / voxels.dims * voxels.scale + voxels.translate
    xyz_max = (voxels.dims - zero_5) / voxels.dims * voxels.scale + voxels.translate
    return torch.from_numpy(voxels.data), torch.from_numpy(xyz_min), torch.from_numpy(xyz_max)
