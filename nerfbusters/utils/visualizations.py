import math
import os
import time

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision
import wandb
from nerfbusters.cubes.datasets3D import Crop
from nerfbusters.cubes.visualize3D import (
    get_image_grid,
    render_mesh_from_multiple_views,
    write_crop_to_mesh,
)
from nerfbusters.utils.utils import get_gaussian_kernel1d


def save_voxel_as_point_cloud(voxels, filename, color=[1, 0.706, 0], offsets=[0, 0, 0]):

    # voxels: C, H, W, D
    assert len(voxels.shape) == 4
    C, H, W, D = voxels.shape

    points = torch.where(voxels[0] > 0.0)
    points = torch.stack(points).T.cpu()  # N, 3
    points = points + torch.tensor(offsets).unsqueeze(0).repeat(points.shape[0], 1)

    if points.shape[0] == 0:
        print("WARNING: occupancy is empty. Adding a single voxel.")
        points = torch.tensor([[0, 0, 0]])
        color = [0, 0, 0]

    # rescale to unit cube
    points = points.float() / np.array([H, W, D])[None, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.float().cpu().numpy())
    pcd.estimate_normals()
    pcd.paint_uniform_color(color)
    o3d.io.write_point_cloud(filename, pcd)


def visualize_grid2d(prefix, sample, save_locally=False, step=None):
    """Visualize the 2D image patches."""

    n_samples = sample.shape[0]

    # denormalize samples
    sample = sample * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    sample = sample.clamp(0, 1)

    if sample.shape[1] == 4:
        depth_maps = sample[:, 3, :, :].cpu().numpy()
        rgb = sample[:, :3]
    elif sample.shape[1] == 3:
        depth_maps = None
        rgb = sample
    elif sample.shape[1] == 1:
        depth_maps = sample[:, 0, :, :].cpu().numpy()
        rgb = None

    if depth_maps is not None:

        # normalize depth maps [0, 255]
        # depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())
        # depth_maps = np.clip(depth_maps, 0, 1)

        # Get the color map by name:
        cm = plt.get_cmap("plasma")

        # Apply the colormap like a function to any array:
        colored_image = np.stack([cm(d) for d in depth_maps])

        # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
        # But we want to convert to RGB in uint8 and save it:
        depth_maps = (colored_image[:, :, :, :3] * 255).astype(np.uint8)

        depth_maps = [torch.from_numpy(d).permute(2, 1, 0) for d in depth_maps]
        grid = torchvision.utils.make_grid(depth_maps, nrow=math.ceil(n_samples**0.5), padding=0)
        image = grid.permute(1, 2, 0).cpu().numpy()
        if save_locally:
            plt.imshow(image)
            plt.savefig(f"{prefix}_depth.png")
        wandb_image = wandb.Image(image)
        wandb_name = os.path.basename(prefix)
        wandb.log({os.path.join("Images/depth", wandb_name + "_depth"): wandb_image}, step=step)

    if rgb is not None:

        # log sampled images
        grid = torchvision.utils.make_grid(rgb, nrow=math.ceil(n_samples**0.5), padding=0)
        image = grid.permute(1, 2, 0).cpu().numpy()
        if save_locally:
            plt.imshow(image)
            plt.savefig(f"{prefix}_sampled_images.png")
        wandb_image = wandb.Image(image)
        wandb_name = os.path.basename(prefix)
        wandb.log({os.path.join("Images/color", wandb_name): wandb_image}, step=step)


trans_t = lambda t: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

rot_phi = lambda phi: np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

rot_theta = lambda th: np.array(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)

rot_psi = lambda psi: np.array(
    [
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


def pose_spherical(theta, psi, radius):
    c2w = trans_t(radius)
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = rot_psi(psi / 180.0 * np.pi) @ c2w
    return c2w


def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape) + 0.5
    return np.stack([rays_o, rays_d], 0)


def make_normals(rays, depth_map):
    rays_o, rays_d = rays.chunk(2, dim=-1)
    pts = rays_o + rays_d * depth_map[..., None]

    dx = pts - torch.roll(pts, -1, dims=1)
    dy = pts - torch.roll(pts, -1, dims=2)
    normal_map = torch.cross(dx, dy)
    norm = torch.norm(normal_map, dim=-1, keepdim=True)
    norm[norm < 1e-5] = 1e-5
    normal_map = normal_map / norm
    return normal_map


def render_rays(cube, rays, near, far, N_samples, clip=True, th = 0.5):
    rays_o, rays_d = rays.chunk(2, dim=-1)
    bs, H, W, _ = rays.shape

    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples).to(cube.device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts = pts.reshape(bs, 1, 1, -1, 3)

    # nearest neighbor interpolation
    alpha = F.grid_sample(cube, pts * 2 - 1, mode="bilinear", padding_mode="zeros", align_corners=False)

    if clip:
        # this is where the problem is....
        mask = torch.logical_or(torch.any(pts <= 0, -1), torch.any(pts >= 1, -1)).unsqueeze(1)

        alpha = torch.where(mask, 0.0, alpha)

    alpha = torch.where(alpha > th, 1.0, 0)
    alpha = alpha.reshape([bs, H, W, N_samples])

    trans = 1.0 - alpha + 1e-10

    trans = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], -1)
    weights = alpha * torch.cumprod(trans, -1)

    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return depth_map, acc_map


def get_3Dimage_fast(x, num_views=4, format=True, th = 0.5):
    """Will return a grid of 3D images.
    Args:
        x: the input and output of the model. (bs, 1, res, res, res)
    """

    assert len(x.shape) == 5, "x must be (bs, 1, res, res, res)"
    assert x.shape[1] == 1, "x must be (bs, 1, res, res, res)"
    assert x.shape[2] == x.shape[3] == x.shape[4], "x must be (bs, 1, res, res, res)"
    bs = x.shape[0]

    # add cube bounding box
    x[:, :, :, 0, 0] = 1
    x[:, :, :, 0, -1] = 1
    x[:, :, :, -1, 0] = 1
    x[:, :, :, -1, -1] = 1

    x[:, :, 0, :, 0] = 1
    x[:, :, 0, :, -1] = 1
    x[:, :, -1, :, 0] = 1
    x[:, :, -1, :, -1] = 1

    x[:, :, 0, 0, :] = 1
    x[:, :, 0, -1, :] = 1
    x[:, :, -1, 0, :] = 1
    x[:, :, -1, -1, :] = 1

    R = 1.5
    H = W = 256
    focal = H * 0.9

    depth_maps = []
    normal_maps = []
    for i in range(0, 360, 360 // num_views):
        c2w = pose_spherical(i + 45, -90, R)
        rays = get_rays(H, W, focal, c2w)
        rays = torch.from_numpy(rays).float().to(x.device)
        rays = rays.permute(1, 2, 0, 3).reshape(H, W, -1)
        rays = rays.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        depth_map, acc_map = render_rays(x, rays, 0, 4, 512, clip=True, th=th)
        normal_map = make_normals(rays, depth_map) * 0.5 + 0.5

        depth_maps.append(depth_map.cpu().numpy())
        normal_maps.append(normal_map.cpu().numpy())

    depth_maps = np.concatenate(depth_maps, 0)
    # normalize depth maps [0, 255]
    depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min())

    # Get the color map by name:
    cm = plt.get_cmap("plasma")

    # Apply the colormap like a function to any array:
    colored_image = np.stack([cm(d) for d in depth_maps])

    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it:
    depth_maps = (colored_image[:, :, :, :3] * 255).astype(np.uint8)

    normal_maps = np.concatenate(normal_maps, 0)

    if format:
        rc = int(math.sqrt(depth_maps.shape[0]))
        image_grid_depth = get_image_grid(depth_maps, rows=rc, cols=rc)
        image_grid_normal = get_image_grid(normal_maps, rows=rc, cols=rc)

        return image_grid_depth, image_grid_normal
    else:
        return depth_maps, normal_maps


def get_3Dimage(x, tmp_filename, resolution=256 * 2, render=True, num_views=4):
    """Will return a grid of 3D images.
    Args:
        x: the input and output of the model. (bs, 1, res, res, res)
        tmp_filename: the filename to save the mesh to. e.g., an .obj file
    """

    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(tmp_filename), exist_ok=True)

    assert len(x.shape) == 5, "x must be (bs, 1, res, res, res)"
    assert x.shape[1] == 1, "x must be (bs, 1, res, res, res)"
    assert x.shape[2] == x.shape[3] == x.shape[4], "x must be (bs, 1, res, res, res)"
    bs = x.shape[0]
    images = []
    # fixed list of colors
    colors = [(0.5, 0.5, 0.5, 1.0), (0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
    for i in range(bs):
        occupancy = x[i, 0] > 0.0  # (res, res, res)
        # draw the 12 lines around the cube
        occupancy[0, 0, :] = 1.0
        occupancy[0, -1, :] = 1.0
        occupancy[-1, 0, :] = 1.0
        occupancy[-1, -1, :] = 1.0
        occupancy[0, :, 0] = 1.0
        occupancy[0, :, -1] = 1.0
        occupancy[-1, :, 0] = 1.0
        occupancy[-1, :, -1] = 1.0
        occupancy[:, 0, 0] = 1.0
        occupancy[:, 0, -1] = 1.0
        occupancy[:, -1, 0] = 1.0
        occupancy[:, -1, -1] = 1.0
        crop = Crop(occupancy=occupancy)

        write_crop_to_mesh(crop, tmp_filename)

        if render:
            color = colors[i % len(colors)]
            im = render_mesh_from_multiple_views(tmp_filename, resolution=resolution, color=color, num_views=num_views)
            images.append(im)

    if render:
        rc = int(math.sqrt(bs))
        image_grid = get_image_grid(images, rows=rc, cols=rc)
        return image_grid


def visualize_grid3d(prefix, sample, working_dir, save_locally=False, step=None, num_views=4):
    """Visualize the 3D voxel cubes."""

    depth_image_grid, normal_image_grid = get_3Dimage_fast(sample, num_views=num_views)

    if save_locally:
        media.write_image(f"{prefix}_sampled_depth_images.png", depth_image_grid)
        media.write_image(f"{prefix}_sampled_normal_images.png", normal_image_grid)
    wandb_image_depth = wandb.Image(depth_image_grid, caption=None)
    wandb_image_normal = wandb.Image(normal_image_grid, caption=None)
    wandb_name = os.path.basename(prefix)
    wandb.log({os.path.join("Images/depth", wandb_name): wandb_image_depth}, step=step)
    wandb.log({os.path.join("Images/normal", wandb_name): wandb_image_normal}, step=step)


def visualize_grid3d_slices(prefix, sample, save_locally=False, step=None, slices=[8, 16, 24]):
    bs = sample.shape[0]
    slices = torch.cat([sample[:, :, :, s, :] for s in [8, 16, 24]])
    im = torchvision.utils.make_grid(slices, nrow=len(slices), ncol=bs, padding=1, normalize=True, scale_each=True)
    im = im.permute(1, 2, 0).cpu().numpy()
    if save_locally:
        media.write_image(f"{prefix}_sampled_slices.png", im)
    wandb_image = wandb.Image(im, caption=None)
    wandb_name = os.path.basename(prefix)
    wandb.log({os.path.join("Images/slices", wandb_name): wandb_image}, step=step)
