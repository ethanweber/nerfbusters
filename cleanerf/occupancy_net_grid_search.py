import os
import sys
import time

import magic_eraser.utils.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from diffusers import DDIMScheduler, DDPMScheduler
from dotmap import DotMap
from cleanerf.cubes.datasets3D import Crop, SyntheticMeshDataset
from cleanerf.lightning.magic_eraser_2d import MagicEraser2D
from cleanerf.utils import metrics
from cleanerf.utils.visualizations import (
    get_3Dimage_fast,
    save_voxel_as_point_cloud,
)
from tqdm.notebook import tqdm as tqdm

from nerfstudio.field_components.encodings import NeRFEncoding, RFFEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.pipelines.magic_eraser_pipeline import sample_cubes


def gt_fn(queries, voxels, interp=True):

    if interp:
        # from [0, 1] to [-1, 1]
        queries = queries * 2 - 1  # [N, 3]
        queries = queries @ torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).float().to(queries.device)
        queries = queries.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, N, 3]
        voxels = voxels.unsqueeze(0).unsqueeze(0)  # [N, C, D, H, W]
        occupancy = F.grid_sample(voxels, queries, mode="bilinear", align_corners=True).squeeze()
    else:
        resolution = voxels.shape[-1] - 1
        # from [0, 1] to [0, resolution]
        queries = (queries * resolution).long()
        # get occupancy values
        occupancy = voxels[queries[:, 0], queries[:, 1], queries[:, 2]]

    return occupancy.float()


def render_rays(net, pos_enc, rays, near, far, N_samples, N_samples_2, clip=True, device="cuda:0"):
    rays_o, rays_d = rays[0], rays[1]

    th = 0.5

    # Compute 3D query points
    z_vals = np.linspace(near, far, N_samples)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts = pts.reshape(-1, 3)
    pts = torch.from_numpy(pts).float()

    # Run network
    alpha = eval(net, pos_enc, pts, batch_size=1000, device=device)
    alpha = alpha.numpy()
    pts = pts.numpy()

    if clip:
        mask = np.logical_or(np.any(pts <= 0, -1), np.any(pts >= 1, -1))
        alpha = np.where(mask, 0.0, alpha)

    alpha = np.where(alpha > th, 1.0, 0)

    alpha = alpha.reshape(list(rays_d.shape[:-1]) + [N_samples])

    trans = 1.0 - alpha + 1e-10
    trans = np.concatenate([np.ones_like(trans[..., :1]), trans[..., :-1]], -1)
    weights = alpha * np.cumprod(trans, -1)

    depth_map = np.sum(weights * z_vals, -1)
    acc_map = np.sum(weights, -1)

    # Second pass to refine isosurface
    z_vals = np.linspace(-1.0, 1.0, N_samples_2) * 0.01 + depth_map[..., None]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts = pts.reshape(-1, 3)
    pts = torch.from_numpy(pts).float()

    # Run network
    alpha = eval(net, pos_enc, pts, batch_size=1000, device=device)
    alpha = alpha.numpy()
    pts = pts.numpy()

    if clip:
        mask = np.logical_or(np.any(pts <= 0, -1), np.any(pts >= 1, -1))
        alpha = np.where(mask, 0.0, alpha)

    alpha = alpha.reshape(list(rays_d.shape[:-1]) + [N_samples])
    alpha = np.where(alpha > th, 1.0, 0)

    trans = 1.0 - alpha + 1e-10
    trans = np.concatenate([np.ones_like(trans[..., :1]), trans[..., :-1]], -1)
    weights = alpha * np.cumprod(trans, -1)

    depth_map_fine = np.sum(weights * z_vals, -1)
    acc_map_fine = np.sum(weights, -1)

    return depth_map, acc_map, depth_map_fine, acc_map_fine


def eval(model, pos_enc, pts, batch_size, device):

    model.eval()

    num_pts = pts.shape[0]
    num_batches = num_pts // batch_size + 1
    occupancy = torch.zeros(num_pts)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_pts)
        if start == end:
            continue

        with torch.no_grad():

            pts_batch = pts[start:end].to(device)

            pts_batch = pts_batch.contiguous()
            inputs_enc = pos_enc(pts_batch)
            occupancy[start:end] = model(inputs_enc).squeeze(-1).cpu()

    return occupancy


# rendering stuff
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


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    # c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def get_rays(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape) + 0.5
    return np.stack([rays_o, rays_d], 0)


def make_normals(rays, depth_map):
    rays_o, rays_d = rays
    pts = rays_o + rays_d * depth_map[..., None]
    dx = pts - np.roll(pts, -1, axis=0)
    dy = pts - np.roll(pts, -1, axis=1)
    normal_map = np.cross(dx, dy)
    normal_map = normal_map / np.maximum(np.linalg.norm(normal_map, axis=-1, keepdims=True), 1e-5)
    return normal_map


def sampler(sampling_grid, sampling_reso, n_pts, avoid_top_percentile=100):

    probs = sampling_grid.view(-1) / sampling_grid.view(-1).sum()
    if avoid_top_percentile < 100:
        probs[probs > np.percentile(probs.cpu().detach().numpy(), avoid_top_percentile)] = 0
        probs = probs / probs.sum()

    dist = torch.distributions.categorical.Categorical(probs)
    sample = dist.sample((n_pts,))

    h = sample // sampling_reso**2
    d = sample % sampling_reso
    w = (sample // sampling_reso) % sampling_reso

    idx = torch.stack([h, w, d], dim=1).float()

    return (idx + torch.rand_like(idx)) / sampling_reso


def get_data(device, mesh="dragon"):

    # load scene with a specific resolution
    resolution = 512  # resolution of the gt occupancy grid
    test_reso = 64  # resolution of the test occupancy grid
    mask_label_reso = 64  # resolution of the mask occupancy grid
    masked_percentage = 0.9  # percentage of the mask to be masked (0.0 = no mask. 1.0 = full mask)
    noisy_percentage = 0.0  # percentage of the mask to be noisy (0.0 = no noise. 1.0 = full noise)

    # Put your mesh files here
    mesh_files = {
        "dragon": "../meshes/dragon.ply",
        "armadillo": "../meshes/Armadillo.ply",
        "bunny": "../meshes/bun_zipper.ply",
        "buddha": "../meshes/happy_vrip.ply",
        "lucy": "../meshes/lucy.ply",
    }

    # load the scene
    data = SyntheticMeshDataset(
        mesh_filename=mesh_files[mesh],
        voxel_method="binvox",
        process_directory="../temp",
        binvox_path="../bins/binvox",
        device=device,
        binvox_resolution=resolution,
    )

    occupancy = data.voxels.float()

    # set random seeed
    torch.manual_seed(0)
    mask_labels = torch.ones(mask_label_reso, mask_label_reso, mask_label_reso, device=device)
    mask_labels[torch.rand_like(mask_labels) < masked_percentage] = 0

    noise = torch.randint_like(mask_labels, 0, 2).float()  # [0, 1]
    where = torch.rand_like(mask_labels) < noisy_percentage
    noise = torch.nn.functional.interpolate(noise[None, None, :, :, :], size=resolution, mode="nearest")[0, 0]
    where = torch.nn.functional.interpolate(where[None, None, :, :, :].float(), size=resolution, mode="nearest")[0, 0]
    occupancy[where.bool()] = noise[where.bool()]

    scene = Crop(occupancy=occupancy.to(device))

    # save gt scene pcd
    if False:
        save_voxel_as_point_cloud(scene.occupancy[None, :, :, :] * 2 - 1, os.path.join(logdir, "gt.ply"))

    # define validation and testing points
    validation_pts = torch.rand((test_reso, 3)).float().to(device)
    testing_pts = torch.linspace(0, 1, test_reso)
    testing_pts = (
        torch.stack(torch.meshgrid([testing_pts, testing_pts, testing_pts], indexing="ij"))
        .reshape(3, -1)
        .permute(1, 0)
        .to(device)
    )

    # sanity check to see if the gt is correct
    if False:

        gt = gt_fn(testing_pts, scene.occupancy, interp=False)
        gt = gt.reshape(test_reso, test_reso, test_reso)
        save_voxel_as_point_cloud(gt[None, :, :, :] * 2 - 1, os.path.join(logdir, "gt_index.ply"))

        gt = gt_fn(testing_pts, scene.occupancy, interp=True)
        gt = gt.reshape(test_reso, test_reso, test_reso)
        save_voxel_as_point_cloud(gt[None, :, :, :] * 2 - 1, os.path.join(logdir, "gt_interp.ply"))

        mask_gt = gt_fn(testing_pts, mask_labels, interp=False)
        mask_gt = mask_gt.reshape(test_reso, test_reso, test_reso)
        save_voxel_as_point_cloud(mask_gt[None, :, :, :] * 2 - 1, os.path.join(logdir, "mask_gt_index.ply"))

        mask_gt = gt_fn(testing_pts, mask_labels, interp=True)
        mask_gt = mask_gt.reshape(test_reso, test_reso, test_reso)
        save_voxel_as_point_cloud(mask_gt[None, :, :, :] * 2 - 1, os.path.join(logdir, "mask_gt_interp.ply"))

    return scene, mask_labels, validation_pts, testing_pts, test_reso


def load_diffusion_model(device):

    # load diffusion model
    diffusion_model_ckpt = "/home/warburg/repo/3D-magic-eraser/projects/magic_eraser/lightning_logs/cubes_shapenet/ddpm/2023-02-26_182437-empty-cubes-60/checkpoints/last.ckpt"
    config = yaml.load(open("../config/shapenet.yaml", "r"), Loader=yaml.Loader)

    config = DotMap(config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    diffusion_model = MagicEraser2D(config, noise_scheduler)
    ckpt = torch.load(diffusion_model_ckpt, map_location=torch.device("cpu"))
    diffusion_model.load_state_dict(ckpt["state_dict"])
    diffusion_model.eval()
    diffusion_model.to(device)

    return diffusion_model


def get_occupancy_net(device, checkpoint_path=None):

    # create occupancy network
    torch.manual_seed(0)
    pos_enc = RFFEncoding(in_dim=3, num_frequencies=16, scale=4.0, include_input=True)
    pos_enc = pos_enc.to(device)
    _, input_size = pos_enc(torch.zeros(1, 3).to(device)).shape
    net = MLP(input_size, 8, 256, 1)

    if checkpoint_path is not None:
        print("==> loading checkpoint '{}'".format(checkpoint_path))
        ckpt = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
        )
        net.load_state_dict(ckpt)
    net = net.to(device)

    return pos_enc, net


def main(regularizer, cube_size, grad_mult, lamda_k, mesh, max_step):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    logdir = f"occupancy_results/{mesh}/{regularizer}"

    if regularizer == "none":
        nerf_path = None
    else:
        nerf_path = logdir.replace(f"{regularizer}", "none")
        nerf_path = os.path.join(nerf_path, "model_10000.pth")
    pos_enc, net = get_occupancy_net(device, checkpoint_path=nerf_path)

    if regularizer == "sds":
        logdir = os.path.join(logdir, f"{cube_size}/{grad_mult}/{max_step}")
    elif regularizer != "none":
        logdir = os.path.join(logdir, f"{lamda_k}")

    os.makedirs(logdir, exist_ok=True)

    scene, mask_labels, validation_pts, testing_pts, test_reso = get_data(device, mesh)

    if regularizer == "sds":
        diffusion_model = load_diffusion_model(device)

    # rendeirng parameters
    H = 256
    W = H
    focal = H * 0.9
    N_samples, N_samples2 = 512, 512

    R = 0.7
    c2w = pose_spherical(45, 0.0, R)
    rays = get_rays(H, W, focal, c2w[:3, :4])

    save_pcd = False

    sampling_reso = 20
    sampling_grid = torch.zeros((sampling_reso, sampling_reso, sampling_reso), device=device)

    depth_map, acc_map, depth_map_fine, acc_map_fine = render_rays(
        net, pos_enc, rays, 0, 4.0, N_samples, N_samples2, True, device
    )
    normal_map = make_normals(rays, depth_map_fine) * 0.5 + 0.5

    plt.imsave(os.path.join(logdir, "depth_0.png"), depth_map)
    plt.imsave(os.path.join(logdir, "depth_fine_0.png"), depth_map_fine)
    plt.imsave(os.path.join(logdir, "normal_0.png"), normal_map)

    N_iters = 1000
    batch_size = 10000  # 64*64*2 * 4 # 30000
    lr = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    save_pcd_every = 1000
    save_model_every = 1000

    cube_loss_mult = grad_mult

    # save_at_steps = [10, 100, 200, 300, 400, 500]
    save_at_steps = []
    t_sample_cubes = 0
    t_query_points = 0
    t_sds_loss = 0

    # sds
    res = 32
    num_cubes = 32
    min_x, min_y, min_z = 0, 0, 0
    max_x, max_y, max_z = 1, 1, 1
    spr_min, spr_max = cube_size[0], cube_size[1]

    interp = True

    pbar = tqdm(range(N_iters + 1))
    for i in pbar:
        net.train()

        pts = torch.rand(size=[batch_size, 3]).to(device)
        gt = gt_fn(pts, scene.occupancy, interp=interp)

        # TODO: trilinear interpolation does not work for mask...
        gt_mask = gt_fn(pts, mask_labels, interp=False)

        # cast to cuda
        gt = gt.unsqueeze(1).to(device)
        pts = pts.to(device)
        gt_mask = gt_mask.unsqueeze(1).to(device)
        gt_mask[gt_mask < 0.5] = 0
        gt_mask[gt_mask >= 0.5] = 1

        inputs_enc = pos_enc(pts)
        density = net(inputs_enc)

        # loss = torch.mean(gt_mask * (density - gt).abs())
        loss = torch.mean(gt_mask * (density - gt) ** 2)
        # loss = (gt_mask * F.binary_cross_entropy(density, gt, reduce=False)).mean()

        # update occupancy grid used for sampling
        index = (pts * sampling_reso).long()
        sampling_grid[index[:, 0], index[:, 1], index[:, 2]] += torch.relu(density.clone().squeeze())

        if regularizer in ("sds", "cube_sparsity", "cube_tv"):

            with torch.no_grad():
                t = time.time()

                centers = sampler(sampling_grid, sampling_reso, num_cubes)
                cube_centers_x, cube_centers_y, cube_centers_z = centers.chunk(3, dim=1)
                cube_centers_x, cube_centers_y, cube_centers_z = (
                    cube_centers_x.squeeze(),
                    cube_centers_y.squeeze(),
                    cube_centers_z.squeeze(),
                )

                pts, _, _, scales = sample_cubes(
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
                    cube_centers_x,
                    cube_centers_y,
                    cube_centers_z,
                    device=device,
                )
                pts = pts.reshape(-1, 3)

            inputs_enc = pos_enc(pts)
            x = net(inputs_enc)
            x = x.reshape(num_cubes, 1, res, res, res)

            if regularizer == "sds":

                t_query_points += time.time() - t
                t = time.time()
                x = (2.0 * x - 1.0).reshape(num_cubes, 1, res, res, res).clamp(min=-1, max=1)
                timesteps = (torch.randint(10, max_step, (1,)).long().to(device)).item()
                # timesteps = torch.randint(50, 100, (num_cubes,)).long().to(device)
                _ = diffusion_model.sds_loss.grad_sds_unconditional(
                    x, diffusion_model.model, timesteps, scale=scales, mult=cube_loss_mult
                )
                t_sds_loss += time.time() - t

            if regularizer == "cube_tv":
                delta_x = x[..., 1:, :-1, :-1] - x[..., :-1, :-1, :-1]
                delta_y = x[..., :-1, 1:, :-1] - x[..., :-1, :-1, :-1]
                delta_z = x[..., :-1, :-1, 1:] - x[..., :-1, :-1, :-1]
                loss += lamda_k * torch.mean(delta_x.abs() + delta_y.abs() + delta_z.abs())

            if regularizer == "cube_sparsity":
                l = 0.05
                v = torch.abs(1.0 - torch.exp(-l * x))
                v = torch.mean(v)
                loss += lamda_k * v

        if regularizer == "sparsity":

            pts = torch.rand(size=[batch_size, 3]).to(device)
            inputs_enc = pos_enc(pts)
            x = net(inputs_enc)

            l = 0.05
            v = torch.abs(1.0 - torch.exp(-l * x))
            v = torch.mean(v)
            loss += lamda_k * v

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("loss: %.4f" % loss.item())
        pbar.set_postfix_str(f"loss {loss.item()}")

        if i % 100 == 0:
            print("train loss", loss.item(), "voxels", (density > 0.5).sum().item(), len(pts))
            print(
                "sds loss",
                np.round(t_sds_loss, 3),
                "t_sample_cubes",
                np.round(t_sample_cubes, 3),
                "t_query_points",
                np.round(t_query_points, 3),
            )

            gt = gt_fn(validation_pts, scene.occupancy, interp=interp)
            gt = gt.unsqueeze(1).to(device)

            inputs_enc = pos_enc(validation_pts)
            density = net(inputs_enc)

            val_loss = torch.mean((density - gt).abs())

            val_iou = torch.logical_and(density > 0.5, gt > 0.5).sum() / torch.logical_or(density > 0.5, gt > 0.5).sum()
            val_acc = ((density > 0.5) == (gt > 0.5)).sum() / len(validation_pts)

            print(
                "validation loss",
                val_loss.item(),
                "iou",
                np.round(val_iou.item(), 2),
                "acc",
                np.round(val_acc.item(), 2),
            )

            t_sample_cubes = 0
            t_query_points = 0
            t_sds_loss = 0

        if (i % save_pcd_every == 0 and i > 1) or i in save_at_steps:
            print("evaluating", i)
            voxels = eval(net, pos_enc, testing_pts, batch_size, device)
            voxels = voxels.reshape(1, test_reso, test_reso, test_reso).cpu() * 2 - 1
            if save_pcd:
                print("voxels", (voxels > 0.5).sum())
                # save voxel as point cloud
                save_voxel_as_point_cloud(voxels, os.path.join(logdir, "pcd_%d.ply" % i))

            depth_map, acc_map, depth_map_fine, acc_map_fine = render_rays(
                net, pos_enc, rays, 0, 4.0, N_samples, N_samples2, True, device
            )
            normal_map = make_normals(rays, depth_map_fine) * 0.5 + 0.5

            plt.imsave(os.path.join(logdir, "depth_%d.png" % i), depth_map)
            plt.imsave(os.path.join(logdir, "depth_fine_%d.png" % i), depth_map_fine)
            plt.imsave(os.path.join(logdir, "normal_%d.png" % i), normal_map)

            gt = gt_fn(testing_pts, scene.occupancy, interp=interp)
            gt = gt.reshape(1, test_reso, test_reso, test_reso).cpu()

            iou = metrics.voxel_iou(voxels, gt)
            acc = metrics.voxel_acc(voxels, gt)
            f1 = metrics.voxel_f1(voxels, gt)
            mse = torch.mean((voxels - gt) ** 2)
            l1 = torch.mean((voxels - gt).abs())

            # write metrics to file
            with open(os.path.join(logdir, "metrics.txt"), "a+") as f:
                f.write("%d " % i)
                f.write(",".join([f"{i}", f"{iou}", f"{acc}", f"{f1}", f"{mse}", f"{l1}"]))
                f.write("\n")

        if (i > 0 and i % save_model_every == 0) or i in save_at_steps:
            print("save model", i)
            torch.save(net.state_dict(), os.path.join(logdir, "model_%d.pth" % i))

    voxels = eval(net, pos_enc, testing_pts, batch_size, device)
    voxels = voxels.reshape(1, test_reso, test_reso, test_reso).cpu() * 2 - 1

    gt = gt_fn(testing_pts, scene.occupancy, interp=interp)
    gt = gt.reshape(1, test_reso, test_reso, test_reso).cpu()

    iou = metrics.voxel_iou(voxels, gt)
    acc = metrics.voxel_acc(voxels, gt)
    f1 = metrics.voxel_f1(voxels, gt)
    mse = torch.mean((voxels - gt) ** 2)
    l1 = torch.mean((voxels - gt).abs())

    # write metrics to file
    with open(os.path.join(logdir, "updated_metrics.txt"), "w") as f:
        f.write(",".join([f"{iou}", f"{acc}", f"{f1}", f"{mse}", f"{l1}"]))
        f.write("\n")


def recompute_metrics(scene, testing_pts, test_reso, regularizer, cube_size, grad_mult, lamda_k, mesh):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logdir = f"occupancy_results/{mesh}/{regularizer}"

    if regularizer == "none":
        logdir = logdir.replace(f"{regularizer}", "none")
        nerf_path = os.path.join(logdir, "model_10000.pth")
    else:
        if regularizer == "sds":
            logdir = os.path.join(logdir, f"{cube_size}/{grad_mult}")
        elif regularizer != "none":
            logdir = os.path.join(logdir, f"{lamda_k}")

        nerf_path = os.path.join(logdir, "model_10000.pth")

    pos_enc, net = get_occupancy_net(device, checkpoint_path=nerf_path)

    interp = True
    batch_size = 10_000

    voxels = eval(net, pos_enc, testing_pts, batch_size, device)
    voxels = voxels.reshape(1, test_reso, test_reso, test_reso).cpu() * 2 - 1

    gt = gt_fn(testing_pts, scene.occupancy, interp=interp)
    gt = gt.reshape(1, test_reso, test_reso, test_reso).cpu()

    iou = metrics.voxel_iou(voxels, gt)
    acc = metrics.voxel_acc(voxels, gt)
    f1 = metrics.voxel_f1(voxels, gt)
    mse = torch.mean((voxels - gt) ** 2)
    l1 = torch.mean((voxels - gt).abs())

    # write metrics to file
    with open(os.path.join(logdir, "updated_metrics.txt"), "w") as f:
        f.write(",".join([f"{iou}", f"{acc}", f"{f1}", f"{mse}", f"{l1}"]))
        f.write("\n")


if __name__ == "__main__":

    # (0.01, 0.05),
    cube_size = (0.01, 0.3)
    # grad_mult = 1e-7
    lamda_k = 0.01
    for mesh in ["dragon"]:  # , "armadillo"]:  # ["buddha"]:  # , "armadillo", "lucy", "bunny"]:

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        scene, mask_labels, validation_pts, testing_pts, test_reso = get_data(device, mesh)

        for grad_mult in [1e-7, 1e-6, 1e-5]:
            for regularizer in ["sds"]:  # , "none", "sparsity", "cube_sparsity", "cube_tv"]:  # , "sds",
                # for lamda_k in [0.01, 0.001, 0.0001, 0.00001]:
                for max_step in [100, 200, 300, 400]:
                    main(regularizer, cube_size, grad_mult, lamda_k, mesh, max_step)
                    # recompute_metrics(
                    #    scene, testing_pts, test_reso, regularizer, cube_size, grad_mult, lamda_k, mesh, max_step
                    # )
