"""
Code to visualize 3D crops.
"""

import os

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
import trimesh

from nerfbusters.cubes.datasets3D import Crop
from nerfbusters.cubes.render import render_scenemesh


def get_crop_as_scatter_fig(crop: Crop, resolution: int):
    """Return a scatter plot of the crop."""
    resolution = crop.origins.shape[-2]
    sizes = {
        8: 10,
        16: 8,
        32: 6,
        64: 4,
    }
    x, y, z = [i.cpu().numpy() for i in crop.origins.view(-1, 3).unbind(-1)]
    color = (crop.rgb.view(-1, 3).cpu().numpy() * 255).astype(np.uint8)
    mask = crop.density.view(-1).cpu().numpy() > 0.5
    scene_box_fig = go.Scatter3d(
        name="scene",
        x=x[mask],
        y=y[mask],
        z=z[mask],
        mode="markers",
        marker=dict(
            size=sizes[resolution],
            color=color[mask],
            opacity=1.0,
        ),
    )
    return scene_box_fig


def get_bbox_as_scatter_fig(minx, maxx, miny, maxy, minz, maxz, color="red"):
    """Return a scatter plot of the crop."""
    x = [minx, maxx, maxx, minx, minx, maxx, maxx, minx]
    y = [miny, miny, maxy, maxy, miny, miny, maxy, maxy]
    z = [minz, minz, minz, minz, maxz, maxz, maxz, maxz]
    bbox_fig = go.Scatter3d(
        name="bbox",
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=10,
            color=color,
            opacity=1.0,
        ),
    )
    return bbox_fig


def write_crop_to_mesh(crop: Crop, mesh_filename: str, cube_scale=0.5, save_box=False):
    """Save the crop to a mesh file.
    Code is referenced from https://towardsdatascience.com/how-to-voxelize-meshes-and-point-clouds-in-python-ca94d403f81d.
        - Draws a cube for each occupied point in the Crop.
        - Draws a box around the crop.
    Args:
        crop: The crop to visualize.
        mesh_filename: The filename to save the mesh to. The filename should end with .obj.
        cube_scale: The size of the cube to draw for each occupied point. 0 to 1.
    """

    assert crop.occupancy is not None, "Crop must have occupancy to be visualized."

    if crop.origins is None:
        dim = crop.occupancy.shape[-1]
        x = torch.linspace(0, 1, dim)
        crop.origins = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1)

    # save bounding box of the crop
    if save_box:
        crop_vertices = torch.stack(
            [
                crop.origins[0, 0, 0],
                crop.origins[-1, 0, 0],
                crop.origins[-1, -1, 0],
                crop.origins[0, -1, 0],
                crop.origins[0, 0, -1],
                crop.origins[-1, 0, -1],
                crop.origins[-1, -1, -1],
                crop.origins[0, -1, -1],
            ]
        )
        crop_faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 1, 5],
                [0, 5, 4],
                [3, 6, 2],
                [3, 7, 6],
                [1, 2, 6],
                [1, 6, 5],
                [0, 7, 3],
                [0, 4, 7],
            ]
        )
        crop_mesh = trimesh.Trimesh(vertices=crop_vertices, faces=crop_faces)
        first, second = mesh_filename.split(".")
        _ = crop_mesh.export(os.path.join(first + "_box." + second))

    # use the not filled voxels
    origins = crop.origins[crop.occupancy == True]  # (N, 3)

    # 2. Draw a cube for each occupied point in the Crop.
    vox_mesh = o3d.geometry.TriangleMesh()
    # compute distance between two adjacent points
    cube_size = torch.linalg.norm(crop.origins[1, 0, 0] - crop.origins[0, 0, 0]).item()
    for origin in origins:
        cube = o3d.geometry.TriangleMesh.create_box(width=cube_scale, height=cube_scale, depth=cube_scale)
        color = np.array((0.5, 0.5, 0.5))
        cube.paint_uniform_color(color)
        # scale the box using the size of the voxel
        cube.scale(cube_size, center=[0, 0, 0])
        # translate the cube to be in the correct location
        cube.translate([origin[0], origin[1], origin[2]], relative=False)
        # add the box to the TriangleMesh object
        vox_mesh += cube

    o3d.io.write_triangle_mesh(mesh_filename, vox_mesh)


def get_image_grid(images, rows=None, cols=None):
    """Returns a grid of images.
    Assumes images are same height and same width.
    """

    def get_image(images, idx):
        # returns white if out of bounds
        if idx < len(images):
            return images[idx]
        else:
            return np.ones_like(images[0]) * 255

    im_rows = []
    idx = 0
    for i in range(rows):
        im_row = []
        for j in range(cols):
            im_row.append(get_image(images, idx))
            idx += 1
        im_rows.append(np.hstack(im_row))
    im = np.vstack(im_rows)
    return im


import torch


def normalize(x):
    """Returns a normalized vector."""
    return x / torch.linalg.norm(x)


def viewmatrix(lookat, up, pos):
    """Returns a camera transformation matrix.

    Args:
        lookat: The direction the camera is looking.
        up: The upward direction of the camera.
        pos: The position of the camera.

    Returns:
        A camera transformation matrix.
    """
    vec2 = -normalize(lookat)
    vec1_avg = normalize(up)
    vec0 = normalize(torch.cross(vec1_avg, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_mesh_from_multiple_views(mesh_filename, resolution=1000, color=(0.5, 0.5, 0.5, 1.0), num_views=4):
    """Renders a mesh from multiple views and returns a tiled image."""

    mesh = trimesh.load(mesh_filename, process=False)
    # TODO: add vertices to the min and max locations for the cube crop
    # get the min and max of the mesh vertices
    min = np.min(mesh.vertices, axis=0)
    max = np.max(mesh.vertices, axis=0)
    crop_center = torch.tensor((min + max) / 2).float()
    crop_diagonal = np.linalg.norm(max - min)

    # for cameras..
    # the z axis points away from the view direction and the x and y axes point to the right and up in the image plane
    import pyrender

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)

    # the 4 positions we want to render the cube from
    # all side views from the corners
    minx = min[0]
    maxx = max[0]
    miny = min[1]
    maxy = max[1]
    minz = min[2]
    maxz = max[2]
    lenz = maxz - minz
    positions = [
        [minx, miny, minz + lenz / 2.0],
        [minx, maxy, minz + lenz / 2.0],
        [maxx, maxy, minz + lenz / 2.0],
        [maxx, miny, minz + lenz / 2.0],
    ]
    positions = positions[:num_views]
    images = []
    for idx in range(len(positions)):
        # up direction of the scene
        # Z is up
        up = torch.tensor([0, 0, 1]).float()
        pos = torch.tensor(positions[idx]).float()
        direction = normalize(pos - crop_center)
        pos = pos + direction * crop_diagonal / 2.0

        pose = viewmatrix(crop_center - pos, up, pos)
        # make the pose matrix 4x4 homogeneous
        pose = torch.cat([pose, torch.tensor([[0, 0, 0, 1]]).float()], dim=0)

        W, H = resolution // 2, resolution // 2
        image, _, _ = render_scenemesh(mesh, pose.numpy(), camera, W, H, color=color)
        images.append(image)

    # tile the image
    image_grid = get_image_grid(images, rows=2, cols=2)
    return image_grid
