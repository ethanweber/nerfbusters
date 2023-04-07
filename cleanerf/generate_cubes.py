"""Script to generate 3D crops.
"""

import gzip
import json
import os
import random
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from tqdm import tqdm
from tyro.conf import FlagConversionOff
from p_tqdm import p_map
from multiprocessing import Pool

from cleanerf.cubes.datasets3D import Crop, NerfstudioDataset, SyntheticMeshDataset
from cleanerf.cubes.visualize3D import write_crop_to_mesh

# objaverse info
OBJAVERSE_DIR = "/home/ethanweber/.objaverse/hf-objaverse-v1/"
OBJAVERSE_UIDS = []
with open("objaverse/uids.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        OBJAVERSE_UIDS.append(line.strip())


def get_obj_filenames_from_shapenet(
    shapenet_dir, shapenet_classes, shapenet_num_per_class: Optional[int] = None
) -> list:
    # get all the obj files of shapenet from the directory
    train_labels = os.path.join(shapenet_dir, "train_labels.txt")
    # read lines from the file
    with open(train_labels, "r") as f:
        lines = f.readlines()

    class_to_obj_filenames = {}
    for line in lines:
        split = line.split(" ")
        prefix = split[0]
        class_ = " ".join([x.strip() for x in split[1:]])
        if class_ in shapenet_classes:
            obj_filename = os.path.join(shapenet_dir, prefix, "models/model_normalized.obj")
            class_to_obj_filenames.setdefault(class_, []).append(obj_filename)

    obj_filenames = []
    if shapenet_num_per_class is not None:
        for class_ in shapenet_classes:
            # sort the list
            class_to_obj_filenames[class_] = sorted(class_to_obj_filenames[class_])
            obj_filenames.extend(class_to_obj_filenames[class_][:shapenet_num_per_class])
    else:
        for class_ in shapenet_classes:
            # sort the list
            class_to_obj_filenames[class_] = sorted(class_to_obj_filenames[class_])
            obj_filenames.extend(class_to_obj_filenames[class_])

    return obj_filenames


def get_mesh_filenames_from_objaverse() -> list:
    # get all the mesh files of objaverse, specified by the UID list
    mesh_filenames = []
    object_path_filename = os.path.join(OBJAVERSE_DIR, "object-paths.json.gz")
    with gzip.open(object_path_filename, "rb") as f:
        uid_to_glb_path = f.read()
        uid_to_glb_path = json.loads(uid_to_glb_path.decode("utf-8"))
    for uid in OBJAVERSE_UIDS:
        mesh_filename = os.path.join(OBJAVERSE_DIR, uid_to_glb_path[uid])
        mesh_filenames.append(mesh_filename)
    return mesh_filenames


def process_mesh(
    mesh_filename,
    voxel_method,
    binvox_resolution,
    binvox_path,
    processed_mesh_filename,
    precompute,
    num_crops_per_mesh,
    crop_percent_range,
    crop_resolution,
) -> str:
    """Function to process the mesh."""

    dataset = SyntheticMeshDataset(
        mesh_filename,
        process_directory=processed_mesh_filename.split(".")[0] + "_processed_directory",
        voxel_method=voxel_method,
        binvox_path=binvox_path,
        binvox_resolution=binvox_resolution,
    )
    dataset.export_mesh(processed_mesh_filename)

    if precompute:
        assert voxel_method == "binvox", "Must use binvox to save precomputed voxels."
        voxels_surface = dataset.voxels_surface.cpu().bool().numpy()
        voxels = dataset.voxels.cpu().bool().numpy()
        # TODO(frederik): save as PNG
        voxels_surface_filename = processed_mesh_filename.split(".")[0] + "_voxels_surface.npy"
        voxels_filename = processed_mesh_filename.split(".")[0] + "_voxels.npy"
        np.save(voxels_surface_filename, voxels_surface)
        np.save(voxels_filename, voxels)
        # save the resolution
        voxels_resolution_filename = processed_mesh_filename.split(".")[0] + "_voxels_resolution.txt"
        with open(voxels_resolution_filename, "w") as f:
            f.write(str(binvox_resolution))
        print("Saved precomputed voxels.")
        return

    crops = []
    skipped_count = 0
    for _ in tqdm(range(num_crops_per_mesh)):
        crop_percent = random.uniform(crop_percent_range[0], crop_percent_range[1])
        crop = dataset.get_crop(resolution=crop_resolution, crop_percent=crop_percent)

        # assert that there is some occupancy
        occupancy_sum = crop.occupancy.sum()
        if occupancy_sum == 0:
            print("Skipping crop with no occupancy.")
            skipped_count += 1
        else:
            crops.append(crop)

    return crops


def main(
    dataset_name: Literal["shapenet", "objaverse", "obj-filenames"] = "obj-filenames",
    shapenet_dir: str = "/home/ethanweber/datasets/ShapeNetCore.v2/",
    shapenet_classes: set = set(["sofa"]),
    shapenet_num_per_class: Optional[int] = None,
    obj_filenames: Optional[str] = None,
    num_meshes: Optional[int] = None,
    num_crops_per_mesh: int = 500,
    crop_resolution: int = 32,
    crop_percent_range: Tuple[float, ...] = (0.01, 0.05),
    voxel_method: Literal["sdf", "binvox"] = "binvox",
    binvox_path: str = "bins/binvox",
    binvox_resolution: int = 1024,
    output_dir: str = "datasets/cubes",
    precompute: FlagConversionOff[bool] = False,
) -> None:
    """Generate 3D cubes from meshes.
    Args:
        obj_filenames: A list of obj filenames to use.
        num_meshes: The number of meshes to use. If None, use all meshes.
        num_crops_per_mesh: The number of crops to generate per mesh.
        output_dir: The directory to save the crop dataset.
    """

    # delete the output directory if it exists
    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    if dataset_name == "shapenet":
        obj_filenames = get_obj_filenames_from_shapenet(shapenet_dir, shapenet_classes, shapenet_num_per_class)
        mesh_filenames = obj_filenames
    elif dataset_name == "objaverse":
        mesh_filenames = get_mesh_filenames_from_objaverse()
    elif dataset_name == "obj-filenames":
        assert obj_filenames is not None, "Must provide obj_filenames"
        obj_filenames = obj_filenames.split(",")
        mesh_filenames = obj_filenames
    print(f"Using {len(mesh_filenames)} meshes.")

    # shuffle the mesh filenames
    # random.shuffle(mesh_filenames)
    if num_meshes is not None:
        mesh_filenames = mesh_filenames[: min(num_meshes, len(mesh_filenames))]

    if not precompute:
        num_crops = len(mesh_filenames) * num_crops_per_mesh
        print(
            f"Generating {num_crops} cubes from {len(mesh_filenames)} meshes. Each mesh will be cropped {num_crops_per_mesh} times."
        )
    else:
        print(f"Precomputing {len(mesh_filenames)} meshes.")

    crop_idx_to_mesh_filename = []
    crop_idx_to_processed_mesh_filename = []
    crops = []
    skipped_count = 0
    failed_meshes = 0

    processed_mesh_filenames = []
    for mesh_filename in mesh_filenames:
        if dataset_name == "shapenet":
            processed_mesh_filename = os.path.join(output_dir, "meshes", mesh_filename.replace(shapenet_dir, ""))
        elif dataset_name == "objaverse":
            processed_mesh_filename = os.path.join(output_dir, "meshes", mesh_filename.replace(OBJAVERSE_DIR, ""))
        else:
            processed_mesh_filename = os.path.join(output_dir, "meshes", mesh_filename)
        processed_mesh_filenames.append(processed_mesh_filename)

    # write all these filenamees to a file in process_dir
    filename = os.path.join(output_dir, "processed_mesh_filenames.txt")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("\n".join(processed_mesh_filenames))

    num_meshes = len(mesh_filenames)
    a = mesh_filenames
    b = [voxel_method] * num_meshes
    c = [binvox_resolution] * num_meshes
    d = [binvox_path] * num_meshes
    e = processed_mesh_filenames
    f = [precompute] * num_meshes
    g = [num_crops_per_mesh] * num_meshes
    h = [crop_percent_range] * num_meshes
    k = [crop_resolution] * num_meshes

    # TODO(frederik): use multiprocessing

    # use p_map
    # crops = p_map(process_mesh, a, b, c, d, e, f, g, h, k)

    args = []
    for i in range(len(mesh_filenames)):
        args.append((a[i], b[i], c[i], d[i], e[i], f[i], g[i], h[i], k[i]))

    # use multiprocessing
    # with Pool() as pool:
    #     crops = pool.starmap(process_mesh, args)

    # sequential
    for arg in args:
        process_mesh(*arg)

    print(f"{failed_meshes} failed. Skipped {skipped_count} crops. Generated {len(crops)} crops.")

    os.makedirs(output_dir, exist_ok=True)

    # save the cubes
    output_filename = os.path.join(output_dir, "crops.pt")
    occ = []
    for crop in crops:
        occ.append(crop.occupancy)
    occ = torch.stack(occ)
    occ = occ.to(torch.bool)
    torch.save(occ, output_filename)

    # save the cubes
    output_filename = os.path.join(output_dir, "scales.pt")
    scales = []
    for crop in crops:
        scales.append(torch.tensor(crop.scale))
    scales = torch.stack(scales)
    torch.save(scales, output_filename)

    # save the origins
    output_filename = os.path.join(output_dir, "origins.pt")
    origins = []
    for crop in crops:
        origins.append(crop.origins)
    origins = torch.stack(origins)
    torch.save(origins, output_filename)

    # save which mesh each cube came from
    with open(os.path.join(output_dir, "mesh_filenames.txt"), "w") as f:
        for fn in crop_idx_to_mesh_filename:
            f.write(fn + "\n")
    # save which mesh each cube came from
    with open(os.path.join(output_dir, "processed_mesh_filenames.txt"), "w") as f:
        for fn in crop_idx_to_processed_mesh_filename:
            f.write(fn + "\n")


if __name__ == "__main__":
    tyro.cli(main)
