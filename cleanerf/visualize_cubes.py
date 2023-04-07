"""
Visualize the cubes datasets.
"""
import os
import random
import shutil
from typing import Optional

import torch
import tyro

from cleanerf.cubes.datasets3D import Crop
from cleanerf.cubes.visualize3D import write_crop_to_mesh


def main(
    dataset_dir: str = "datasets/shapenet-cubes/", num_meshes_to_show: Optional[int] = None, num_cubes_to_show: int = 5
) -> None:
    """Visualize a cubes dataset.
    Args:
        dataset_dir: The directory of the dataset.
        num_meshes_to_show: The number of meshes to show.
        num_cubes_to_show: The number of cubes to show for each mesh.
    """
    visualize_dir = os.path.join(dataset_dir, "visualize")

    # delete visualize dir if it already exists
    if os.path.exists(visualize_dir):
        os.system(f"rm -rf {visualize_dir}")

    mesh_filenames = []
    with open(os.path.join(dataset_dir, "processed_mesh_filenames.txt"), "r") as f:
        for line in f:
            mesh_filenames.append(line.strip())

    # load the cubes
    occupancy = torch.load(os.path.join(dataset_dir, "crops.pt"))
    origins = torch.load(os.path.join(dataset_dir, "origins.pt"))

    mesh_filenames_set = set(mesh_filenames)
    if num_meshes_to_show is None:
        num_meshes_to_show = len(mesh_filenames_set)
    selected_mesh_filenames = random.sample(mesh_filenames_set, min(num_meshes_to_show, len(mesh_filenames_set)))
    for mesh_filename in selected_mesh_filenames:
        print("Visualizing mesh... ", mesh_filename)

        output_folder = os.path.join(visualize_dir, mesh_filename.replace(dataset_dir, "").split(".")[0])
        # delete the folder if it already exists
        if os.path.exists(output_folder):
            os.system(f"rm -rf {output_folder}")

        # copy to output folder
        output_mesh_filename = os.path.join(output_folder, os.path.basename(mesh_filename))
        os.makedirs(output_mesh_filename, exist_ok=True)
        shutil.copy(mesh_filename, output_mesh_filename)

        # create list of cubes that correspond to this mesh
        mesh_cubes = []
        for i in range(len(mesh_filenames)):
            if mesh_filenames[i] == mesh_filename:
                cube = Crop(origins=origins[i], occupancy=occupancy[i])
                mesh_cubes.append(cube)

        print(f"Found {len(mesh_cubes)} cubes for this mesh.")
        print(f"Showing {num_cubes_to_show} cubes.")

        cubes_to_show = random.sample(mesh_cubes, min(num_cubes_to_show, len(mesh_cubes)))
        import time

        for idx, cube in enumerate(cubes_to_show):
            s = time.time()
            write_crop_to_mesh(cube, os.path.join(output_folder, f"cube_{idx}.obj"))
            e = time.time()
            print(f"Saved cube {idx} in {e - s} seconds")


if __name__ == "__main__":
    tyro.cli(main)
