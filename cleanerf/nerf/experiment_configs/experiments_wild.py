from cleanerf.nerf_configs.utils import Argument
from cleanerf.nerf_configs.data import dataset_lists
import copy

# if local folder...
# output_folder = "projects/magic_eraser/outputs"

# if shared folder...
output_folder = "/shared/ethanweber/magic-eraser/nerf-outputs"

dataset_lists = [
    Argument(
        name="kitchen",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data-wild/kitchen --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints-wild/kitchen-baseline.ckpt --output-dir {output_folder}/garbage",
    )
]

arguments_list_of_lists = []
arguments_list_of_lists.append(dataset_lists)

experiments_list = [
    Argument(
        name="nerfacto",
        arg_string="",
    ),
    Argument(
        name="nerfacto-frustum",
        arg_string="--pipeline.use_frustum_loss True",
    ),
    Argument(
        name="nerfacto-frustum-cube",
        arg_string="--pipeline.use_frustum_loss True --pipeline.use_singlestep_cube_loss True",
    ),
]
arguments_list_of_lists.append(experiments_list)

# The data list needs to come last because it has subparser arguments.
visibility_list = [
    Argument(
        name="wild-split",
        arg_string="nerfstudio-data --eval-mode train-split-percentage --train-split-percentage 0.05",
    )
]
arguments_list_of_lists.append(visibility_list)
