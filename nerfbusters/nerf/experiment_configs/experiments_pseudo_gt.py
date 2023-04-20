import copy

from nerfbusters.nerf.experiment_configs.experiments_baseline import dataset_lists
from nerfbusters.nerf.experiment_configs.utils import Argument

# Here we replace the string .baseline.ckpt with .pseudo-gt.ckpt
# to load the pseudo-gt checkpoint instead of the baseline checkpoint.
pseudo_gt_datasets_lists = []
for argument in copy.deepcopy(dataset_lists):
    argument.arg_string = argument.arg_string.replace("-baseline.ckpt", "-pseudo-gt.ckpt")
    pseudo_gt_datasets_lists.append(argument)

arguments_list_of_lists = []
arguments_list_of_lists.append(pseudo_gt_datasets_lists)

experiments_list = [
    Argument(
        name="nerfacto",
        arg_string="--pipeline.use_visibility_loss False --pipeline.use_singlestep_cube_loss False",
    )
]
arguments_list_of_lists.append(experiments_list)

# The data list needs to come last because it has subparser arguments.
pseudo_gt_list = [
    Argument(
        name="pseudo-gt",
        arg_string="nerfstudio-data --train-frame-indices 0 1",
    )
]
arguments_list_of_lists.append(pseudo_gt_list)
