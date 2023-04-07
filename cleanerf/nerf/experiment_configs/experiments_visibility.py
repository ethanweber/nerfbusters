from cleanerf.nerf_configs.utils import Argument
from cleanerf.nerf_configs.data import dataset_lists
import copy

# Here we replace the string .baseline.ckpt with .visibility.ckpt
# to load the visibility checkpoint instead of the baseline checkpoint.
visibility_datasets_lists = []
for argument in copy.deepcopy(dataset_lists):
    argument.arg_string = argument.arg_string.replace("baseline.ckpt", "visibility.ckpt")
    visibility_datasets_lists.append(argument)

arguments_list_of_lists = []
arguments_list_of_lists.append(visibility_datasets_lists)

experiments_list = [
    Argument(
        name="nerfacto",
        arg_string="",
    )
]
arguments_list_of_lists.append(experiments_list)

# The data list needs to come last because it has subparser arguments.
visibility_list = [
    Argument(
        name="visibility-split",
        arg_string="nerfstudio-data --train-frame-indices 0 1",
    )
]
arguments_list_of_lists.append(visibility_list)
