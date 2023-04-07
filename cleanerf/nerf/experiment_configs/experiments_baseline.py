from cleanerf.nerf_configs.utils import Argument
from cleanerf.nerf_configs.data import dataset_lists

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
    Argument(
        name="nerfacto-frustum-sparsity",
        arg_string="--pipeline.use_frustum_loss True --pipeline.use_sparsity_loss True",
    ),
    Argument(
        name="nerfacto-frustum-TV",
        arg_string="--pipeline.use_frustum_loss True --pipeline.use_total_variation_loss True",
    ),
    Argument(
        name="nerfacto-frustum-regnerf",
        arg_string="--pipeline.use_frustum_loss True --pipeline.use_regnerf_loss True",
    ),
    # Argument(
    #     name="nerfacto-frustum-threshold",
    #     arg_string="--pipeline.use_frustum_loss True --pipeline.use_threshold_loss True",
    # ),
]
arguments_list_of_lists.append(experiments_list)
