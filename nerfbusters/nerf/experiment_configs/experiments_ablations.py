from nerfbusters.nerf.experiment_configs.utils import Argument

dataset_lists = [
    Argument(
        name="garbage",
        arg_string="--data data/nerfbusters-dataset/garbage --pipeline.nerf_checkpoint_path outputs-checkpoints/garbage-baseline.ckpt --output-dir outputs-postprocessed/garbage",
    )
]

arguments_list_of_lists = []
arguments_list_of_lists.append(dataset_lists)

experiments_list = [
    # sampling
    Argument(
        name="nerfacto-visibility-cube-sampling-weights",
        arg_string="--pipeline.use_visibility_loss True --pipeline.use_singlestep_cube_loss True --pipeline.weight_grid_quantity weights",
    ),
    Argument(
        name="nerfacto-visibility-cube-sampling-uniform",
        arg_string="--pipeline.use_visibility_loss True --pipeline.use_singlestep_cube_loss True --pipeline.sample_method uniform",
    ),
    # activation
    Argument(
        name="nerfacto-visibility-cube-activation-sigmoid",
        arg_string="--pipeline.use_visibility_loss True --pipeline.use_singlestep_cube_loss False --pipeline.use_cube_loss True --pipeline.density_to_x_activation sigmoid",
    ),
    Argument(
        name="nerfacto-visibility-cube-activation-clamp",
        arg_string="--pipeline.use_visibility_loss True --pipeline.use_singlestep_cube_loss False --pipeline.use_cube_loss True --pipeline.density_to_x_activation clamp",
    ),
    # cubesize
    Argument(
        name="nerfacto-visibility-cube-cubescale-1-10",
        arg_string="--pipeline.use_visibility_loss True --pipeline.use_singlestep_cube_loss True --pipeline.cube_scale_perc_range 0.1 0.10",
    ),
    Argument(
        name="nerfacto-visibility-cube-cubescale-10-20",
        arg_string="--pipeline.use_visibility_loss True --pipeline.use_singlestep_cube_loss True --pipeline.cube_scale_perc_range 0.10 0.20",
    )
]
arguments_list_of_lists.append(experiments_list)
