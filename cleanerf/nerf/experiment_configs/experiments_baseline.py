from cleanerf.nerf.experiment_configs.utils import Argument

arguments_list_of_lists = []

output_folder = "outputs-postprocessed"

dataset_lists = [
    Argument(
        name="aloe",
        arg_string=f"--data data/cleanerf-dataset/aloe --pipeline.nerf_checkpoint_path outputs-checkpoints/aloe-baseline.ckpt --output-dir {output_folder}/aloe",
    ),
    # Argument(
    #     name="art",
    #     arg_string=f"--data data/cleanerf-dataset/art --pipeline.nerf_checkpoint_path outputs-checkpoints/art-baseline.ckpt --output-dir {output_folder}/art",
    # ),
    # Argument(
    #     name="car",
    #     arg_string=f"--data data/cleanerf-dataset/car --pipeline.nerf_checkpoint_path outputs-checkpoints/car-baseline.ckpt --output-dir {output_folder}/car",
    # ),
    # Argument(
    #     name="century",
    #     arg_string=f"--data data/cleanerf-dataset/century --pipeline.nerf_checkpoint_path outputs-checkpoints/century-baseline.ckpt --output-dir {output_folder}/century",
    # ),
    # Argument(
    #     name="flowers",
    #     arg_string=f"--data data/cleanerf-dataset/flowers --pipeline.nerf_checkpoint_path outputs-checkpoints/flowers-baseline.ckpt --output-dir {output_folder}/flowers",
    # ),
    # Argument(
    #     name="garbage",
    #     arg_string=f"--data data/cleanerf-dataset/garbage --pipeline.nerf_checkpoint_path outputs-checkpoints/garbage-baseline.ckpt --output-dir {output_folder}/garbage",
    # ),
    # Argument(
    #     name="picnic",
    #     arg_string=f"--data data/cleanerf-dataset/picnic --pipeline.nerf_checkpoint_path outputs-checkpoints/picnic-baseline.ckpt --output-dir {output_folder}/picnic",
    # ),
    # Argument(
    #     name="pikachu",
    #     arg_string=f"--data data/cleanerf-dataset/pikachu --pipeline.nerf_checkpoint_path outputs-checkpoints/pikachu-baseline.ckpt --output-dir {output_folder}/pikachu",
    # ),
    # Argument(
    #     name="pipe",
    #     arg_string=f"--data data/cleanerf-dataset/pipe --pipeline.nerf_checkpoint_path outputs-checkpoints/pipe-baseline.ckpt --output-dir {output_folder}/pipe",
    # ),
    # Argument(
    #     name="plant",
    #     arg_string=f"--data data/cleanerf-dataset/plant --pipeline.nerf_checkpoint_path outputs-checkpoints/plant-baseline.ckpt --output-dir {output_folder}/plant",
    # ),
    # Argument(
    #     name="roses",
    #     arg_string=f"--data data/cleanerf-dataset/roses --pipeline.nerf_checkpoint_path outputs-checkpoints/roses-baseline.ckpt --output-dir {output_folder}/roses",
    # ),
    # Argument(
    #     name="table",
    #     arg_string=f"--data data/cleanerf-dataset/table --pipeline.nerf_checkpoint_path outputs-checkpoints/table-baseline.ckpt --output-dir {output_folder}/table",
    # ),
]
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
    )
]
arguments_list_of_lists.append(experiments_list)
