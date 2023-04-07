from cleanerf.nerf_configs.utils import Argument

# if local folder...
# output_folder = "projects/magic_eraser/outputs"

# if shared folder...
output_folder = "/shared/ethanweber/magic-eraser/nerf-outputs"

dataset_lists = [
    Argument(
        name="flowers",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/flowers --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/flowers-baseline.ckpt --output-dir {output_folder}/flowers",
    ),
    Argument(
        name="pipe",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/pipe --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/pipe-baseline.ckpt --output-dir {output_folder}/pipe",
    ),
    Argument(
        name="roses",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/roses --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/roses-baseline.ckpt --output-dir {output_folder}/roses",
    ),
    Argument(
        name="pikachu",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/pikachu --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/pikachu-baseline.ckpt --output-dir {output_folder}/pikachu",
    ),
    Argument(
        name="car",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/car --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/car-baseline.ckpt --output-dir {output_folder}/car",
    ),
    Argument(
        name="aloe",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/aloe --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/aloe-baseline.ckpt --output-dir {output_folder}/aloe",
    ),
    Argument(
        name="picnic",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/picnic --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/picnic-baseline.ckpt --output-dir {output_folder}/picnic",
    ),
    Argument(
        name="table",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/table --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/table-baseline.ckpt --output-dir {output_folder}/table",
    ),
    Argument(
        name="century",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/century --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/century-baseline.ckpt --output-dir {output_folder}/century",
    ),
    Argument(
        name="plant",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/plant --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/plant-baseline.ckpt --output-dir {output_folder}/plant",
    ),
    Argument(
        name="garbage",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/garbage --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/garbage-baseline.ckpt --output-dir {output_folder}/garbage",
    ),
    Argument(
        name="art",
        arg_string=f"--data /shared/ethanweber/magic-eraser/nerf-data/art --pipeline.nerf_checkpoint_path /shared/ethanweber/magic-eraser/nerf-checkpoints/art-baseline.ckpt --output-dir {output_folder}/art",
    ),
]
