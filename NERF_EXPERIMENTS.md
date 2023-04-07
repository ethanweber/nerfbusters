# Training NeRFs Experiments

First, you'll need to process some data with our favorite tool, COLMAP! `ns-process-data` takes care of this for us.

```bash
export DATASET=stdunstanfish
export BASENAME_EXPERIMENT_NAME=${DATASET}-train-baseline
export VISIBILITY_EXPERIMENT_NAME=${DATASET}-train+eval-baseline
export OUTPUT_FOLDER=projects/magic_eraser/outputs/${DATASET}
export RENDER_FOLDER=projects/magic_eraser/outputs-renders/${DATASET}
```

# Using our viewer

Visualize training with our local viewer (with modifications from the hosted viewer)

Follow the steps [here](https://docs.nerf.studio/en/latest/developer_guides/viewer/viewer_overview.html) and then go to [http://localhost:4001/?websocket_url=ws://localhost:5555](http://localhost:4001/?websocket_url=ws://localhost:5555).

# Launch NeRF training experiments

> Train a NeRF and then improve it! We train two NeRFs, one which can serve as a pseudo ground truth for our evaluations. We don't need to run eval on these since it will slow things down and use more memory.

Train NeRF on train images, eval on eval

```bash
ns-train magic-eraser --vis viewer+wandb --data projects/magic_eraser/data/${DATASET} --experiment-name ${BASENAME_EXPERIMENT_NAME} --output-dir ${OUTPUT_FOLDER} --steps_per_eval_batch 0 --steps_per_eval_image 0 --steps_per_eval_all_images 0
```

Train NeRF on all images, eval on eval. This needed to compute visibility masks for computing metrics.

```bash
ns-train magic-eraser --vis viewer+wandb --data projects/magic_eraser/data/${DATASET} --experiment-name ${VISIBILITY_EXPERIMENT_NAME} --output-dir ${OUTPUT_FOLDER} --steps_per_eval_batch 0 --steps_per_eval_image 0 --steps_per_eval_all_images 0 nerfstudio-data --train-frame-indices 0 1
```

TODO: export the visiblity masks, so we can use them for the other experiments
```
```

# Improve the NeRFs

Next, improve your NeRF with various regularizers. Wahoo!

With the baseline experiment name...

```bash
python projects/magic_eraser/scripts/launch_nerf.py train --data projects/magic_eraser/data/${DATASET} --output-folder ${OUTPUT_FOLDER} --baseline-experiment-name ${BASENAME_EXPERIMENT_NAME} --dry-run
```

Or without and then it's up to your configs to handle everything!

```bash
python projects/magic_eraser/scripts/launch_nerf.py train --dry-run
```

Remove `--dry-run` when the output of the script looks good to you. Notice that can also take any of these commands and run them on your own individually.

Finally, when the models are trained to completition, you can use them to render videos! Yay! :) Run the following commands to make that happen.

```bash
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder ${OUTPUT_FOLDER} \
    --output-folder ${RENDER_FOLDER} \
    --downscale-factor 1 \
    --dry-run
```

Here is how we run it for all our data.

```bash
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/flowers \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/flowers \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/pikachu \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/pikachu \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/pipe \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/pipe \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/plant \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/plant \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/roses \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/roses \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/table \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/table \
    --downscale-factor 2;

python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/aloe \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/aloe \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/picnic \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/picnic \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/table \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/table \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/century \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/century \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/plant \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/plant \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/garbage \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/garbage \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/art \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/art \
    --downscale-factor 2;



python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/flowers \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/flowers \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/pipe \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/pipe \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/roses \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/roses \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/pikachu \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/pikachu \
    --downscale-factor 2;
python projects/magic_eraser/scripts/launch_nerf.py render \
    --input-folder /shared/ethanweber/magic-eraser/nerf-outputs/car \
    --output-folder /shared/ethanweber/magic-eraser/nerf-renders/car \
    --downscale-factor 2;
```


<!-- Argument(
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
    ), -->

# Compute the metrics

Now we compute number of the rendered images.

```bash
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder ${RENDER_FOLDER} \
    --visibility-experiment-name ${VISIBILITY_EXPERIMENT_NAME}
```

Here is how we run it for all our data.

```bash
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/flowers \
    --visibility-experiment-name nerfacto-visibility---flowers;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/pikachu \
    --visibility-experiment-name nerfacto-visibility---pikachu;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/pipe \
    --visibility-experiment-name nerfacto-visibility---pipe;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/plant \
    --visibility-experiment-name nerfacto-visibility---plant;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/roses \
    --visibility-experiment-name nerfacto-visibility---roses;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/table \
    --visibility-experiment-name nerfacto-visibility---table;
```

```
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/aloe \
    --visibility-experiment-name aloe---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/picnic \
    --visibility-experiment-name picnic---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/table \
    --visibility-experiment-name table---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/century \
    --visibility-experiment-name century---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/plant \
    --visibility-experiment-name plant---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/garbage \
    --visibility-experiment-name garbage---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/art \
    --visibility-experiment-name art---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/flowers \
    --visibility-experiment-name flowers---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/pikachu \
    --visibility-experiment-name pikachu---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/pipe \
    --visibility-experiment-name pipe---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/plant \
    --visibility-experiment-name plant---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/roses \
    --visibility-experiment-name roses---nerfacto---visibility-split;
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/car \
    --visibility-experiment-name car---nerfacto---visibility-split;
```

Our chosen data...
```bash
# TODO: add table
python projects/magic_eraser/scripts/launch_nerf.py metrics \
    --input-folder /shared/ethanweber/magic-eraser/nerf-renders/garbage \
    --visibility-experiment-name garbage---nerfacto---visibility-split;
```
