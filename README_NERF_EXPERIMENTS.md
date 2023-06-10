# NeRFs Experiments

This section of our documentation assumes that you have the processed data. We will walk you through our NeRF experiments.

# Train models before post-processing techniques

We first train two NeRFs per capture. (1) One which will be the NeRF that we improve with post-processing techniques, and (2) another which will serve as a pseudo ground truth for our evaluations. (1) is only trained on the training split, while (2) is trained on both the training and evaluation splits. We use the nerfacto method for these training runs.

Set these for pretraining the NeRFs.

```bash
export DATASET=aloe;
export BASENAME_EXPERIMENT_NAME=${DATASET}-baseline;
export GT_EXPERIMENT_NAME=${DATASET}-pseudo-gt;
export OUTPUT_FOLDER_PRE=outputs-pretrained/${DATASET};
```

Then set these for post-processing the NeRFs.

```bash
export OUTPUT_FOLDER_POST=outputs-postprocessed/${DATASET};
export RENDER_FOLDER_POST=renders-postprocessed/${DATASET};
```

Train NeRF on train images, evaluate on eval images.

```bash
ns-train nerfacto \
    --vis viewer+wandb \
    --data data/nerfbusters-dataset/${DATASET} \
    --experiment-name ${BASENAME_EXPERIMENT_NAME} \
    --output-dir ${OUTPUT_FOLDER_PRE} \
    nerfstudio-data \
    --eval-mode eval-frame-index \
    --train-frame-indices 0 \
    --eval-frame-indices 1
```

Train NeRF on all (train + eval) images, evaluate on eval images.

```bash
ns-train nerfacto \
    --vis viewer+wandb \
    --data data/nerfbusters-dataset/${DATASET} \
    --experiment-name ${GT_EXPERIMENT_NAME} \
    --output-dir ${OUTPUT_FOLDER_PRE} \
    nerfstudio-data \
    --eval-mode eval-frame-index \
    --train-frame-indices 0 1 \
    --eval-frame-indices 1
```

Then we copy the checkpoints into a folder `outputs-checkpoints`. We also give the checkpoints names in the format `outputs-checkpoints/aloe-baseline.ckpt` and `outputs-checkpoints/aloe-pseudo-gt.ckpt`. We do this for every capture.

# Improve the NeRFs with post-processing

Now we improve the NeRFs with various regularizers, including our Nerfbusters method. Remove `--dry-run` from any of the following commands to actually execute them.

To run the **baseline experiments**, modify the file [nerfbusters/nerf/experiment_configs/experiments_baseline.py](nerfbusters/nerf/experiment_configs/experiments_baseline.py). _We've commented out everything except the aloe scene._

```bash
python scripts/launch_nerf.py train --experiment-name baselines --dry-run
```

To run the **ablations experiments**, modify the file [nerfbusters/nerf/experiment_configs/experiments_ablations.py](nerfbusters/nerf/experiment_configs/experiments_ablations.py). _This will require preprocessing the garbage scene._

```bash
python scripts/launch_nerf.py train --experiment-name ablations --dry-run
```

To run the **pseudo gt experiments**, modify the file [nerfbusters/nerf/experiment_configs/experiments_pseudo_gt.py](nerfbusters/nerf/experiment_configs/experiments_pseudo_gt.py). _We've commented out everything except the aloe scene._

```bash
python scripts/launch_nerf.py train --experiment-name pseudo-gt --dry-run
```

Remove `--dry-run` when the output of the script looks good to you. Notice that can also take any of these commands and run them on your own individually.

# Render out the results

Finally, when the models are trained to completition, you can use them to render videos! Run the following commands to make that happen. Reminder to remove `--dry-run`.

```bash
python scripts/launch_nerf.py render \
    --input-folder ${OUTPUT_FOLDER_POST} \
    --output-folder ${RENDER_FOLDER_POST} \
    --downscale-factor 2 \
    --dry-run
```

You can also use the script `scripts/render_all.sh`.

# Compute the metrics

Now we compute number of the rendered images. This is no `--dry-run` flag for this.

```bash
python scripts/launch_nerf.py metrics \
    --input-folder ${RENDER_FOLDER_POST} \
    --pseudo-gt-experiment-name ${DATASET}---nerfacto---pseudo-gt
```

You can also use the script `scripts/metrics_all.sh`.

# Creating a table from results

You can see our notebook at [notebooks/nerf_table.ipynb](notebooks/nerf_table.ipynb) for creating tables.
