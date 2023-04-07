# CleaNeRF

This project is CleaNeRF: 🧹 Erasing Artifacts from Casually Captured NeRFs 🧹. Our project page is [https://ethanweber.me/CleaNeRF/](https://ethanweber.me/CleaNeRF/) with website code [here](https://github.com/ethanweber/CleaNeRF/). CleaNeRF is a post-processing method to clean up NeRFs.

# Project page

See the `gh-pages` branch of this repo for the [project website](https://ethanweber.me/CleaNeRF/) code.

# Installation

First, you need to install the nerfstudio environment. Then you can install additional the dependencies for CleaNeRF as follows.

Install Nerfstudio and dependencies

> Currently we are using the branch [cleanerf-changes](https://github.com/nerfstudio-project/nerfstudio/tree/cleanerf-changes).

```bash
cd /path/to/nerfstudio
pip install -e .
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install CleaNeRF

```bash
conda create --name cleanerf -y python=3.8
conda activate cleanerf
python -m pip install --upgrade pip
pip install -e .
```

# Train 3D diffusion model

TODO: explain how to download the shapenet dataset

Then you can train the 3D diffusion model.

```bash
CUDA_VISIBLE_DEVICES=0 python cleanerf/run.py --config config/shapenet-cubes-15K.yaml --name shapenet-experiment
```

# Running CleaNeRF in-the-wild

Here we explain how you'd run CleaNeRF on your data to remove floater artifacts.

Train a Nerfacto model. Make a note of the path where the model checkpoint is saved.

```bash
ns-train nerfacto --data data/twitter/stdunstanfish
```

Set the checkpoint path to use later.

```
# TODO: support using a config instead of a checkpoint...
export NERF_CHECKPOINT_PATH=outputs/stdunstanfish/nerfacto/2023-04-04_002153/nerfstudio_models/step-000029999.ckpt
```

Now you'll need to download the diffusion model weights. Run `python scripts/download_cleanerf_dataset.py diffusion-cube-weights` if you haven't downloaded them yet or haven't trained your own model. Then you can run post-processing with our method. *If you wan't to use a your own / a specific 3D diffusion model, then update the config `--pipeline.diffusioncube_config_path` and weights checkpoint with `--pipeline.diffusioncube_ckpt_path`.*

Now you can fine-tune (i.e., post-process) with our CleaNeRF method!

```bash
ns-train cleanerf --data data/twitter/stdunstanfish --pipeline.nerf-checkpoint-path $NERF_CHECKPOINT_PATH nerfstudio-data --eval-mode train-split-fraction
```

# Using CleaNeRF to evaluate methods

Here we use the CleaNeRF evaluation procedure to run the experiments in our paper. You'll need our dataset for this step. You can download it with the following commands. We provide both the original videos or the already-processed versions of our dataset. The following commands will write to the `data/` folder.

To download the already-processed version, run the following command. This download is 12.6GB. This writes to the `data/cleanerf-dataset` folder.

```bash
python scripts/download_cleanerf_dataset.py dataset
```

If you want to download the two orignial videos per "capture", run this. The download is 381MB. This writes to the `data/cleanerf-captures` folder.

```bash
python scripts/download_cleanerf_dataset.py captures
```

You'll notice that the file structure has two videos where one is for training and the `-eval` one is for evaluation. If you want to replicate the processing that we did, simply run the following.

```bash
export DATASET=aloe
ns-process-data video --data data/${DATASET}/${DATASET}.mp4 projects/magic_eraser/captures/${DATASET}/${DATASET}-eval.mp4 --output-dir projects/magic_eraser/data/${DATASET} --num-frames-target 300
```

### NeRF experiments

See for [NERF_EXPERIMENTS.md](NERF_EXPERIMENTS.md) to replicate our experiments and evaluation procedure described in the paper.