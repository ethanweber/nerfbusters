# CleaNeRF

This project is CleaNeRF: 🧹 Erasing Artifacts from Casually Captured NeRFs 🧹. Our project page is [https://ethanweber.me/CleaNeRF/](https://ethanweber.me/CleaNeRF/) with website code [here](https://github.com/ethanweber/CleaNeRF/). CleaNeRF is a post-processing method to clean up NeRFs.

# Project page

See the `gh-pages` branch of this repo for the [project website](https://ethanweber.me/CleaNeRF/).

# Installation

1) Setup conda enviroment

```bash
conda create --name cleanerf -y python=3.8
conda activate cleanerf
python -m pip install --upgrade pip
```

2) Install Nerfstudio and dependencies

> Currently we are using the branch [cleanerf-changes](https://github.com/nerfstudio-project/nerfstudio/tree/cleanerf-changes).

```bash
cd path/to/nerfstudio
pip install -e .
```

3) Install CleaNeRF 

```bash
cd ../
git clone https://github.com/ethanweber/CleaNeRF
cd CleaNeRF
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e .
```

4) Install binvox to voxelize cubes

```bash
mkdir bins
cd bins
wget -O binvox https://www.patrickmin.com/binvox/linux64/binvox?rnd=16811490753710
cd ../
chmod +x bins/binvox
```

Your expected folder structure, should look like this

    repos
    ├── nerfstudio          # nerfstudio files
    └── CleaNeRF            # CleaNeRF files 
       ├── cleanerf
       └── bins
          └── binvox        # binvox to voxelize cubes

# Train 3D diffusion model

The CleaNeRF local 3D diffusion model is trained on local cubes from ShapeNet. To download the ShapeNet dataset, login or create an account at https://shapenet.org and then download the [ShapeNetCore.v2](https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip) dataset.

Then you can train the 3D diffusion model.

```bash
python cleanerf/run.py --config config/shapenet.yaml --name shapenet-experiment
```

We also provide checkpoints for a trained diffusion model [pretrained checkpoint]()

TODO: Ethan, do you want to host checkpoint?

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
export DATASET=aloe;
ns-process-data video --data data/cleanerf-captures/${DATASET}/${DATASET}.mp4 data/cleanerf-captures/${DATASET}/${DATASET}-eval.mp4 --output-dir data/cleanerf-processed-data/${DATASET} --num-frames-target 300;
```

### NeRF experiments

See for [NERF_EXPERIMENTS.md](NERF_EXPERIMENTS.md) to replicate our experiments and evaluation procedure described in the paper.
