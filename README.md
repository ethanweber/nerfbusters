# 👻 Nerfbusters 🧹

This project is 👻 Nerfbusters 🧹: Removing Ghostly Artifacts from Casually Captured NeRFs. Our project page is [https://ethanweber.me/nerfbusters/](https://ethanweber.me/nerfbusters/) with website code [here](https://github.com/ethanweber/nerfbusters/). nerfbusters is a post-processing method to clean up NeRFs.

# Project page

See the `gh-pages` branch of this repo for the [project website](https://ethanweber.me/nerfbusters/).

# Installation

1. Setup conda enviroment

```bash
conda create --name nerfbusters -y python=3.8
conda activate nerfbusters
python -m pip install --upgrade pip
```

2. Install Nerfstudio and dependencies. Installation guide can be found [install nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html)

> Currently we are using the branch [nerfbusters-changes](https://github.com/nerfstudio-project/nerfstudio/tree/nerfbusters-changes). You may have to run the viewer locally if you want full functionality.

```bash
cd path/to/nerfstudio
pip install -e .
pip install torch==1.13.1 torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

3. Install Nerfbusters

```bash
cd ../
git clone https://github.com/ethanweber/nerfbusters
cd nerfbusters
pip install -e .
```

4. Install binvox to voxelize cubes

```bash
mkdir bins
cd bins
wget -O binvox https://www.patrickmin.com/binvox/linux64/binvox?rnd=16811490753710
cd ../
chmod +x bins/binvox
```

# Train 3D diffusion model

The Nerfbusters local 3D diffusion model is trained on local cubes from ShapeNet. To download the ShapeNet dataset, login or create an account at https://shapenet.org and then download the [ShapeNetCore.v2](https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip) dataset.

Your expected folder structure, should look like this

    repos
    ├── nerfstudio             # nerfstudio files (but this can live anywhere)
    └── nerfbusters            # nerfbusters files
       ├── nerfbusters
       ├── data
          └── ShapeNetCore.v2  # ShapeNet data
       └── bins
          └── binvox           # binvox to voxelize cubes

Then you can train the 3D diffusion model.

```bash
python nerfbusters/run.py --config config/shapenet.yaml --name shapenet-experiment
```

We also provide checkpoints for a trained diffusion model [pretrained checkpoint](https://data.nerf.studio/nerfbusters-diffusion-cube-weights.ckpt)

# Running Nerfbusters in-the-wild

Here we explain how you'd run Nerfbusters on your data to remove floater artifacts.

Train a Nerfacto model. Make a note of the path where the model checkpoint is saved.

```bash
ns-train nerfacto --data path/to/data
```

Set the checkpoint path to use later.

```
# TODO: support using a config instead of a checkpoint...
export NERF_CHECKPOINT_PATH=path/to/nerfstudio_models/step-000029999.ckpt
```

Now you'll need to download the diffusion model weights. Run `python nerfbusters/download_nerfbusters_dataset.py diffusion-cube-weights` if you haven't downloaded them yet or haven't trained your own model. Then you can run post-processing with our method. _If you wan't to use a your own / a specific 3D diffusion model, then update the config `--pipeline.diffusioncube_config_path` and weights checkpoint with `--pipeline.diffusioncube_ckpt_path`._

Now you can fine-tune (i.e., post-process) with our Nerfbusters method!

```bash
ns-train nerfbusters --data path/to/data --pipeline.nerf-checkpoint-path $NERF_CHECKPOINT_PATH nerfstudio-data --eval-mode train-split-fraction
```

Finally, render a path!

```bash
ns-render --load-config path/to/config.yml  --traj filename --camera-path-filename path/to/camera-path.json --output-path renders/my-render.mp4
```

# Using Nerfbusters to evaluate methods

Here we use the Nerfbusters evaluation procedure to run the experiments in our paper. You'll need our dataset for this step. You can download it with the following commands. We provide both the original videos or the already-processed versions of our dataset. The following commands will write to the `data/` folder.

To download the already-processed version, run the following command. This download is 12.6GB. This writes to the `data/nerfbusters-dataset` folder.

```bash
python nerfbusters/download_nerfbusters_dataset.py dataset
```

If you want to download the two orignial videos per "capture", run this. The download is 381MB. This writes to the `data/nerfbusters-captures` folder.

```bash
python nerfbusters/download_nerfbusters_dataset.py captures
```

You'll notice that the file structure has two videos where one is for training and the `-eval` one is for evaluation. If you want to replicate the processing that we did, simply run the following.

```bash
export DATASET=aloe;
ns-process-data video --data data/nerfbusters-captures/${DATASET}/${DATASET}.mp4 data/nerfbusters-captures/${DATASET}/${DATASET}-eval.mp4 --output-dir data/nerfbusters-processed-data/${DATASET} --num-frames-target 300;
```

*If you want to download any of the data from Google Drive directly, you can access the folder [here](https://drive.google.com/drive/folders/19NkX_FXLMnD4Mzv5efxESLrSln5TN1ZB?usp=drive_link).*

### NeRF experiments

See for [README_NERF_EXPERIMENTS.md](README_NERF_EXPERIMENTS.md) to replicate our experiments and evaluation procedure described in the paper.


# Citing

If you find this code or data useful for your research, please consider citing the following paper:

    @inproceedings{Nerfbusters2023,
	    Title        = {Nerfbusters: Removing Ghostly Artifacts from Casually Captured NeRFs},
	    Author       = {Frederik Warburg* and Ethan Weber* and Matthew Tancik and Aleksander Hołyński and Angjoo Kanazawa},
        Booktile     = {ICCV},
	    Year         = {2023}
    }
