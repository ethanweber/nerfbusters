import argparse
import os
import random
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from dotmap import DotMap
from nerfbusters.data_modules.datamodule import DataModule
from nerfbusters.lightning.nerfbusters_trainer import NerfbustersTrainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/shapenet.yaml",
        type=str,
        help="config file",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--validate-only", action="store_true", help="only validate")
    parser.add_argument("--model-path", type=str, help="path to model ckpt")
    parser.add_argument("--name", type=str, help="name for wandb")
    parser.add_argument("--gpu", type=int, default=0, help="what gpus to use")
    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as file:
        config = yaml.full_load(file)

    config = DotMap(config)
    print(config)

    return args, config


def main(args, config):

    scene_name = ""
    if "scene" in config:
        scene_name += f"_{config.scene}"
    if "train_split_percentage" in config:
        scene_name += f"_{config.train_split_percentage}"

    name = f"{config.dataset}{scene_name}/{config.noise_scheduler}/"
    name += datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if args.name:
        name += f"-{args.name}"
    savepath = f"outputs/diffusion/{name}"
    os.makedirs(savepath, exist_ok=True)

    # lightning trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="Val_acc/dsds_epoch",
        dirpath=f"{savepath}/checkpoints",
        filename="{epoch:02d}-{Val_acc/dsds_epoch:.2f}",
        save_top_k=-1,
        mode="max",
        save_last=True,
    )

    logger = WandbLogger(
        name=name,
        project="nerfbusters-diffusion",
        save_dir=savepath,
    )

    callbacks = [LearningRateMonitor(logging_interval="step"), checkpoint_callback]

    model = NerfbustersTrainer(config, savepath=savepath)
    data_module = DataModule(**config.toDict())

    trainer = pl.Trainer.from_argparse_args(
        config,
        accelerator="gpu",
        devices=[args.gpu],
        precision=32,
        max_epochs=100,
        check_val_every_n_epoch=10,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
    )

    if args.validate_only:
        if os.path.isfile(args.model_path):
            statedict = torch.load(args.model_path)
            statedict = statedict["state_dict"]
            model.load_state_dict(statedict)

        trainer.test(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":

    args, config = parse_args()
    set_seed(args.seed)
    main(args, config)
