import os
from random import shuffle

import pytorch_lightning as pl

from nerfbusters.data_modules.cubes3d import Cubes3D
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 512,
        val_batch_size: int = 32,
        workers=8,
        dataset="mnist",
        scene="room",
        train_split_percentage=1,
        cubes_filename=None,
        percentage_of_empty_cubes=0.1,
        train_len=None,
        val_cubes_filename=None,
        val_len=None,
        input="rgbd",
        **args,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.dataset = dataset
        self.workers = workers
        self.scene = scene
        self.train_split_percentage = train_split_percentage
        self.cubes_filename = cubes_filename
        self.percentage_of_empty_cubes = percentage_of_empty_cubes
        self.train_len = train_len
        self.val_cubes_filename = val_cubes_filename
        self.val_len = val_len
        self.input = input

    def setup(self, stage="fit"):

        assert self.cubes_filename is not None, "cubes_filename must be specified for cubes dataset"
        # check if cubes filename is a folder

        self.train_dataset = Cubes3D(
            self.cubes_filename,
            train=True,
            percentage_of_empty_cubes=self.percentage_of_empty_cubes,
            train_len=self.train_len,
            val_len=self.val_len,
        )
        self.val_dataset = Cubes3D(
            self.val_cubes_filename,
            train=False,
            percentage_of_empty_cubes=self.percentage_of_empty_cubes,
            train_len=self.train_len,
            val_len=self.val_len,
        )
        self.test_dataset = Cubes3D(
            self.val_cubes_filename,
            train=False,
            percentage_of_empty_cubes=self.percentage_of_empty_cubes,
            train_len=self.train_len,
            val_len=self.val_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )

        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )

        return dataloader
