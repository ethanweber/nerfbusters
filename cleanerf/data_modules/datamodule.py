import os
from random import shuffle

import pytorch_lightning as pl
from datasets import load_dataset
from cleanerf.data_modules.cubes3d import Cubes3D
from cleanerf.data_modules.hypersim import Hypersim
from cleanerf.data_modules.patches2d import Patches2D
from cleanerf.data_modules.plane3d import Plane3D
from cleanerf.data_modules.sintel import Sintel2Dpatches
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST


def get_transform_test():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def get_transform_train():
    return transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/warburg/data",
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

        self.data_dir = data_dir
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

        train_transform = get_transform_train()
        test_transform = get_transform_test()

        if self.dataset == "patches2d":

            if self.scene == "room":
                path = f"/home/warburg/data/nerf_llff_data/{self.scene}/images/"

            elif self.scene == "kitchen":
                path = f"/home/warburg/repo/3D-magic-eraser/data/nerfstudio/{self.scene}/images/"

            elif self.scene == "synthetic":
                path = "/home/warburg/data/nerf_synthetic/"

            self.train_dataset = Patches2D(
                path,
                train=True,
                transform=train_transform,
                train_split_percentage=self.train_split_percentage,
            )
            self.val_dataset = Patches2D(
                path, train=False, transform=test_transform, train_split_percentage=self.train_split_percentage
            )

        elif self.dataset == "sintel":
            self.train_dataset = Sintel2Dpatches(self.data_dir, train=True)
            self.val_dataset = Patches2D(
                f"/home/warburg/data/nerf_llff_data/room/images/", train=False, transform=test_transform
            )

        elif self.dataset == "mnist":
            self.train_dataset = MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
            self.val_dataset = MNIST(self.data_dir, train=False, download=True, transform=transforms.ToTensor())

        elif self.dataset == "fashionmnist":
            self.train_dataset = FashionMNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
            self.val_dataset = FashionMNIST(self.data_dir, train=False, download=True, transform=transforms.ToTensor())

        elif self.dataset == "cifar10":

            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.train_dataset = CIFAR10(self.data_dir, train=True, download=True, transform=transform)
            self.val_dataset = CIFAR10(self.data_dir, train=False, download=True, transform=transform)

        elif self.dataset == "hypersim":

            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            print("loading hypersim dataset")
            print("==> using: ", self.input)
            self.train_dataset = Hypersim(self.data_dir, train=True, input=self.input, transform=train_transform)
            self.test_dataset = Hypersim(self.data_dir, train=False, input=self.input, transform=train_transform)
            self.val_dataset = Hypersim(self.data_dir, train=False, input=self.input, transform=train_transform)

        elif self.dataset == "flowers":

            self.train_dataset = load_dataset(
                "huggan/flowers-102-categories",
                split="train",
            )
            self.val_dataset = load_dataset(
                "huggan/flowers-102-categories",
                split="train",
            )  # doesnt have a test split, so we use the train split

            # Preprocessing the datasets and DataLoaders creation.
            augmentations = transforms.Compose(
                [
                    transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

            def tf(examples):
                images = [augmentations(image.convert("RGB")) for image in examples["image"]]
                return {"input": images}

            self.train_dataset.set_transform(tf)
            self.val_dataset.set_transform(tf)

        elif self.dataset == "cubes":
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

        elif self.dataset == "planes":

            self.train_dataset = Plane3D(
                train=True,
                train_len=self.train_len,
            )
            self.val_dataset = Plane3D(
                train=False,
                val_len=self.val_len,
            )
            self.test_dataset = Plane3D(
                train=False,
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
