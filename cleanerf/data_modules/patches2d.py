import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision import transforms


class Patches2D(data.Dataset):

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}

    def __init__(self, root, crop_size=(32, 32), train=True, transform=None, train_split_percentage=None):
        super().__init__()

        self.root = Path(root)
        self.paths = sorted(path for path in self.root.rglob("*") if path.suffix.lower() in self.IMG_EXTENSIONS)
        self.paths = [p for p in self.paths if ("depth" not in str(p) and "normal" not in str(p))]
        self.paths = np.array(self.paths)

        print(f"Found {len(self.paths)} images in {self.root}")
        self.train = train

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(self.paths)
        num_train_images = math.ceil(num_images * train_split_percentage)
        num_eval_images = num_images - num_train_images

        # make sure at least one image is used for evaluation
        num_train_images = num_train_images - 1 if num_eval_images == 0 else num_train_images
        num_eval_images = 1 if num_eval_images == 0 else num_eval_images

        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images

        self.paths = self.paths[i_train] if self.train else self.paths[i_eval]
        print(f'Using {len(self.paths)} images in {self.root} for {"training" if train else "evaluation"}')

        to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.images = [Image.open(p).convert("RGB") for p in self.paths]
        self.images = [im.resize((640, 480)) for im in self.images]
        self.images = [to_tensor(im) for im in self.images]
        c, self.h, self.w = self.images[0].shape
        self.crop_size = crop_size

        self.count = 0
        self.refresh()

    def refresh(self):
        self.idx = torch.randint(0, len(self.images), (50000,))
        self.h_crops = torch.randint(0, self.h - self.crop_size[0], (50000,))
        self.w_crops = torch.randint(0, self.w - self.crop_size[1], (50000,))

    def __repr__(self):
        return f'FolderOfImages(root="{self.root}", len: {len(self)})'

    def __len__(self):
        return len(self.h_crops)

    def __getitem__(self, key):

        self.count += 1
        if self.count % 50000 == 0:
            self.refresh()
            self.count = 1

        idx = self.idx[key]
        i, j = self.h_crops[key], self.w_crops[key]
        h, w = self.crop_size

        patch = self.images[idx][:, i : i + h, j : j + w]

        assert patch.shape == (3, 32, 32)

        patch = self.normalize(patch)

        return patch
