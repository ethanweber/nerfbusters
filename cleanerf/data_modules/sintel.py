import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import Sintel


class Sintel2Dpatches(data.Dataset):
    def __init__(self, root, train, crop_size=(32, 32), transform=None):

        self.data = Sintel(root, split="train")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.h = 436
        self.w = 1024
        self.crop_size = crop_size
        self.count = 0
        self.refresh()

    def refresh(self):
        self.idx = torch.randint(0, self.data.__len__(), (50000,))
        self.h_crops = torch.randint(0, self.h - self.crop_size[0], (50000,))
        self.w_crops = torch.randint(0, self.w - self.crop_size[1], (50000,))

    def __len__(self):
        return len(self.h_crops)

    def __getitem__(self, idx):
        index = self.idx[idx]
        im, _, _ = self.data.__getitem__(index)

        i, j = self.h_crops[idx], self.w_crops[idx]
        h, w = self.crop_size

        im = self.to_tensor(im)

        patch = im[:, i : i + h, j : j + w]

        assert patch.shape == (3, 32, 32)

        patch = self.normalize(patch)

        return patch
