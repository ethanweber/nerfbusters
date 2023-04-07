import math
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision import transforms


def convert_depth(npyDistance):
    intWidth = 1024
    intHeight = 768
    fltFocal = 886.81

    # convert depth from distance to optical center to depth from image plane
    npyImageplaneX = (
        np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
        .reshape(1, intWidth)
        .repeat(intHeight, 0)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneY = (
        np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight)
        .reshape(intHeight, 1)
        .repeat(intWidth, 1)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / (np.linalg.norm(npyImageplane, 2, 2) * fltFocal + 1e-6)

    return npyDepth


def load_depth(depth_path):
    f = h5py.File(depth_path, "r")
    depth = f["dataset"][()]
    depth = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
    f.close()
    return depth


def load_color(color_path):
    img = Image.open(color_path)
    return np.array(img)


class Hypersim(data.Dataset):
    def __init__(self, path, crop_size=(48, 48), train=True, input="rgb", transform=None, train_split_percentage=0.01):
        super().__init__()

        self.crop_size = crop_size
        self.train = train
        self.input = input
        self.depth_paths = []
        self.color_paths = []

        for seq in os.listdir(path):
            for cam in ["00", "01"]:
                seq_path = os.path.join(path, seq, "images", f"scene_cam_{cam}_final_preview")
                if os.path.exists(seq_path):
                    frames = [f[6:10] for f in os.listdir(seq_path)]
                    frames = np.unique(frames)

                    for frame in frames:
                        depth_path = os.path.join(
                            path, seq, "images", f"scene_cam_{cam}_geometry_hdf5", f"frame.{frame}.depth_meters.hdf5"
                        )
                        # depth_path = os.path.join(
                        #    path, seq, "images", f"scene_cam_{cam}_geometry_preview", f"frame.{frame}.depth_meters.png"
                        # )
                        color_path = os.path.join(
                            path, seq, "images", f"scene_cam_{cam}_final_preview", f"frame.{frame}.color.jpg"
                        )

                        self.depth_paths.append(depth_path)
                        self.color_paths.append(color_path)

        split = int(len(self.depth_paths) * train_split_percentage)
        if train:
            self.depth_paths = self.depth_paths[split:]
            self.color_paths = self.color_paths[split:]
        else:
            self.depth_paths = self.depth_paths[:split]
            self.color_paths = self.color_paths[:split]

        if not self.train:
            torch.manual_seed(0)

            self.x = np.random.randint(0, 1024 - self.crop_size[0], len(self.depth_paths))
            self.y = np.random.randint(0, 768 - self.crop_size[1], len(self.depth_paths))
            self.noise = torch.randn((len(self.depth_paths), len(self.input), self.crop_size[0], self.crop_size[1]))
            self.noise_levels = 0.5
            self.test_angle = 0
            self.test_percent_of_scene = 0.05
            self.test_dilation_iterations = 3

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, index):

        depth = load_depth(self.depth_paths[index])
        depth = convert_depth(depth)

        color = load_color(self.color_paths[index])

        # normalize depth between -1 and 1
        # depth = np.clip(depth / 50.0, 0, 1) * 2.0 - 1.0
        # depth = (depth - np.mean(depth)) / (np.std(depth) + 1e-6) # [-mu, depth.max() - mu]
        # depth = depth / torch.min(depth.max() - depth.min(), 1)
        # depth = np.clip(depth, 0, 1) * 2.0 - 1.0
        depth = np.log(depth + 1e-6)
        depth = np.clip(depth, -1, 1)

        # normalize color between -1 and 1
        color = (color / 255.0) * 2.0 - 1.0

        # concatenate depth and color
        if self.input == "rgbd":
            image = np.concatenate([color, depth[:, :, None]], 2).astype(np.float32)
        elif self.input == "d":
            image = depth[:, :, None].astype(np.float32)
        elif self.input == "rgb":
            image = color.astype(np.float32)

        # crop image
        h, w, _ = image.shape
        th, tw = self.crop_size

        if self.train:
            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)

            image = image[y1 : y1 + th, x1 : x1 + tw, :]
            image = np.transpose(image, (2, 0, 1))

            return {"input": image, "scale": 0}
        else:

            x1 = self.x[index]
            y1 = self.y[index]

            image = image[y1 : y1 + th, x1 : x1 + tw, :]
            image = np.transpose(image, (2, 0, 1))

            noise_level = self.noise_levels

            corrupted_image = noise_level * self.noise[index] + (1 - noise_level) * image
            corrupted_image = torch.clamp(corrupted_image, -1.0, 1.0)

            return {"corrupted_input": corrupted_image, "input": image, "scale": 0}
