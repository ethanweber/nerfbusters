import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from dotmap import DotMap
from lightning.magic_eraser_2d import MagicEraser2D
from PIL import Image
from tqdm import tqdm


def denormalize(x):
    return (x + 1) / 2


normalize = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def get_image_paths(path):

    image_paths = sorted([os.path.join(path, f) for f in os.listdir(path)])

    return image_paths


def load_model(model_path):

    args = {"dataset": "patches2d", "noise_scheduler": "ddim"}
    args = DotMap(args)

    model = MagicEraser2D(args)

    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["state_dict"])

    return model


def optimize_image(model, x_corrupted, lr=0.01, num_steps=100):

    x = x_corrupted.clone()
    c, h, w = x.shape

    # optimize patch
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=lr)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    for i in tqdm(range(num_steps)):

        # very primitive data loader
        x_crops = []
        for _ in range(512):
            i = torch.randint(0, h - 32, (1,))
            j = torch.randint(0, w - 32, (1,))
            x_crop = x[:, i : i + 32, j : j + 32]
            x_crops.append(x_crop)
        x_crops = torch.stack(x_crops)

        # optimize
        optimizer.zero_grad()
        _ = model.sds_loss(x_crops)
        optimizer.step()
        # scheduler.step()

    return x.detach()


def main(data_path, model_path):

    image_paths = get_image_paths(data_path)[:1]
    model = load_model(model_path)
    model.cuda()
    model.noise_scheduler.alphas_cumprod = model.noise_scheduler.alphas_cumprod.cuda()
    model.eval()

    model.min_step = 10
    model.max_step = 400
    steps = 300

    results = []
    for image_path in image_paths:
        image_corrupted = Image.open(image_path)
        image_corrupted = torchvision.transforms.ToTensor()(image_corrupted)
        image_corrupted = normalize(image_corrupted)
        image_corrupted = image_corrupted.cuda()

        image_decorrupted = optimize_image(model, image_corrupted, lr=1e-2, num_steps=steps)

        image_decorrupted = denormalize(image_decorrupted)
        image_decorrupted = torch.clamp(image_decorrupted.cpu(), 0, 1).numpy()

        results.append(image_decorrupted)
        save_path = image_path.replace("images", "images_decorrupted")

        image_decorrupted = (image_decorrupted.transpose(1, 2, 0) * 255).astype(np.uint8)
        plt.imsave(save_path, image_decorrupted)

    return results


if __name__ == "__main__":

    data_path = "/home/warburg/repo/3D-magic-eraser/renders/kitchen_train_split_percentage01/images/"
    os.makedirs(data_path.replace("images", "images_decorrupted"), exist_ok=True)
    model_path = "../lightning_logs/patches2d_kitchen_0.1/ddim/checkpoints/last.ckpt"

    main(data_path, model_path)
