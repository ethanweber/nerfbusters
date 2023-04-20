import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel
from nerfbusters.lightning.dsds_loss import DSDSLoss
from nerfbusters.models.model import get_model
from nerfbusters.utils import metrics
from nerfbusters.utils.utils import get_gaussian_kernel1d
from nerfbusters.utils.visualizations import (
    visualize_grid2d,
    visualize_grid3d,
    visualize_grid3d_slices,
)
from torch.optim.lr_scheduler import ExponentialLR


def format_batch(batch):

    x = batch["input"]
    scale = batch["scale"]

    return x, scale


class NerfbustersTrainer(pl.LightningModule):
    def __init__(self, config, savepath=""):
        super().__init__()

        self.config = config
        self.savepath = savepath

        self.save_images_locally = config.get("save_images_locally", False)
        self.val_batch_size = config.get("val_batch_size", 32)

        # size of shapenet cubes
        self.data_size = (1, 32, 32, 32)
        
        self.model = get_model(self.config)
        # self.model.convert_to_fp16()

        if config.noise_scheduler == "ddim":
            num_train_timesteps = config.get("num_train_timesteps", 1000)
            num_inference_steps = config.get("num_inference_steps", 50)
            self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
            self.noise_scheduler.num_inference_steps = num_inference_steps
        elif config.noise_scheduler == "ddpm":
            num_train_timesteps = config.get("num_train_timesteps", 1000)
            num_inference_steps = config.get("num_inference_steps", 1000)
            beta_start = config.get("beta_start", 0.0015)
            beta_end = config.get("beta_end", 0.05)
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps, beta_start=beta_start, beta_end=beta_end
            )
            self.noise_scheduler.num_inference_steps = num_inference_steps
        else:
            raise ValueError(f"Unknown noise scheduler: {config.noise_scheduler}")

        # Create EMA for the model.
        if config.get("use_ema", True):
            self.ema_model = EMAModel(
                self.model.parameters(),
                use_ema_warmup=True,
            )

        self.loss_fn = nn.MSELoss()
        self.dsds_loss = DSDSLoss(
            self.noise_scheduler,
            diffusion_loss=config.get("diffusion_loss", "loss_dsds_unguided"),
            guidance_weight=config.get("guidance_weight", 1),
            anneal=config.get("anneal", "random"),
        )

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):

        x, scale = format_batch(batch)
        assert x.min() >= -1 and x.max() <= 1, f"{x.min()} {x.max()}"

        bs = x.shape[0]
        noise = torch.randn_like(x, device=x.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), dtype=torch.long, device=x.device)

        # add low frequency noise (extra data augmentation)
        with torch.no_grad():
            sigma = torch.rand((1,)) * 7 + 0.1  # 0.1 to 5.1
            k = get_gaussian_kernel1d(31, sigma)
            k3d = torch.einsum("i,j,k->ijk", k, k, k)
            k3d = k3d / k3d.sum()
            k3d = k3d.to(x.device)
            x_smooth = F.conv3d(x, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
            low_freq_noise = x - x_smooth

            noise = noise + low_freq_noise

        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

        noise_pred = self.model(noisy_x, timesteps, scale=scale).sample

        loss = self.loss_fn(noise_pred, noise)
        self.log("Train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if hasattr(self, "ema_model"):
            self.ema_model.step(self.model.parameters())
            self.log("Train/ema_decay", self.ema_model.decay, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):

        corrupted_x, x, scale = batch["corrupted_input"], batch["input"], batch["scale"]
        num_inference_steps = None
        num_sds_steps = 600

        # if it is a sanity check, then speed this up!
        if self.trainer.sanity_checking:
            if batch_idx == 0:
                # make the batch small
                corrupted_x = corrupted_x[:2]
                scale = scale[:2]
                x = x[:2]
                num_inference_steps = 10
                num_sds_steps = 10

            else:
                return

        if hasattr(self, "ema_model"):
            self.ema_model.copy_to(self.model.parameters())

        assert corrupted_x.min() >= -1 and corrupted_x.max() <= 1, f"{corrupted_x.min()} {corrupted_x.max()}"
        assert x.min() >= -1 and x.max() <= 1, f"{x.min()} {x.max()}"

        # print("Running generation with the reverse process...")
        t = time.time()
        generated = self.reverse_process(bs=x.shape[0], scale=None, num_inference_steps=num_inference_steps)
        generated_time = time.time() - t

        # print("Running decorruption with the reverse process...")
        t = time.time()
        decorrupted = self.reverse_process(sample=corrupted_x, scale=scale, num_inference_steps=num_inference_steps)
        decorrupted_time = time.time() - t

        # print("Running decorruption with DSDS optimization...")
        torch.set_grad_enabled(True)
        t = time.time()
        decorrupted_opt, _ = self.dsds_loss.optimize_patch(
            corrupted_x, self, gt=x, num_steps=num_sds_steps, scale=scale
        )
        decorrupted_opt_time = time.time() - t
        torch.set_grad_enabled(False)

        self.log("Timing/decorrupted_opt_time", decorrupted_opt_time)
        self.log("Timing/generated_time", generated_time)
        self.log("Timing/decorrupted_time", decorrupted_time)

        for k, v in [
            ("original", x),
            ("generated", generated),
            ("corrupted_x", corrupted_x),
            ("decorrupted", decorrupted),
            ("dsds", decorrupted_opt),
        ]:  
            
            # 3D if 5 dimensions
            assert len(v.shape) == 5

            # 3D metrics
            iou = metrics.voxel_iou(pred=v, gt=x)
            acc = metrics.voxel_acc(pred=v, gt=x)
            f1 = metrics.voxel_f1(pred=v, gt=x)
            self.log(f"Val_iou/{k}", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"Val_acc/{k}", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"Val_f1/{k}", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            # 3D visualization
            visualize_grid3d(
                f"{self.savepath}/{k}",
                v[:4],
                working_dir=self.savepath,
                save_locally=self.save_images_locally,
                num_views=4,
            )

            # 2D visualization of slices
            visualize_grid3d_slices(
                f"{self.savepath}/{k}",
                v[:4],
                save_locally=self.save_images_locally,
            )

    def test_step(self, batch, batch_idx):

        if hasattr(self, "use_ema"):
            self.ema_model.copy_to(self.model.parameters())

        x, scale = format_batch(batch)

        # evaluate sds denoising for different noise levels
        noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for noise_level in noise_levels:
            print(f"noise level: {noise_level}")

            corrupted_crops = noise_level * torch.randn_like(x) + (1 - noise_level) * x
            corrupted_crops = torch.clamp(corrupted_crops, -1.0, 1.0)

            assert (
                corrupted_crops.min() >= -1 and corrupted_crops.max() <= 1
            ), f"{corrupted_crops.min()} {corrupted_crops.max()}"

            optimized_crops, out = self.dsds_loss.optimize_patch(
                corrupted_crops, self.model, gt=x, num_steps=600, scale=scale
            )

            for key in out:
                data = [[x, y] for (x, y) in enumerate(out[key])]
                table = wandb.Table(data=data, columns=["x", "y"])
                wandb.log(
                    {f"noise_lvl_{noise_level}": wandb.plot.line(table, "x", "y", title=f"noise lvl: {noise_level}")},
                    step=self.global_step,
                )

    @torch.no_grad()
    def reverse_process(self, sample=None, scale=None, bs=None, num_inference_steps=None, starting_t=1000):
        """Run the reverse process of the model, either starting from a sample or from random noise.

        Args:
            sample (torch.Tensor): The samples to denoise.
            bs (int): The number of samples to generate.
        """

        self.noise_scheduler.betas = self.noise_scheduler.betas.to(self.device)
        self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        if sample is not None:
            pass
        else:
            shape = [bs] + list(self.data_size)
            sample = torch.randn(*shape, device=self.device)

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_inference_steps

        step_size = starting_t // num_inference_steps
        timesteps = torch.arange(starting_t - 1, 0, -step_size).to(self.device)
        for _, t in enumerate(timesteps):

            # 1. predict noise model_output
            bs = sample.shape[0]
            t = torch.tensor([t], dtype=torch.long, device=self.device)

            noise_pred = self.model(sample, t, scale=scale).sample
            
            # 2. compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample

        return sample

    @torch.no_grad()
    def single_step_reverse_process(self, sample, starting_t, scale=None):

        x = sample.clone()

        with torch.no_grad():
            t = torch.tensor([starting_t], dtype=torch.long, device=x.device)

            # 1. predict noise model_output
            noise_pred = self.model(x, t, scale=scale).sample

            # 2. compute previous image: x_t -> x_0
            x = self.noise_scheduler.step(noise_pred, t, sample).pred_original_sample

        w = self.noise_scheduler.alphas[t] ** 2

        return x, w

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = ExponentialLR(optimizer, gamma=0.99)

        return [optimizer], [scheduler]
