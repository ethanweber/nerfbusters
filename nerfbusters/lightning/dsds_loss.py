import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfbusters.utils import metrics
from tqdm import tqdm
import math


class DSDSLoss(nn.Module):
    def __init__(self, noise_scheduler, diffusion_loss="loss_dsds_unguided", guidance_weight=1.0, anneal="random"):
        super().__init__()

        self.diffusion_loss = diffusion_loss
        assert diffusion_loss in ["grad_sds_guided", "grad_sds_unguided", "loss_dsds_unguided"]

        if diffusion_loss == "guided":
            assert guidance_weight > 1

        self.guidance_weight = guidance_weight
        self.anneal = anneal

        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.noise_scheduler.alphas_cumprod

    def get_timesteps(self, num_steps):
        if self.anneal == "linear":
            # linearly decreasing timesteps
            timesteps = torch.linspace(self.max_step, self.min_step, num_steps).long()
        elif self.anneal == "linear_cosine":
            # based on this paper (B.1)
            # https://arxiv.org/pdf/2206.09012.pdf
            a, b = 20, 0.25
            timesteps = torch.linspace(self.max_step, self.min_step, num_steps)
            timesteps = torch.cos(timesteps / self.max_step * a) * timesteps * b + timesteps
            timesteps = torch.clamp(timesteps, 0, self.max_step).long()
        else:
            # random timesteps
            timesteps = torch.randint(self.min_step, self.max_step, (num_steps,)).long()
        return timesteps

    def optimize_patch(self, x_corrupted, model, scale=None, gt=None, lr=0.01, num_steps=100, multistep_sds_steps=1):

        x = x_corrupted.clone()

        timesteps = self.get_timesteps(num_steps)
        timesteps = timesteps.to(x.device)

        # optimize patch
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=lr)

        iou = []
        acc = []
        for i in range(num_steps):
            optimizer.zero_grad()
            if "loss" in self.diffusion_loss:
                loss = self.forward(x, model, timesteps[i].item(), scale, multistep_sds_steps)
                loss.backward()
            elif "grad" in self.diffusion_loss:
                _ = self.forward(x, model.model, timesteps[i].item(), scale, multistep_sds_steps)
            optimizer.step()

            if gt is not None and len(x.shape) == 5:
                iou.append(metrics.voxel_iou(pred=x, gt=gt))
                acc.append(metrics.voxel_acc(pred=x, gt=gt))

        output = {"iou": iou, "acc": acc}

        return x.detach(), output

    def forward(self, x, model, timesteps, scale, multistep_sds_steps):

        if self.diffusion_loss == "grad_sds_unguided":
            out = self.grad_sds_unconditional(x, model, timesteps, scale)
        elif self.diffusion_loss == "grad_sds_guided":
            out = self.grad_sds_guided(x, model, timesteps, scale)
        elif self.diffusion_loss == "loss_dsds_unguided":
            out = self.loss_dsds(x, model, timesteps, scale)
        else:
            raise NotImplementedError
        return out


    def grad_sds_guided(self, x, model, timesteps, scale, guidance_scale=10.0, mult=1.0):

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(x)
            noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

            # pred noise
            noise_pred_cond = model(noisy_x, timesteps, scale=scale).sample
            noise_pred_uncond = model(noisy_x, timesteps).sample

            noise_pred = noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            noise_pred = guidance_scale * noise_pred

        w = 1 - self.alphas[timesteps] ** 2
        w = w.to(x.device)
        w = w.reshape(-1, *([1] * (len(x.shape) - 1)))
        grad = w * (noise_pred - noise)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        x.backward(gradient=grad * mult, retain_graph=True)

        return grad

    def grad_sds_unconditional(self, x, model, timesteps, scale, guidance_scale=1.0, mult=1.0):

        timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(x)
            noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

            # pred noise
            noise_pred = model(noisy_x, timesteps, scale=scale).sample

            noise_pred = guidance_scale * noise_pred

        w = 1 - self.alphas[timesteps] ** 2
        w = w.to(x.device)
        w = w.reshape(-1, *([1] * (len(x.shape) - 1)))
        grad = w * (noise_pred - noise)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        x.backward(gradient=grad * mult, retain_graph=True)

        return grad

    def loss_dsds(self, density, model, starting_t, scale, mult=1.0, singlestep_target=1.0):

        with torch.no_grad():
            xhat, w = model.single_step_reverse_process(sample=density, scale=scale, starting_t=starting_t)

        xhat = torch.where(xhat < 0, -1, 1)
        mask_empty = xhat == -1
        mask_full = xhat == 1
        density = density.unsqueeze(1)
        loss = (density * mask_empty).sum()
        loss += (torch.clamp(singlestep_target - density, 0) * mask_full).sum()
        loss = loss / math.prod(density.shape)  # average loss

        return loss