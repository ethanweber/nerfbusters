import torch
import torch.nn as nn
import torch.nn.functional as F
from cleanerf.utils import metrics
from tqdm import tqdm


class SDSLoss(nn.Module):
    def __init__(self, noise_scheduler, sds_type="unguided", guidance_weight=1.0, anneal="random", dataset="cubes"):
        super().__init__()

        self.sds_type = sds_type
        assert sds_type in ["guided", "unguided", "temporal_guidance", "multistep_sds"]

        if sds_type == "guided":
            assert guidance_weight > 1

        self.guidance_weight = guidance_weight
        self.anneal = anneal
        self.dataset = dataset

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
        # scheduler = ExponentialLR(optimizer, gamma=0.9)

        # grads = []
        iou = []
        acc = []
        # for i in tqdm(range(num_steps)):
        for i in range(num_steps):
            optimizer.zero_grad()
            if self.sds_type in ("multistep_sds", "reverse_sds"):
                loss = self.forward(x, model, timesteps[i].item(), scale, multistep_sds_steps)
                loss.backward()
            else:
                _ = self.forward(x, model, timesteps[i].item(), scale, multistep_sds_steps)
            optimizer.step()
            # scheduler.step()
            # grads.append(grad)

            if gt is not None and len(x.shape) == 5:
                iou.append(metrics.voxel_iou(pred=x, gt=gt))
                acc.append(metrics.voxel_acc(pred=x, gt=gt))

        output = {"iou": iou, "acc": acc}

        return x.detach(), output

    def forward(self, x, model, timesteps, scale, multistep_sds_steps, starting_t=999):

        if self.sds_type == "unguided":
            grad = self.grad_sds_unconditional(x, model, timesteps, scale)
        elif self.sds_type == "guided":
            grad = self.grad_sds_guided(x, model, timesteps, scale)
        elif self.sds_type == "temporal_guidance":
            grad = self.grad_sds_temporal_guidance(x, model, timesteps, scale)
        elif self.sds_type == "log_likelihood":
            grad = self.grad_log_likelihood(x, model, timesteps, scale)
        elif self.sds_type == "multistep_sds":
            loss = self.multistep_sds(x=x, model=model, timesteps=timesteps, scale=scale, n_steps=multistep_sds_steps)
            return loss
        elif self.sds_type == "reverse_sds":
            loss = self.reverse_sds(
                sample=x,
                model=model,
                starting_t=timesteps,
                scale=scale,
                n_steps=multistep_sds_steps,
            )
            return loss
        else:
            raise NotImplementedError
        return grad

    def grad_sds_temporal_guidance(self, x, model, t1, scale, guidance_scale=100, delta_t=100, mult=1.0):

        t2 = t1 + delta_t

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(x)
            noisy_x1 = self.noise_scheduler.add_noise(x, noise, t1)
            noisy_x2 = self.noise_scheduler.add_noise(x, noise, t2)

            # pred noise
            noise_pred1 = model(noisy_x1, t1, scale=scale).sample
            noise_pred2 = model(noisy_x2, t2, scale=scale).sample

            noise_pred = noise_pred1 + guidance_scale * (noise_pred1 - noise_pred2)
            # noise_pred = noise_pred2 + w * (noise_pred1 - noise_pred2)

        w = 1 - self.alphas[t1] ** 2
        # TODO: not sure if 1 - w is better
        grad = w * (noise_pred - noise)

        # grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        x.backward(gradient=grad * mult, retain_graph=True)

        return grad

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

        # w(t), sigma_t^2

        # TODO: maybe it should be 1 - alphas[t] ** 2
        w = 1 - self.alphas[timesteps] ** 2
        w = w.to(x.device)
        w = w.reshape(-1, *([1] * (len(x.shape) - 1)))
        grad = w * (noise_pred - noise)

        # grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        x.backward(gradient=grad * mult, retain_graph=True)

        return grad

    def grad_log_likelihood(self, x, model, timesteps, scale, mult=1.0):

        timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)

        with torch.no_grad():
            # pred noise
            if self.dataset == "cubes":
                noise_pred = model(x, timesteps, scale=scale).sample
            else:
                noise_pred = model(x, timesteps).sample

            noise_pred = noise_pred

        # w(t), sigma_t^2

        # TODO: maybe it should be 1 - alphas[t] ** 2
        w = 1 - self.alphas[timesteps] ** 2
        w = w.to(x.device)
        w = w.reshape(-1, *([1] * (len(x.shape) - 1)))
        grad = w * (noise_pred)

        # grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

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
            if self.dataset == "cubes":
                noise_pred = model(noisy_x, timesteps, scale=scale).sample
            else:
                noise_pred = model(noisy_x, timesteps).sample

            noise_pred = guidance_scale * noise_pred

        # w(t), sigma_t^2

        # TODO: maybe it should be 1 - alphas[t] ** 2
        # w = 1 - self.alphas[timesteps] ** 2
        w = self.alphas[timesteps] ** 2
        w = w.to(x.device)
        grad = w * (noise_pred - noise)

        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad = torch.clamp(grad, -10.0, 10.0)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        x.backward(gradient=grad * mult, retain_graph=True)

        return grad

    def multistep_sds(self, x, model, timesteps, scale, n_steps=10, mult=1.0):

        x_denoise = model.reverse_process(sample=x, scale=scale, num_inference_steps=n_steps, starting_t=timesteps)

        return mult * torch.mean((x_denoise - x) ** 2)

    def reverse_sds(self, sample, model, starting_t, scale, n_steps=10, mult=1.0):

        x, w = model.single_step_reverse_process(sample=sample, scale=scale, starting_t=starting_t)

        return mult * torch.mean((sample - x) ** 2) / (1 - w)
