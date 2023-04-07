from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers.utils import BaseOutput
from torch import nn

from .nn import checkpoint, normalization, timestep_embedding, zero_module


@dataclass
class Output(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels

        x = F.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2), mode="nearest")

        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = (2, 2, 2)
        if use_conv:
            self.op = nn.Conv3d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(3, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


def contract_block(in_channels, out_channels, kernel_size, padding):

    contract = nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        normalization(out_channels),
        torch.nn.ReLU(),
        torch.nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        normalization(out_channels),
        torch.nn.ReLU(),
        Downsample(out_channels, use_conv=True, out_channels=out_channels),
    )

    return contract


def expand_block(in_channels, out_channels, kernel_size, padding):

    expand = nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
        normalization(out_channels),
        torch.nn.ReLU(),
        torch.nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
        normalization(out_channels),
        torch.nn.ReLU(),
        Upsample(out_channels, use_conv=True, out_channels=out_channels),
    )
    return expand


class TinyUNET(nn.Module):
    def __init__(self, in_channels, time_encodings=32):
        super().__init__()

        self.time_encodings = 32
        time_embed_dim = 32

        self.conv1 = contract_block(in_channels, 32, 3, 1)
        self.conv2 = contract_block(32, 32, 3, 1)
        self.conv3 = contract_block(32, 32, 3, 1)

        self.upconv3 = expand_block(32, 32, 3, 1)
        self.upconv2 = expand_block(32, 32, 3, 1)
        self.upconv1 = expand_block(32, 32, 3, 1)

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_encodings, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.proj1 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 32))
        self.proj2 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 32))
        self.proj3 = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 32))

        self.last_layer = torch.nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)

    def __call__(self, x, timesteps):

        emb = self.time_embed(timestep_embedding(timesteps, self.time_encodings))

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        time_embed = self.proj3(emb)
        conv3 = conv3 + time_embed[:, :, None, None, None]
        upconv3 = self.upconv3(conv3)

        time_embed = self.proj2(emb)
        upconv3 = upconv3 + time_embed[:, :, None, None, None]
        upconv2 = self.upconv2(upconv3 + conv2)

        time_embed = self.proj1(emb)
        upconv2 = upconv2 + time_embed[:, :, None, None, None]
        upconv1 = self.upconv1(upconv2 + conv1)

        sample = self.last_layer(upconv1)

        return Output(sample=sample)
