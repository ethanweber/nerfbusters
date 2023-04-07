from abc import abstractmethod
from dataclasses import dataclass
from functools import partial

import torch
from diffusers.utils import BaseOutput
from torch import nn

from .nn import checkpoint, normalization, timestep_embedding, zero_module


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb_out):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = h + emb_out
        h = self.out_layers(h)

        return self.skip_connection(x) + h


@dataclass
class Output(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class Approx3dConvBlockTimeConditioned(nn.Module):
    def __init__(self, input_size, emb_channels, kernel_size=1):
        super().__init__()

        self.conv_block1 = Approx3dConvBlock(input_size, kernel_size)
        self.conv_block2 = Approx3dConvBlock(input_size, kernel_size)
        self.conv_block3 = Approx3dConvBlock(input_size, kernel_size)

        self.proj1 = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, input_size))
        self.proj2 = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, input_size))
        self.proj3 = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, input_size))

    def forward(self, x, emb):

        time_embed = self.proj1(emb)
        h = self.conv_block1(x, time_embed[:, None, None, :])

        # h = h + time_embed[:, None, None, :]
        # h = x + h

        time_embed = self.proj2(emb)
        h = self.conv_block2(h, time_embed[:, None, :, None])

        # h = h + time_embed[:, None, :, None]
        # h = x + h

        time_embed = self.proj3(emb)
        h = self.conv_block3(h, time_embed[:, :, None, None])

        # h = h + time_embed[:, :, None, None]
        # h = x + h

        return h


class Approx3dConvBlock(nn.Module):
    def __init__(self, input_size, kernel_size=1) -> None:
        super().__init__()

        self.resblock1 = ResBlock(input_size, input_size, dropout=0.0, out_channels=input_size)
        self.resblock2 = ResBlock(input_size, input_size, dropout=0.0, out_channels=input_size)
        self.resblock3 = ResBlock(input_size, input_size, dropout=0.0, out_channels=input_size)

        # self.act = nn.SiLU()

    def forward(self, x, emb):

        # x = self.act(x)
        x = self.resblock1(x, emb)  # [B, H, W, D]
        x = x.permute(0, 2, 3, 1)  # [B, W, D, H]

        # x = self.act(x)
        x = self.resblock2(x, emb)
        x = x.permute(0, 2, 3, 1)  # [B, D, H, W]

        # x = self.act(x)
        x = self.resblock3(x, emb)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, D]

        return x


class ConvPermuter(nn.Module):
    def __init__(self, cube_size, depth=8, time_embed_dim=32, time_encodings=64, kernel_size=1):
        super().__init__()

        self.time_encodings = time_encodings

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_encodings, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.conv_blocks = nn.ModuleList(
            [Approx3dConvBlockTimeConditioned(cube_size, time_embed_dim, kernel_size) for _ in range(depth)]
        )

        # self.conv = nn.Sequential(nn.Conv3d(1, 1, 3, 1, 1), nn.SiLU(), nn.Conv3d(1, 1, 3, 1, 1))

    def forward(self, x, timesteps):

        b, c, h, w, d = x.shape
        x = x.view(b, h, w, d)

        emb = self.time_embed(timestep_embedding(timesteps, self.time_encodings))

        for block in self.conv_blocks:
            x = block(x, emb)

        x = x.view(b, 1, h, w, d)

        # x = self.conv(x)

        return Output(sample=x)
