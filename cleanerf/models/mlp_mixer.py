from abc import abstractmethod
from dataclasses import dataclass
from functools import partial

import torch
from diffusers.utils import BaseOutput
from torch import nn

from .nn import timestep_embedding


@dataclass
class Output(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class MixerTimestepBlock(TimestepBlock):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def __init__(self, emb_channels, out_channels, fn):
        super().__init__()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                out_channels,
            ),
        )

        # self.linear = nn.Sequential(
        #    nn.SiLU(),
        #    nn.Linear(
        #        input_size,
        #        input_size,
        #    ),
        # )

        self.fn = fn

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

        emb_out = self.emb_layers(emb)

        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]

        h1 = x + emb_out

        h1 = self.fn(h1)

        return x + h1


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


def FeedForward(dim, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(dense(dim, dim), nn.GELU(), nn.Dropout(dropout), dense(dim, dim), nn.Dropout(dropout))


class MLPMixer(nn.Module):
    def __init__(self, cube_size, depth=8, time_embed_dim=32, time_encodings=128, dropout=0.0):
        super().__init__()

        self.time_encodings = time_encodings

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_encodings, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.mixer = TimestepEmbedSequential(
            Reshape((cube_size, cube_size * cube_size)),
            *[
                TimestepEmbedSequential(
                    PreNormResidual(cube_size * cube_size, FeedForward(cube_size, dropout, chan_first)),
                    MixerTimestepBlock(
                        cube_size,
                        cube_size,
                        FeedForward(cube_size, dropout, chan_first),
                    ),
                    PreNormResidual(cube_size * cube_size, FeedForward(cube_size * cube_size, dropout, chan_last)),
                    MixerTimestepBlock(
                        cube_size,
                        cube_size,
                        FeedForward(cube_size * cube_size, dropout, chan_last),
                    ),
                )
                for _ in range(depth)
            ],
            Reshape((1, cube_size, cube_size, cube_size)),
            nn.Conv3d(1, 1, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(1, 1, 3, 1, 1),
        )

    def forward(self, x, timesteps):

        emb = self.time_embed(timestep_embedding(timesteps, self.time_encodings))

        return Output(sample=self.mixer(x, emb))
