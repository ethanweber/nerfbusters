from diffusers import UNet2DModel
from cleanerf.models.tiny_unet import TinyUNET

from .conv_permuter import ConvPermuter
from .mlp_mixer import MLPMixer
from .unet import UNetModel


def get_model(config):

    if config.dataset in ("cubes", "planes"):
        arch = config.get("architecture", "unet3d")
        if arch == "unet3d":
            print("Using 3D unet")
            model_channels = config.get("model_channels", 32)
            num_res_blocks = config.get("num_res_blocks", 2)
            channel_mult = tuple(config.get("channel_mult", (1, 2, 4, 8)))
            attention_resolutions = tuple(config.get("attention_resolutions", (16, 8)))
            dropout = config.get("dropout", 0.0)
            return UNetModel(
                image_size=32,
                in_channels=1,
                out_channels=1,
                model_channels=model_channels,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult,
                attention_resolutions=attention_resolutions,
                dropout=dropout,
                dims=3,
                condition_on_scale=config.get("condition_on_scale", False),
            )
        elif arch == "unet2d":
            print("Using 2D unet")
            model_channels = config.get("model_channels", 32)
            num_res_blocks = config.get("num_res_blocks", 2)
            channel_mult = tuple(config.get("channel_mult", (1, 2, 4, 8)))
            attention_resolutions = tuple(config.get("attention_resolutions", (16, 8)))
            return UNetModel(
                image_size=32,
                in_channels=1,
                out_channels=1,
                model_channels=model_channels,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult,
                attention_resolutions=attention_resolutions,
                dims=2,
            )
        elif arch == "mlpmixer":
            print("Using MLPMixer")
            return MLPMixer(32, depth=8)
        elif arch == "convpermuter":
            print("Using ConvPermuter")
            kernel_size = config.get("kernel_size", 1)
            depth = config.get("depth", 8)
            return ConvPermuter(32, depth=depth, kernel_size=kernel_size)
        elif arch == "tinyunet":
            return TinyUNET(1)

    elif config.dataset in ("mnist", "fashionmnist"):
        model = UNet2DModel(
            sample_size=28,  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )
    elif config.dataset == "flowers":
        model = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            # block_out_channels=(32, 64, 64, 64),
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    elif config.dataset == "hypersim":

        model = UNet2DModel(
            sample_size=48,  # the target image resolution
            in_channels=len(config.input),  # the number of input channels, 3 for RGB images
            out_channels=len(config.input),  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                # "DownBlock2D",
                # "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                # "UpBlock2D",
                # "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    else:
        model = UNet2DModel(
            sample_size=32,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(
                32,
                64,
                128,
                256,
            ),  # the number of output channes for each UNet block
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    return model
