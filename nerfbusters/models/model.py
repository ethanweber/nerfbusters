from .unet import UNetModel


def get_model(config):
    
    print("==> Using 3D unet")
    model_channels = config.get("model_channels", 32)
    num_res_blocks = config.get("num_res_blocks", 2)
    channel_mult = tuple(config.get("channel_mult", (1, 2, 4, 8)))
    attention_resolutions = tuple(config.get("attention_resolutions", (16, 8)))
    dropout = config.get("dropout", 0.0)

    model = UNetModel(
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

    return model
