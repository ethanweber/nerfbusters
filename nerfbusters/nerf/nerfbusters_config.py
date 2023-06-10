"""
Define the Nerfbusters config.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig


from nerfbusters.nerf.nerfbusters_pipeline import NerfbustersPipelineConfig

nerfbusters_config = MethodSpecification(
    TrainerConfig(
        method_name="nerfbusters",
        project_name="nerfbusters-project",
        steps_per_eval_batch=1000,
        steps_per_eval_image=1000,
        steps_per_save=5000,
        steps_per_eval_all_images=0,
        save_only_latest_checkpoint=False,
        max_num_iterations=5001,
        mixed_precision=True,
        pipeline=NerfbustersPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(eval_mode="eval-frame-index"),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15, predict_normals=True, depth_method="median"),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=None),
        vis="viewer",
    ),
    description="Uses the Nerfbusters pipeline."
)