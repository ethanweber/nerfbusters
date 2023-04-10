"""
Benchmarking script for nerfstudio paper.
- nerfacto and instant-ngp methods on mipnerf360 data
- nerfacto ablations
"""

import copy
import glob
import json
import os
import random
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import GPUtil
import mediapy as media
import numpy as np
import torch
import tyro

from cleanerf.nerf.experiment_configs.experiments_ablations import arguments_list_of_lists as experiments_ablations_list
from cleanerf.nerf.experiment_configs.experiments_baseline import arguments_list_of_lists as experiments_baselines_list
from cleanerf.nerf.experiment_configs.experiments_visibility import (
    arguments_list_of_lists as experiments_visibility_list,
)
from cleanerf.nerf.experiment_configs.experiments_wild import arguments_list_of_lists as experiments_wild_list


from cleanerf.nerf.experiment_configs.utils import (
    get_experiment_name_and_argument_combinations,
)
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
from typing_extensions import Annotated

from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.utils.scripts import run_command
from nerfstudio.viewer.server.subprocess import get_free_port


def get_free_ports(num: int) -> List[int]:
    """Fund num free ports."""
    ports = set()
    while len(ports) < num:
        port = get_free_port()
        if port not in ports:
            ports.add(port)
    return list(ports)


def launch_experiments(jobs, dry_run: bool = False, gpu_ids: Optional[List[int]] = None):
    """Launch the experiments.
    Args:
        jobs: list of commands to run
        dry_run: if True, don't actually run the commands
        gpu_ids: list of gpu ids that we can use. If none, we can use any
    """

    num_jobs = len(jobs)
    while jobs:
        # get GPUs that capacity to run these jobs
        gpu_devices_available = GPUtil.getAvailable(order="first", limit=10, maxMemory=0.1)
        print("-" * 80)
        print("Available GPUs: ", gpu_devices_available)
        if gpu_ids:
            print("Restricting to subset of GPUs: ", gpu_ids)
            gpu_devices_available = [gpu for gpu in gpu_devices_available if gpu in gpu_ids]
            print("Using GPUs: ", gpu_devices_available)
        print("-" * 80)

        if len(gpu_devices_available) == 0:
            print("No GPUs available, waiting 10 seconds...")
            time.sleep(10)
            continue

        # thread list
        threads = []
        while gpu_devices_available and jobs:
            gpu = gpu_devices_available.pop(0)
            command = f"CUDA_VISIBLE_DEVICES={gpu} " + jobs.pop(0)

            def task():
                print(f"Command:\n{command}\n")
                if not dry_run:
                    _ = run_command(command, verbose=False)
                # print("Finished command:\n", command)

            threads.append(threading.Thread(target=task))
            threads[-1].start()

            # NOTE(ethan): here we need a delay, otherwise the wandb/tensorboard naming is messed up... not sure why
            if not dry_run:
                time.sleep(5)

        # wait for all threads to finish
        for t in threads:
            t.join()

        # print("Finished all threads")
    print(f"Finished all {num_jobs} jobs")


@dataclass
class ExperimentConfig(PrintableConfig):
    """Experiment config code."""

    dry_run: bool = False
    output_folder: Optional[Path] = None
    gpu_ids: Optional[List[int]] = None

    def main(self, dry_run: bool = False) -> None:
        """Run the code."""
        raise NotImplementedError


@dataclass
class Train(ExperimentConfig):
    """Train cleanerf models."""

    baseline_experiment_name: Optional[Path] = None
    nerf_checkpoint_path: Optional[Path] = None
    data: Optional[Path] = None

    def main(self, dry_run: bool = False):
        jobs = []
        experiment_names = []
        argument_combinations = []

        # For our experiments in the paper...
        experiment_names_ab, argument_combinations_ab = get_experiment_name_and_argument_combinations(
            experiments_visibility_list
        )
        experiment_names_ba, argument_combinations_ba = get_experiment_name_and_argument_combinations(
            experiments_baselines_list
        )
        experiment_names = experiment_names_ab + experiment_names_ba
        argument_combinations = argument_combinations_ab + argument_combinations_ba

        # For the ablations experiments in the paper...
        # experiment_names, argument_combinations = get_experiment_name_and_argument_combinations(
        #     experiments_ablations_list
        # )

        # For fun with nerfstudio wild captures...
        # experiment_names, argument_combinations = get_experiment_name_and_argument_combinations(experiments_wild_list)

        num_jobs = len(experiment_names)
        websocket_ports = get_free_ports(num_jobs)
        for experiment_name, argument_string, websocket_port in zip(
            experiment_names, argument_combinations, websocket_ports
        ):
            # base_cmd = f"ns-train magic-eraser --experiment-name {experiment_name} --vis viewer+wandb --output-dir {self.output_folder}"
            base_cmd = f"ns-train magic-eraser --experiment-name {experiment_name} --vis wandb"
            if self.output_folder is not None:
                base_cmd += f" --output-dir {self.output_folder}"
            if self.data is not None:
                base_cmd += f" --data {self.data}"
            if self.baseline_experiment_name is not None:
                glob_str = str(
                    self.output_folder / self.baseline_experiment_name / "magic-eraser/*/nerfstudio_models/*"
                )
                nerf_checkpoint_path = sorted(glob.glob(glob_str))[-1]
                # TODO: throw some sort of assert here if needed
                base_cmd += f" --pipeline.nerf_checkpoint_path {nerf_checkpoint_path}"
            # jobs.append(
            #     f" {base_cmd} {argument_string} --viewer.websocket-port {websocket_port} --viewer.quit_on_train_completion True"
            # )
            jobs.append(f" {base_cmd} {argument_string}")

        # randomize the order of the jobs
        # random.shuffle(jobs)

        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


@dataclass
class Render(ExperimentConfig):
    """Render cleanerf models."""

    input_folder: Path = Path("outputs")
    rendered_output_names: List[str] = field(
        default_factory=lambda: ["rgb", "depth", "normals", "visibility_median_count"]
    )
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = 0.0
    """Specifies the near plane for depth rendering."""
    depth_far_plane: Optional[float] = 2.0
    """Specifies the far plane for depth rendering."""
    downscale_factor: Optional[int] = 1

    def main(self, dry_run: bool = False):
        # get all the experiment names and configs from the input folder
        jobs = []
        experiment_names = os.listdir(self.input_folder)
        for experiment_name in experiment_names:
            print(experiment_name)
            globbed = glob.glob(str(self.input_folder / experiment_name / "*/*/config.yml"))
            if len(globbed) == 0:
                continue
            config_filename = sorted(globbed)[-1]
            output_path = self.output_folder / Path(str(experiment_name) + ".mp4")
            rendered_output_names = " ".join(self.rendered_output_names)
            job = f"ns-render --output-format images --load-config {config_filename} --traj eval-images --output-path {output_path} --rendered_output_names {rendered_output_names}"
            # optional args
            if self.depth_near_plane is not None:
                job += f" --depth-near-plane {self.depth_near_plane}"
            if self.depth_far_plane is not None:
                job += f" --depth-far-plane {self.depth_far_plane}"
            if self.downscale_factor is not None:
                job += f" --downscale-factor {self.downscale_factor}"
            jobs.append(job)

        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


@dataclass
class Metrics(ExperimentConfig):
    """Render cleanerf models."""

    input_folder: Path = Path("outputs-renders")
    visibility_experiment_name: Path = Path("visibility_experiment_name")
    device: str = "cuda:0"
    max_depth: float = 2
    min_views: int = 1

    def main(self, dry_run: bool = False):

        # TODO: move this code elsewhere so we can spawn multiple jobs here

        print("Using visibility masks from experiment: ", self.visibility_experiment_name)

        psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none", return_full_image=True).to(
            self.device
        )
        lpips_module = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        mse_module = MSELoss()

        # go through all the experiments
        experiment_names = os.listdir(self.input_folder)
        print(experiment_names)
        for experiment_name in experiment_names:
            rgb_gt_filenames = sorted(glob.glob(str(self.input_folder / experiment_name / "rgb_gt" / "*")))
            visibility_filenames = sorted(
                glob.glob(str(self.input_folder / self.visibility_experiment_name / "visibility_median_count" / "*"))
            )
            metrics = defaultdict(list)
            video = []  # images to make a video
            print(f"Processing experiment: {experiment_name} ...")
            if len(rgb_gt_filenames) == 0:
                print("No rgb_gt images found, skipping experiment")
                continue
            for idx, rgb_gt_filename in enumerate(rgb_gt_filenames):
                rgb_filename = rgb_gt_filename.replace("rgb_gt", "rgb")
                normals_filename = rgb_gt_filename.replace("rgb_gt", "normals")
                depth_filename = rgb_gt_filename.replace("rgb_gt", "depth")
                depth_raw_filename = rgb_gt_filename.replace("rgb_gt", "depth_raw").replace(".png", ".npy")
                rgb_gt = media.read_image(rgb_gt_filename)
                pseudo_gt_visibility = media.read_image(visibility_filenames[idx])
                rgb = media.read_image(rgb_filename)
                normals = media.read_image(normals_filename)
                depth = media.read_image(depth_filename)
                depth_raw = np.load(depth_raw_filename)[..., 0]
                pseudo_gt_depth_raw_filename = (
                    visibility_filenames[idx].replace("visibility_median_count", "depth_raw").replace(".png", ".npy")
                )
                psuedo_gt_depth_raw = np.load(pseudo_gt_depth_raw_filename)[..., 0]
                psuedo_gt_normals_filename = visibility_filenames[idx].replace("visibility_median_count", "normals")
                psuedo_gt_normals = media.read_image(psuedo_gt_normals_filename)
                assert len(depth_raw.shape) == 2
                assert len(psuedo_gt_depth_raw.shape) == 2

                # move to torch and device
                rgb_gt = torch.from_numpy(rgb_gt).float().to(self.device) / 255.0  # (H, W, 3)
                rgb = torch.from_numpy(rgb).float().to(self.device) / 255.0
                pseudo_gt_visibility = (
                    torch.from_numpy(pseudo_gt_visibility).float().to(self.device)
                )  # has integer values
                psuedo_gt_depth_raw = torch.from_numpy(psuedo_gt_depth_raw).float().to(self.device)
                depth_raw = torch.from_numpy(depth_raw).float().to(self.device)
                depth = torch.from_numpy(depth).float().to(self.device) / 255.0  # 'depth' is a colormap
                psuedo_gt_normals = torch.from_numpy(psuedo_gt_normals).float().to(self.device) / 255.0
                psuedo_gt_normals_raw = psuedo_gt_normals * 2.0 - 1.0
                normals = torch.from_numpy(normals).float().to(self.device) / 255.0
                normals_raw = normals * 2.0 - 1.0

                # create masks
                visibilty_mask = (pseudo_gt_visibility[..., 0] >= self.min_views).float()
                depth_mask = (depth_raw < self.max_depth).float()
                mask = visibilty_mask * depth_mask  # (H, W)
                # make mask 3 channels
                mask = mask[..., None].repeat(1, 1, 3)

                # show masked versions
                rgb_gt_masked = rgb_gt * mask
                rgb_masked = rgb * mask
                depth_masked = depth * mask
                normals_masked = normals * mask

                # compute psnr
                psnr = float(psnr_module(rgb[mask == 1].view(-1, 3), rgb_gt[mask == 1].view(-1, 3)))

                # compute ssim
                ssim_value, ssim_image = ssim_module(
                    preds=rgb_masked.unsqueeze(0).permute(0, 3, 1, 2),
                    target=rgb_gt_masked.unsqueeze(0).permute(0, 3, 1, 2),
                )
                ssim = float(ssim_image[0].permute(1, 2, 0)[mask[..., 0] == 1].mean())

                # compute lpips
                lpips_mask = mask.unsqueeze(0).permute(0, 3, 1, 2)[0, 0]  # (H, W)
                lpips = float(
                    lpips_module(
                        rgb_masked.unsqueeze(0).permute(0, 3, 1, 2),
                        rgb_gt_masked.unsqueeze(0).permute(0, 3, 1, 2),
                        lpips_mask,
                    )
                )
                # print(lpips)

                # depth
                depth_mask = mask[..., 0] == 1
                depth_mse = float(mse_module(depth_raw[depth_mask], psuedo_gt_depth_raw[depth_mask]))

                # disparity
                disparity_raw = 1.0 / depth_raw
                psuedo_gt_disparity_raw = 1.0 / psuedo_gt_depth_raw
                disparity = float(torch.abs(disparity_raw[depth_mask] - psuedo_gt_disparity_raw[depth_mask]).mean())

                # make sure the normals raw are normalized
                normals_raw = normals_raw / torch.norm(normals_raw, dim=-1, keepdim=True)
                psuedo_gt_normals_raw = psuedo_gt_normals_raw / torch.norm(psuedo_gt_normals_raw, dim=-1, keepdim=True)
                eps = 1e-8
                costheta = (
                    (normals_raw[mask == 1].view(-1, 3) * psuedo_gt_normals_raw[mask == 1].view(-1, 3))
                    .sum(dim=-1)
                    .clamp(-1 + eps, 1 - eps)
                )
                theta = torch.abs(torch.acos(costheta) * 180.0 / np.pi)
                normals_mse = float(theta.mean())
                normals_median = float(theta.median())

                # the angle thresholds
                metrics["normals_11.25"].append(float((theta < 11.25).float().mean()))
                metrics["normals_22.5"].append(float((theta < 22.5).float().mean()))
                metrics["normals_30"].append(float((theta < 30).float().mean()))

                # coverage
                coverage = float(mask[..., 0].sum() / visibilty_mask.sum())

                metrics["psnr_list"].append(psnr)
                metrics["ssim_list"].append(ssim)
                metrics["lpips_list"].append(lpips)
                metrics["depth_list"].append(depth_mse)
                metrics["disparity_list"].append(disparity)
                metrics["normals_list"].append(normals_mse)
                metrics["normals_median_list"].append(normals_median)
                metrics["coverage_list"].append(coverage)

                # TODO: compute the metrics for masked versions

                rgb_gt = (rgb_gt * 255.0).cpu().numpy().astype(np.uint8)
                pseudo_gt_visibility = (pseudo_gt_visibility).cpu().numpy().astype(np.uint8)
                rgb_gt_masked = (rgb_gt_masked * 255.0).cpu().numpy().astype(np.uint8)
                rgb_masked = (rgb_masked * 255.0).cpu().numpy().astype(np.uint8)
                depth_masked = (depth_masked * 255.0).cpu().numpy().astype(np.uint8)
                normals_masked = (normals_masked * 255.0).cpu().numpy().astype(np.uint8)
                image = np.concatenate(
                    [rgb_gt, pseudo_gt_visibility, rgb_gt_masked, rgb_masked, depth_masked, normals_masked], axis=1
                )
                image_filename = self.input_folder / experiment_name / "composited" / f"{idx:04d}.png"
                image_filename.parent.mkdir(parents=True, exist_ok=True)
                media.write_image(image_filename, image)

                video.append(image)

            # write out the video
            video_filename = self.input_folder / experiment_name / f"{experiment_name}.mp4"
            media.write_video(video_filename, video, fps=30)

            # convert metrics dict to a proper dictionary
            metrics = dict(metrics)
            metrics["psnr"] = np.mean(metrics["psnr_list"])
            metrics["ssim"] = np.mean(metrics["ssim_list"])
            metrics["lpips"] = np.mean(metrics["lpips_list"])
            metrics["depth"] = np.mean(metrics["depth_list"])
            metrics["disparity"] = np.mean(metrics["disparity_list"])
            metrics["normals"] = np.mean(metrics["normals_list"])
            metrics["normals_median"] = np.mean(metrics["normals_median_list"])
            metrics["coverage"] = np.mean(metrics["coverage_list"])
            metrics["normals_11.25"] = np.mean(metrics["normals_11.25"])
            metrics["normals_22.5"] = np.mean(metrics["normals_22.5"])
            metrics["normals_30"] = np.mean(metrics["normals_30"])
            for metric_name in sorted(metrics.keys()):
                if "_list" not in metric_name:
                    print(f"{metric_name}: {metrics[metric_name]}")

            # write to a json file
            metrics_filename = self.input_folder / experiment_name / f"{experiment_name}.json"
            with open(metrics_filename, "w") as f:
                json.dump(metrics, f, indent=4)


@dataclass
class MetricsAll(ExperimentConfig):
    """Run all the metrics."""

    def main(self, dry_run: bool = False):
        jobs = []
        # TODO: clean this up
        DATASETS = [
            "flowers",
            "pipe",
            "roses",
            "pikachu",
            "car",
            "aloe",
            "picnic",
            "table",
            "century",
            "art",
            "plant",
            "garbage",
        ]
        for dataset in DATASETS:
            job = f"python projects/magic_eraser/scripts/launch_nerf.py metrics --input-folder /shared/ethanweber/magic-eraser/nerf-renders/{dataset} --visibility-experiment-name {dataset}---nerfacto---visibility-split"
            jobs.append(job)
        launch_experiments(jobs, dry_run=dry_run, gpu_ids=self.gpu_ids)


Commands = Union[
    Annotated[Train, tyro.conf.subcommand(name="train")],
    Annotated[Render, tyro.conf.subcommand(name="render")],
    Annotated[Metrics, tyro.conf.subcommand(name="metrics")],
    Annotated[MetricsAll, tyro.conf.subcommand(name="metrics-all")],
]


def main(
    benchmark: ExperimentConfig,
):
    """Script to run the benchmark experiments for the Nerfstudio paper.
    - nerfacto-on-mipnerf360: The MipNeRF-360 experiments on the MipNeRF-360 Dataset.
    - nerfacto-ablations: The Nerfacto ablations on the Nerfstudio Dataset.
    Args:
        benchmark: The benchmark to run.
    """
    benchmark.main(dry_run=benchmark.dry_run)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
