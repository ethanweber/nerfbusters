"""
Downloads the CleaNeRF dataset.
We download the videos and the already processed results from running COLMAP.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gdown
import tyro
from rich.console import Console
from typing_extensions import Annotated

from nerfstudio.configs.base_config import PrintableConfig
from nerfstudio.utils import install_checks
from nerfstudio.utils.scripts import run_command

CONSOLE = Console(width=120)


@dataclass
class DatasetDownload(PrintableConfig):
    """Download a dataset"""

    capture_name = None

    save_dir: Path = Path("data/")
    """The directory to save the dataset to"""

    def download(self, save_dir: Path) -> None:
        """Download the dataset"""
        raise NotImplementedError


@dataclass
class CleaNeRFDiffusionCubeWeightsDownload(DatasetDownload):
    """Download the CleaNeRF captures."""

    def download(self, save_dir: Path):
        # Download the files
        url = "https://data.nerf.studio/cleanerf-diffusion-cube-weights.ckpt"
        download_path = str(save_dir / "cleanerf-diffusion-cube-weights.ckpt")
        gdown.download(url, output=download_path)


@dataclass
class CleaNeRFCapturesDownload(DatasetDownload):
    """Download the CleaNeRF captures."""

    def download(self, save_dir: Path):

        # Download the files
        url = "https://data.nerf.studio/cleanerf-captures.zip"
        download_path = str(save_dir / "cleanerf-captures.zip")
        gdown.download(url, output=download_path)
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        os.remove(download_path)


@dataclass
class CleaNeRFDataDownload(DatasetDownload):
    """Download the CleaNeRF dataset."""

    def download(self, save_dir: Path):

        # Download the files
        url = "https://data.nerf.studio/cleanerf-dataset.zip"
        download_path = str(save_dir / "cleanerf-dataset.zip")
        gdown.download(url, output=download_path)
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        os.remove(download_path)

def main(
    dataset: DatasetDownload,
):
    """Script to download the CleaNeRF data.
    - captures: These are the videos which were used the ns-process-data.
    - data: These are the already-processed results from running ns-process-data (ie COLMAP).
    Args:
        dataset: The dataset to download (from).
    """
    dataset.save_dir.mkdir(parents=True, exist_ok=True)

    dataset.download(dataset.save_dir)


Commands = Union[
    Annotated[CleaNeRFDiffusionCubeWeightsDownload, tyro.conf.subcommand(name="diffusion-cube-weights")],
    Annotated[CleaNeRFCapturesDownload, tyro.conf.subcommand(name="captures")],
    Annotated[CleaNeRFDataDownload, tyro.conf.subcommand(name="dataset")],
]


def entrypoint():
    """Entrypoint."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()
