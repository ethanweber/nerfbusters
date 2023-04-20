"""
Downloads the Nerfbusters dataset.
We download the videos and the already processed results from running COLMAP.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gdown
import tyro
from rich.console import Console
from typing_extensions import Annotated

from nerfstudio.configs.base_config import PrintableConfig

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
class NerfbustersDiffusionCubeWeightsDownload(DatasetDownload):
    """Download the Nerfbusters captures."""

    def download(self, save_dir: Path):
        # Download the files
        url = "https://data.nerf.studio/nerfbusters-diffusion-cube-weights.ckpt"
        download_path = str(save_dir / "nerfbusters-diffusion-cube-weights.ckpt")
        gdown.download(url, output=download_path)


@dataclass
class NerfbustersCapturesDownload(DatasetDownload):
    """Download the Nerfbusters captures."""

    def download(self, save_dir: Path):

        # Download the files
        url = "https://data.nerf.studio/nerfbusters-captures.zip"
        download_path = str(save_dir / "nerfbusters-captures.zip")
        gdown.download(url, output=download_path)
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        os.remove(download_path)


@dataclass
class NerfbustersDataDownload(DatasetDownload):
    """Download the Nerfbusters dataset."""

    def download(self, save_dir: Path):

        # Download the files
        url = "https://data.nerf.studio/nerfbusters-dataset.zip"
        download_path = str(save_dir / "nerfbusters-dataset.zip")
        gdown.download(url, output=download_path)
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(str(save_dir))
        os.remove(download_path)


def main(
    dataset: DatasetDownload,
):
    """Script to download the Nerfbusters data.
    - captures: These are the videos which were used the ns-process-data.
    - data: These are the already-processed results from running ns-process-data (ie COLMAP).
    Args:
        dataset: The dataset to download (from).
    """
    dataset.save_dir.mkdir(parents=True, exist_ok=True)

    dataset.download(dataset.save_dir)


Commands = Union[
    Annotated[NerfbustersDiffusionCubeWeightsDownload, tyro.conf.subcommand(name="diffusion-cube-weights")],
    Annotated[NerfbustersCapturesDownload, tyro.conf.subcommand(name="captures")],
    Annotated[NerfbustersDataDownload, tyro.conf.subcommand(name="dataset")],
]


def nerfbusters_setup():
    """The function that needs to be run to setup Nerfbusters for the Nerfstudio codebase."""
    NerfbustersDiffusionCubeWeightsDownload().download(Path("data/"))


def entrypoint():
    """Entrypoint."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()
