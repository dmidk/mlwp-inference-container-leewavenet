#!/usr/bin/env python
"""Download the leewavenet model files into storage/models/.

Run from anywhere: python scripts/download_models.py
"""

import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger

MODEL_FILES = [
    "amplitude_model.pkl",
    "orientation_model.pkl",
    "segmentation_model.pkl",
    "wavelength_model.pkl",
]

DEFAULT_STORAGE_MODELS_DIR = (
    Path(__file__).resolve().parent.parent / "storage" / "models"
)


class ModelDownloader(ABC):
    def __init__(self, storage_dir=DEFAULT_STORAGE_MODELS_DIR):
        self.storage_dir = Path(storage_dir)

    @abstractmethod
    def url_template(self, filename: str) -> str: ...

    def download(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        for filename in MODEL_FILES:
            fp = self.storage_dir / filename
            url = self.url_template(filename)
            logger.info(f"Downloading {url} -> {fp}")
            urllib.request.urlretrieve(url, fp)

        logger.info(f"Done. Models saved to {self.storage_dir}")


class ZenodoDownloader(ModelDownloader):
    def url_template(self, filename: str) -> str:
        return (
            f"https://zenodo.org/api/records/23296247/files/{filename}/content"
        )


if __name__ == "__main__":
    ZenodoDownloader().download()
