#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fsspec
from fsspec.utils import get_protocol
from loguru import logger


def _join(destination, name):
    return f"{destination.rstrip('/')}/{name}"


def _storage_options(url, storage_options=None):
    # Local destinations need parent directories created on the fly; object
    # stores such as S3 have no directories (and reject the option)
    options = {"auto_mkdir": True} if get_protocol(url) == "file" else {}
    return {**options, **(storage_options or {})}


def write_predictions(ds, destination, storage_options=None):
    """Write the predictions as `predictions.zarr` under `destination`.

    `destination` can be a local directory (e.g. "storage/output") or a remote
    URL (e.g. "s3://ml-model-output/leewavenet"). S3 credentials and endpoint
    are picked up from the same environment variables as `read_source`.
    """
    url = _join(destination, "predictions.zarr")
    logger.info(f"Writing predictions to: {url}")

    # Drop encodings inherited from the source Zarr (chunks, compressors),
    # which no longer match the output arrays
    ds = ds.copy()
    for var in ds.variables.values():
        var.encoding = {}

    mapper = fsspec.get_mapper(url, **_storage_options(url, storage_options))
    ds.to_zarr(mapper, mode="w")


def save_figure(fig, destination, filename, storage_options=None):
    """Write a figure as PNG to `destination`.

    `destination` is a local directory or an fsspec URL.
    """
    url = _join(destination, filename)
    logger.info(f"Saving {url}")
    with fsspec.open(url, "wb", **_storage_options(url, storage_options)) as f:
        fig.savefig(f, format="png")
