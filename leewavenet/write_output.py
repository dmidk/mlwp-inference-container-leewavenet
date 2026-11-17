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


def reset_encoding(ds):
    """Drop the encodings inherited from the source Zarr.

    The source chunks and compressors no longer match the output arrays,
    so they go. Two encodings are set deliberately instead of cleared:

    - `_FillValue` is suppressed on the floating-point variables. JSON
      has no NaN, so xarray writes the attribute as a base64 blob that
      no CF reader can interpret, and CF requires `_FillValue` to have
      the type of its variable. The Zarr fill value stays "NaN", which
      Zarr v3 does encode, so missing values still round-trip exactly.
    - `coordinates: None` is kept where it was set (the grid mapping
      variable), so the container keeps no coordinates of its own.
    """
    ds = ds.copy()
    for var in ds.variables.values():
        drop_coordinates = var.encoding.get("coordinates", False) is None
        var.encoding = {}
        if drop_coordinates:
            var.encoding["coordinates"] = None
        if var.dtype.kind == "f":
            var.encoding["_FillValue"] = None
    return ds


def write_predictions(ds, destination, storage_options=None):
    """Write the predictions as `predictions.zarr` under `destination`.

    `destination` can be a local directory (e.g. "storage/output") or a remote
    URL (e.g. "s3://ml-model-output/leewavenet"). S3 credentials and endpoint
    are picked up from the same environment variables as `read_source`.
    """
    url = _join(destination, "predictions.zarr")
    logger.info(f"Writing predictions to: {url}")

    ds = reset_encoding(ds)

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
