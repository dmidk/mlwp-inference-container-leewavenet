#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fsspec
import xarray as xr
from loguru import logger


def read_source(source, storage_options=None):
    """Open a Zarr dataset from any fsspec-supported location.

    `source` can be a local path
    (e.g. "storage/ig-recent/pressure_levels.zarr")
    or a remote URL (e.g. "s3://bucket/path/pressure_levels.zarr").
    For S3, credentials and endpoint are picked up from the standard AWS
    environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    AWS_ENDPOINT_URL) unless given explicitly in `storage_options`.
    """
    logger.info(f"Reading source: {source}")
    mapper = fsspec.get_mapper(source, **(storage_options or {}))
    ds = xr.open_zarr(mapper)

    # Rename variables for compatibility but not physically sound
    ds = ds.rename({"tw": "wz"})

    return ds
