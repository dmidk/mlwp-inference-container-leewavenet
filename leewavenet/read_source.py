#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fsspec
import xarray as xr
from loguru import logger

# The models take the vertical velocity at this pressure level
PRESSURE_LEVEL_HPA = 700


def read_source(source, storage_options=None):
    """Open a Zarr dataset from any fsspec-supported location and select
    the pressure level the models take (`PRESSURE_LEVEL_HPA`).

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

    # zarr-creator stores the vertical velocity as `tw` (GRIB2 0/2/9, m s-1);
    # the models call it `wz`
    ds = ds.rename({"tw": "wz"})

    # Select the level by value, so the order of the levels doesn't matter
    try:
        return ds.sel(pressure=PRESSURE_LEVEL_HPA)
    except KeyError:
        raise ValueError(
            f"The input has no {PRESSURE_LEVEL_HPA} hPa level, only "
            f"{ds['pressure'].values.tolist()}"
        ) from None
