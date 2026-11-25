#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small synthetic stand-ins for the HARMONIE-IG input and the
LeeWaveNet predictions.

The grids are a few points across so the tests run in milliseconds, but
they are built the same way as the real ones: a Lambert Conformal Conic
projection, `lat`/`lon` computed from `x`/`y` through that projection,
and the GRIB attributes the zarr-creator passes on. Nothing here needs
the models, the network or S3.
"""
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer

NT, NY, NX = 2, 6, 5

# The HARMONIE-IG projection parameters, as pyproj reads them out of
# the source's `crs_wkt`.
PROJECTION = {
    "grid_mapping_name": "lambert_conformal_conic",
    "standard_parallel": (55.5, 55.5),
    "latitude_of_projection_origin": 55.5,
    "longitude_of_central_meridian": -8.0,
    "false_easting": 0.0,
    "false_northing": 0.0,
    "earth_radius": 6371229.0,
}

# The level the models run on, and the level type the source states.
PRESSURE_HPA = 700
LEVEL_TYPE = "isobaricInhPa"

SOURCE_PROVENANCE = {
    "zarr_creation_time": "2026-10-05T07:29:42.260795+00:00",
    "zarr_creator_repo": (
        "https://github.com/dmidk/nwp-forecast-zarr-creator"
    ),
    "zarr_creator_version": "0.8.1.dev25+g712eaad.d20260926",
}


def _grid():
    """Projected coordinates and the lat/lon they correspond to."""
    crs = CRS.from_cf(PROJECTION)
    x = np.linspace(-2.0e5, 2.0e5, NX)
    y = np.linspace(-1.5e5, 1.5e5, NY)
    transformer = Transformer.from_crs(crs, CRS("EPSG:4326"), always_xy=True)
    lon, lat = transformer.transform(*np.meshgrid(x, y))
    return x, y, lat, lon, crs


def make_source(projection_name="ig_projection", level_type=LEVEL_TYPE):
    """A stand-in for the pressure-level dataset `read_source` returns.

    `projection_name` is deliberately a parameter: the real source calls
    its grid mapping `dini_projection`, and nothing in the output
    pipeline may depend on that particular name.
    """
    x, y, lat, lon, crs = _grid()
    times = np.array(
        ["2026-10-30T06:00:00", "2026-10-30T07:00:00"],
        dtype="datetime64[ns]",
    )[:NT]

    rng = np.random.default_rng(0)
    wz = rng.normal(size=(NT, NY, NX))

    source = xr.Dataset(
        {
            "wz": (
                ("time", "y", "x"),
                wz,
                {
                    # The GRIB attributes the source carries through.
                    "cell_methods": "time: point",
                    "cfVarName": "tw",
                    "grid_mapping": projection_name,
                    "name": "Geometric vertical velocity",
                    "paramId": 260238,
                    "shortName": "tw",
                    "stepType": "instant",
                    "typeOfLevel": level_type,
                    "units": "m s**-1",
                },
            ),
            projection_name: ((), np.float64(0.0), {"crs_wkt": crs.to_wkt()}),
        },
        coords={
            "time": ("time", times, {"standard_name": "time"}),
            "y": (
                "y",
                y,
                {
                    "long_name": "y coordinate (northing)",
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                },
            ),
            "x": (
                "x",
                x,
                {
                    "long_name": "x coordinate (easting)",
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                },
            ),
            "lat": (
                ("y", "x"),
                lat,
                {
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            "lon": (
                ("y", "x"),
                lon,
                {
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "pressure": ((), PRESSURE_HPA),
        },
        attrs=dict(SOURCE_PROVENANCE),
    )
    return source


def make_predictions(source):
    """Predictions shaped exactly as `main.process_time_step` builds
    them: one Dataset per time step, concatenated along `time`.

    The values are arbitrary but fixed; the tests only check that
    nothing downstream changes them.
    """
    steps = []
    for index in range(source.sizes["time"]):
        velocity = source.wz.isel(time=index)

        # A checkerboard of detections, so both flag values occur.
        mask = ((np.arange(NY)[:, None] + np.arange(NX)[None, :]) % 2).astype(
            "int64"
        )
        waves = mask == 1

        # The characteristics are NaN outside the detected waves, as
        # `process_time_step` writes them.
        amplitude = np.where(waves, 1.5 + index, np.nan)
        wavelength = np.where(waves, 20.0 + index, np.nan)
        orientation = np.where(waves, -45.0 + index, np.nan)

        steps.append(
            xr.Dataset(
                {
                    "upward_air_velocity": velocity,
                    "wave_mask": (("y", "x"), mask),
                    "wavelength": (("y", "x"), wavelength, {"units": "km"}),
                    "orientation": (
                        ("y", "x"),
                        orientation,
                        {"units": "degree"},
                    ),
                    "amplitude": (("y", "x"), amplitude, {"units": "m s-1"}),
                },
                coords={
                    "y": source.y,
                    "x": source.x,
                    "time": source.time[index],
                },
            )
        )

    return xr.concat(steps, dim="time")


@pytest.fixture
def source():
    return make_source()


@pytest.fixture
def predictions(source):
    return make_predictions(source)


@pytest.fixture
def written(tmp_path, source, predictions):
    """The predictions after the real metadata and writing steps.

    Returns the path of the `predictions.zarr` that the pipeline would
    upload, written to a local directory instead of S3.
    """
    from leewavenet.cf_metadata import add_cf_metadata
    from leewavenet.write_output import write_predictions

    decorated = add_cf_metadata(
        predictions, source, source_url="memory://source.zarr"
    )
    write_predictions(decorated, str(tmp_path))
    return tmp_path / "predictions.zarr"
