#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CF compliance of the written `predictions.zarr`.

The tests write a small synthetic dataset through the real metadata and
writing code and then inspect the result both as raw Zarr v3 metadata
and through `xarray.open_zarr`. They need neither the models nor S3.
"""
import json
import os

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer

from leewavenet.cf_metadata import (
    CONVENTIONS,
    add_cf_metadata,
    grid_mapping,
    pressure_units,
)

from .conftest import NT, NX, NY, make_predictions, make_source

PREDICTION_VARIABLES = (
    "wave_mask",
    "amplitude",
    "wavelength",
    "orientation",
    "upward_air_velocity",
)

# The variables that may carry a `standard_name`: the CF standard name
# table has no entry for a lee wave detection or its characteristics.
STANDARD_NAMES = {
    "upward_air_velocity": "upward_air_velocity",
    "lat": "latitude",
    "lon": "longitude",
    "time": "time",
    "pressure": "air_pressure",
    "x": "projection_x_coordinate",
    "y": "projection_y_coordinate",
}

MODEL_PREDICTIONS = ("wave_mask", "amplitude", "wavelength", "orientation")


def read_metadata(store, name=None):
    """The raw Zarr v3 metadata of the root group or of one array."""
    path = store / name / "zarr.json" if name else store / "zarr.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------- #
# 1. A valid Zarr v3 dataset
# ---------------------------------------------------------------- #


def test_output_is_zarr_v3(written):
    root = read_metadata(written)
    assert root["zarr_format"] == 3
    assert root["node_type"] == "group"

    for name in PREDICTION_VARIABLES:
        array = read_metadata(written, name)
        assert array["zarr_format"] == 3
        assert array["node_type"] == "array"


# ---------------------------------------------------------------- #
# 2. and 3. Variables, dimensions and coordinates
# ---------------------------------------------------------------- #


def test_expected_variables_exist(written):
    ds = xr.open_zarr(written)
    assert set(PREDICTION_VARIABLES) <= set(ds.data_vars)


def test_dimensions_and_coordinates_preserved(written, predictions):
    ds = xr.open_zarr(written)

    assert ds.sizes["time"] == NT
    assert ds.sizes["y"] == NY
    assert ds.sizes["x"] == NX

    for name in PREDICTION_VARIABLES:
        assert ds[name].dims == ("time", "y", "x")

    # The auxiliary coordinates survive and stay two-dimensional.
    assert ds["lat"].dims == ("y", "x")
    assert ds["lon"].dims == ("y", "x")
    assert ds["pressure"].shape == ()

    for name in ("lat", "lon", "pressure", "time", "x", "y"):
        assert name in ds.coords


def test_prediction_variables_reference_their_coordinates(written):
    """CF: the auxiliary coordinates are named in `coordinates`."""
    for name in PREDICTION_VARIABLES:
        attrs = read_metadata(written, name)["attributes"]
        assert set(attrs["coordinates"].split()) == {
            "lat",
            "lon",
            "pressure",
        }


# ---------------------------------------------------------------- #
# 4. The Lambert Conformal grid mapping
# ---------------------------------------------------------------- #


def test_grid_mapping_present_and_valid(written):
    ds = xr.open_zarr(written)

    names = {ds[name].attrs["grid_mapping"] for name in PREDICTION_VARIABLES}
    assert len(names) == 1, "the variables disagree on the grid mapping"
    name = names.pop()

    # The reference must resolve: this is what was missing before.
    assert name in ds.variables

    attrs = ds[name].attrs
    assert attrs["grid_mapping_name"] == "lambert_conformal_conic"
    for parameter in (
        "standard_parallel",
        "latitude_of_projection_origin",
        "longitude_of_central_meridian",
        "false_easting",
        "false_northing",
    ):
        assert parameter in attrs

    # The attributes alone must describe a usable CRS.
    crs = CRS.from_cf({k: v for k, v in attrs.items() if k != "crs_wkt"})
    assert crs.is_projected


def test_grid_mapping_matches_the_coordinates(written):
    """The projection must reproduce `lat`/`lon` from `x`/`y`.

    This is the check that catches a grid mapping copied from the wrong
    grid: the parameters are only right if they put the grid points
    back where the source says they are.
    """
    ds = xr.open_zarr(written)
    name = ds["upward_air_velocity"].attrs["grid_mapping"]
    attrs = {k: v for k, v in ds[name].attrs.items() if k != "crs_wkt"}

    transformer = Transformer.from_crs(
        CRS.from_cf(attrs), CRS("EPSG:4326"), always_xy=True
    )
    grid_x, grid_y = np.meshgrid(ds.x.values, ds.y.values)
    lon, lat = transformer.transform(grid_x, grid_y)

    assert np.allclose(lat, ds.lat.values, atol=1e-9)
    # Compare longitudes wrapped, so the antimeridian does not matter.
    difference = (lon - ds.lon.values + 180) % 360 - 180
    assert np.allclose(difference, 0.0, atol=1e-9)


def test_grid_mapping_comes_from_the_source_whatever_its_name():
    """Nothing may depend on the projection variable's name."""
    for name in ("ig_projection", "dini_projection", "crs"):
        source = make_source(projection_name=name)
        ds = add_cf_metadata(make_predictions(source), source)

        assert name in ds.variables
        assert ds["amplitude"].attrs["grid_mapping"] == name
        assert ds[name].attrs["grid_mapping_name"] == "lambert_conformal_conic"


def test_grid_mapping_is_a_bare_container(written):
    """A grid mapping variable carries no coordinates of its own."""
    ds = xr.open_zarr(written)
    name = ds["upward_air_velocity"].attrs["grid_mapping"]

    assert "coordinates" not in read_metadata(written, name)["attributes"]
    # It describes a projection rather than referencing one.
    assert "grid_mapping" not in ds[name].attrs


def test_source_without_a_projection_is_not_invented(predictions):
    """With no projection to preserve, none is made up."""
    source = make_source()
    bare = source.drop_vars("ig_projection")
    bare["wz"].attrs.pop("grid_mapping")

    assert grid_mapping(bare) == (None, None)

    ds = add_cf_metadata(predictions, bare)
    assert "grid_mapping" not in ds["amplitude"].attrs


# ---------------------------------------------------------------- #
# 5. The pressure coordinate
# ---------------------------------------------------------------- #


def test_pressure_metadata(written):
    ds = xr.open_zarr(written)
    attrs = ds["pressure"].attrs

    assert attrs["standard_name"] == "air_pressure"
    assert attrs["long_name"] == "Pressure level"
    assert attrs["units"] == "hPa"
    assert attrs["positive"] == "down"

    # The value itself is untouched.
    assert ds["pressure"].item() == 700


@pytest.mark.parametrize(
    "level_type, expected",
    [("isobaricInhPa", "hPa"), ("isobaricInPa", "Pa")],
)
def test_pressure_units_follow_the_source(level_type, expected):
    """The units come from the source, they are not assumed."""
    source = make_source(level_type=level_type)
    ds = add_cf_metadata(make_predictions(source), source)
    assert ds["pressure"].attrs["units"] == expected


def test_pressure_units_already_on_the_source_win():
    source = make_source()
    source["pressure"].attrs["units"] = "Pa"
    predictions = make_predictions(source)
    assert pressure_units(predictions, source) == "Pa"


# ---------------------------------------------------------------- #
# 6. Coordinate metadata
# ---------------------------------------------------------------- #


def test_geographic_coordinate_attributes(written):
    ds = xr.open_zarr(written)

    assert ds["lat"].attrs["standard_name"] == "latitude"
    assert ds["lat"].attrs["units"] == "degrees_north"
    assert ds["lon"].attrs["standard_name"] == "longitude"
    assert ds["lon"].attrs["units"] == "degrees_east"

    assert ds["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert ds["x"].attrs["units"] == "m"
    assert ds["y"].attrs["standard_name"] == "projection_y_coordinate"
    assert ds["y"].attrs["units"] == "m"


def test_time_coordinate_attributes(written):
    stored = read_metadata(written, "time")["attributes"]

    assert stored["standard_name"] == "time"
    assert " since " in stored["units"]
    assert stored["calendar"] == "proleptic_gregorian"

    # And it still decodes to the times that went in.
    ds = xr.open_zarr(written)
    assert ds.time.dtype.kind == "M"


# ---------------------------------------------------------------- #
# 7. Prediction variable metadata
# ---------------------------------------------------------------- #


def test_prediction_variables_are_described(written):
    ds = xr.open_zarr(written)

    for name in PREDICTION_VARIABLES:
        attrs = ds[name].attrs
        assert attrs.get("long_name"), f"{name} has no long_name"
        assert attrs.get("units"), f"{name} has no units"


def test_prediction_variable_units(written):
    ds = xr.open_zarr(written)
    expected = {
        "wave_mask": "1",
        "amplitude": "m s-1",
        "wavelength": "km",
        "orientation": "degree",
        "upward_air_velocity": "m s-1",
    }
    for name, units in expected.items():
        assert ds[name].attrs["units"] == units


def test_standard_names_are_not_invented(written):
    """Only quantities that are in the CF table may claim one."""
    ds = xr.open_zarr(written)

    for name in MODEL_PREDICTIONS:
        assert "standard_name" not in ds[name].attrs, (
            f"{name} is a model prediction; CF has no standard name " "for it"
        )

    for name, standard_name in STANDARD_NAMES.items():
        assert ds[name].attrs["standard_name"] == standard_name


def test_orientation_convention_is_documented(written):
    """The angular convention is not guessable from the values."""
    ds = xr.open_zarr(written)
    comment = ds["orientation"].attrs["comment"]

    assert "anticlockwise from geographic east" in comment
    # It is an axis, ambiguous by 180 degrees, not a direction.
    assert "180" in comment


# ---------------------------------------------------------------- #
# 8. The wave mask flags
# ---------------------------------------------------------------- #


def test_wave_mask_flags(written, predictions):
    ds = xr.open_zarr(written)
    attrs = ds["wave_mask"].attrs

    assert attrs["long_name"] == "Lee wave detection mask"
    assert attrs["units"] == "1"
    assert list(np.asarray(attrs["flag_values"])) == [0, 1]
    assert attrs["flag_meanings"] == "no_lee_wave lee_wave_detected"

    # The flags must describe what the mask actually contains.
    assert set(np.unique(ds["wave_mask"].values)) <= {0, 1}

    # CF: as many meanings as values.
    assert len(attrs["flag_meanings"].split()) == len(
        np.asarray(attrs["flag_values"])
    )


def test_wave_mask_flags_match_the_characteristics(written):
    """`lee_wave_detected` must be the value the characteristics use."""
    ds = xr.open_zarr(written)
    detected = ds["wave_mask"].values == 1

    # main.py writes the characteristics only where the mask is 1.
    assert np.all(np.isfinite(ds["amplitude"].values) == detected)
    assert np.all(np.isfinite(ds["wavelength"].values) == detected)


# ---------------------------------------------------------------- #
# 9. Missing values
# ---------------------------------------------------------------- #


def test_float_fill_values_are_readable(written):
    """`_FillValue` must never be written as an unreadable blob.

    JSON has no NaN, so xarray encodes a NaN `_FillValue` attribute as
    base64. CF also requires `_FillValue` to have the type of its
    variable, which a base64 string does not. The Zarr fill value
    carries NaN instead.
    """
    for name in ("amplitude", "wavelength", "orientation"):
        array = read_metadata(written, name)

        assert array["fill_value"] == "NaN"
        stored = array["attributes"].get("_FillValue")
        assert not isinstance(stored, str), (
            f"{name}: _FillValue is the string {stored!r}, which no CF "
            "reader can interpret"
        )


def test_wave_mask_has_no_fill_value(written):
    """0 means `no_lee_wave`; it must not be read as missing."""
    array = read_metadata(written, "wave_mask")
    assert "_FillValue" not in array["attributes"]
    assert "missing_value" not in array["attributes"]


def test_missing_values_round_trip(written, predictions):
    """NaN goes in and NaN comes out, in the same places."""
    ds = xr.open_zarr(written)

    for name in ("amplitude", "wavelength", "orientation"):
        np.testing.assert_array_equal(
            np.isnan(ds[name].values), np.isnan(predictions[name].values)
        )


# ---------------------------------------------------------------- #
# 10. The predictions themselves
# ---------------------------------------------------------------- #


def test_values_are_unchanged(written, predictions):
    """Metadata only: every predicted value survives bit for bit."""
    ds = xr.open_zarr(written)

    for name in PREDICTION_VARIABLES:
        np.testing.assert_array_equal(
            ds[name].values, predictions[name].values
        )
        assert ds[name].dtype == predictions[name].dtype


def test_coordinate_values_are_unchanged(written, predictions):
    ds = xr.open_zarr(written)

    for name in ("lat", "lon", "x", "y", "pressure"):
        np.testing.assert_array_equal(
            ds[name].values, predictions[name].values
        )
    np.testing.assert_array_equal(ds.time.values, predictions.time.values)


def test_add_cf_metadata_leaves_its_input_alone(source, predictions):
    """The caller's datasets must not be modified in place."""
    before_predictions = {
        name: dict(predictions[name].attrs) for name in predictions.variables
    }
    before_source = {
        name: dict(source[name].attrs) for name in source.variables
    }

    add_cf_metadata(predictions, source)

    assert {
        name: dict(predictions[name].attrs) for name in predictions.variables
    } == before_predictions
    assert {
        name: dict(source[name].attrs) for name in source.variables
    } == before_source
    assert "ig_projection" not in predictions.variables


# ---------------------------------------------------------------- #
# 11. Reopening the dataset
# ---------------------------------------------------------------- #


def test_reopens_with_open_zarr(written):
    ds = xr.open_zarr(written)
    assert ds.sizes["time"] == NT
    # Reading the data, not just the metadata, must work too.
    assert np.isfinite(ds["upward_air_velocity"].values).all()


def test_reopens_through_fsspec(written):
    """`open_zarr` over an fsspec mapper, as a remote reader would."""
    import fsspec

    mapper = fsspec.get_mapper(f"file://{written}")
    ds = xr.open_zarr(mapper)
    assert set(PREDICTION_VARIABLES) <= set(ds.data_vars)


# ---------------------------------------------------------------- #
# 12. Global metadata and CF validation
# ---------------------------------------------------------------- #


def test_global_attributes(written):
    attrs = read_metadata(written)["attributes"]

    assert attrs["Conventions"] == CONVENTIONS
    assert attrs["title"]
    assert attrs["institution"] == "Danish Meteorological Institute"
    assert attrs["source"] == "LeeWaveNet machine learning inference"
    assert "history" in attrs


def test_provenance_is_carried_through_not_invented(written):
    attrs = read_metadata(written)["attributes"]

    # What the source stated about itself.
    assert attrs["source_zarr_creator_repo"].endswith(
        "nwp-forecast-zarr-creator"
    )
    assert attrs["source_zarr_creator_version"]
    assert attrs["source_dataset"] == "memory://source.zarr"

    # The models are identified by the record they are downloaded from.
    assert "zenodo.org" in attrs["model_source"]


def test_history_records_the_run(written):
    attrs = read_metadata(written)["attributes"]
    assert "leewavenet" in attrs["history"]
    assert "memory://source.zarr" in attrs["history"]


def test_every_units_attribute_is_udunits(written):
    """CF requires `units` to be a UDUNITS string.

    This is the check that backs the `Conventions` declaration: a unit
    UDUNITS cannot parse is a CF error, not a style problem.
    """
    cf_units = pytest.importorskip("cf_units")
    ds = xr.open_zarr(written)

    for name, variable in {**ds.variables}.items():
        units = variable.attrs.get("units")
        if units is None or name == "time":
            # Time units are checked by decoding them above.
            continue
        cf_units.Unit(units)  # raises if UDUNITS cannot parse it


def test_no_reserved_attributes_are_misused(written):
    """A few CF rules that are easy to break by accident."""
    ds = xr.open_zarr(written)

    for name in PREDICTION_VARIABLES:
        attrs = ds[name].attrs
        # `coordinates` belongs to the encoding, never the attributes.
        assert "coordinates" not in attrs
        # A data variable must not claim to be its own grid mapping.
        assert "grid_mapping_name" not in attrs


def cf_checker():
    """The CF checker module, or skip when UDUNITS is missing.

    cfchecker loads UDUNITS-2 through ctypes, so it needs the system
    `libudunits2` library rather than a wheel, and raises OSError
    instead of ImportError when it cannot find it.

    Skipping keeps a developer without that library from seeing a
    confusing failure. CI installs it and sets
    LEEWAVENET_REQUIRE_CF_CHECKER, which turns the skip into a failure
    there, so the validation can never quietly stop running.
    """
    try:
        import cfchecker.cfchecks as cfchecks
    except (ImportError, OSError) as error:
        message = (
            f"the CF checker is unavailable ({error}); install the "
            "libudunits2 system package"
        )
        if os.environ.get("LEEWAVENET_REQUIRE_CF_CHECKER"):
            pytest.fail(message)
        pytest.skip(message)

    return cfchecks


def test_declared_convention_is_one_we_can_check():
    """CF-x.y may only be declared if a checker can validate it.

    This is what keeps `CONVENTIONS` honest: raising it without a
    checker that understands that version fails here.
    """
    cfchecks = cf_checker()

    declared = CONVENTIONS.removeprefix("CF-")
    newest = str(cfchecks.newest_version).removeprefix("CF-")

    assert tuple(map(int, declared.split("."))) <= tuple(
        map(int, newest.split("."))
    ), (
        f"the output declares CF-{declared} but the installed checker "
        f"only validates up to CF-{newest}"
    )


def test_written_dataset_passes_cf_checker(written, tmp_path):
    """Run the real CF checker over the output.

    The checker works on NetCDF, so the dataset is exported first; what
    it validates is the metadata written to Zarr, unchanged.
    """
    pytest.importorskip("netCDF4")
    cfchecks = cf_checker()

    ds = xr.open_zarr(written)
    netcdf = tmp_path / "predictions.nc"
    ds.to_netcdf(netcdf)

    checker = cfchecks.CFChecker(
        version=CONVENTIONS.removeprefix("CF-"),
        silent=True,
    )
    checker.checker(str(netcdf))

    errors = {
        name: result["FATAL"] + result["ERROR"]
        for name, result in checker.results["variables"].items()
        if result["FATAL"] or result["ERROR"]
    }
    globals_ = (
        checker.results["global"]["FATAL"] + checker.results["global"]["ERROR"]
    )
    assert not errors, f"CF errors in variables: {errors}"
    assert not globals_, f"global CF errors: {globals_}"
