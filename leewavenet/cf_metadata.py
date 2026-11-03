#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CF metadata for the LeeWaveNet predictions.

Everything here only sets attributes, re-attaches the grid mapping
variable the source dropped and fills in the global metadata. No
function in this module touches a predicted value.

The projection parameters are never written by hand: they are derived
from the `crs_wkt` the source grid definition carries, so the output
follows the input grid instead of a copy of it that can drift.
"""
import datetime
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from loguru import logger

# The CF version the output declares.
#
# CF-1.8 rather than a newer one because this is the newest version the
# output is actually checked against: cfchecker 4.1.0, the latest
# release, validates up to CF-1.8 and rejects anything newer as an
# unknown convention. The output passes it with no errors and no
# warnings (see tests/test_cf_metadata.py).
#
# Nothing written here was introduced after CF-1.6 (grid mappings,
# flag variables, auxiliary coordinates and `positive`), so declaring a
# later version would add no information. To raise it, bump this
# constant once a checker that supports that version is available; the
# test asserts the two agree.
CONVENTIONS = "CF-1.8"

TITLE = "LeeWaveNet predictions over the HARMONIE-IG domain"
INSTITUTION = "Danish Meteorological Institute"
SOURCE = "LeeWaveNet machine learning inference"
REFERENCES = "https://github.com/dmidk/mlwp-inference-container-leewavenet"

# The .pkl models come from this Zenodo record. Keep in sync with
# scripts/download_models.py, which downloads them.
MODELS_URL = "https://zenodo.org/records/23296247"

# Root attributes the zarr-creator writes on the source. They describe
# the input, so they are copied under a `source_` prefix.
SOURCE_PROVENANCE = (
    "zarr_creation_time",
    "zarr_creator_repo",
    "zarr_creator_version",
)

# GRIB's `typeOfLevel` names the unit of the pressure levels.
_LEVEL_TYPE_UNITS = {
    "isobaricInhPa": "hPa",
    "isobaricInPa": "Pa",
}

# `upward_air_velocity` is the vertical velocity the models run on,
# passed through from the source. The others are model predictions, for
# which no CF standard name exists, so they only get a `long_name`.
PREDICTION_ATTRS = {
    "upward_air_velocity": {
        "standard_name": "upward_air_velocity",
        "long_name": "Geometric vertical air velocity",
        "units": "m s-1",
    },
    "wave_mask": {
        "long_name": "Lee wave detection mask",
        "units": "1",
        "flag_meanings": "no_lee_wave lee_wave_detected",
    },
    "amplitude": {
        "long_name": "Predicted lee wave amplitude",
        "units": "m s-1",
        "comment": (
            "Amplitude of the vertical velocity of the detected "
            "trapped lee waves. NaN where no lee wave is detected."
        ),
    },
    "wavelength": {
        "long_name": "Predicted lee wave wavelength",
        "units": "km",
        "comment": (
            "Horizontal wavelength of the detected trapped lee waves. "
            "NaN where no lee wave is detected."
        ),
    },
    "orientation": {
        "long_name": "Predicted lee wave orientation",
        "units": "degree",
        "comment": (
            "Orientation of the detected trapped lee wave pattern, "
            "measured anticlockwise from geographic east. The model "
            "returns an arctangent, so the value lies in (-90, 90) "
            "degrees and defines an axis rather than a direction: an "
            "orientation and that orientation plus 180 degrees "
            "describe the same pattern. NaN where no lee wave is "
            "detected."
        ),
    },
}

# The value the mask takes where it detects a wave, as main.py reads it.
WAVE_FLAG_VALUES = (0, 1)


def _package_version():
    """The installed leewavenet version, or None outside an install."""
    try:
        return version("leewavenet")
    except PackageNotFoundError:  # pragma: no cover - editable checkouts
        logger.debug("leewavenet is not installed; omitting its version")
        return None


def _grid_mapping_name(source):
    """The name of the grid mapping variable `source` points at.

    Follows the `grid_mapping` attribute of the data variables rather
    than assuming a name, so a renamed projection variable still works.
    """
    for var in source.data_vars.values():
        name = var.attrs.get("grid_mapping")
        if name in source.variables:
            return name

    # No usable reference: fall back to a variable that looks like a
    # grid mapping on its own.
    for name, var in source.variables.items():
        if {"grid_mapping_name", "crs_wkt"} & set(var.attrs):
            return name

    return None


def _cf_from_wkt(wkt):
    """The CF grid mapping attributes described by a WKT string."""
    try:
        from pyproj import CRS
    except ImportError:  # pragma: no cover - pyproj is a dependency
        logger.warning("pyproj is missing; cannot expand crs_wkt")
        return {}

    try:
        attrs = CRS.from_wkt(wkt).to_cf()
    except Exception as error:
        logger.warning(f"Cannot read the projection WKT: {error}")
        return {}

    # pyproj returns tuples (e.g. standard_parallel) and None for the
    # parameters that do not apply; neither belongs in the output.
    attrs = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in attrs.items()
        if value is not None
    }

    # CF describes a spherical earth with `earth_radius`, which pyproj
    # reports as two equal axes.
    major = attrs.get("semi_major_axis")
    if major is not None and major == attrs.get("semi_minor_axis"):
        attrs.setdefault("earth_radius", major)

    return attrs


def grid_mapping(source):
    """The source's grid mapping variable, with its CF attributes.

    Returns a `(name, DataArray)` pair, or `(None, None)` when the
    source has no projection to preserve.
    """
    name = _grid_mapping_name(source)
    if name is None:
        logger.warning(
            "The source has no grid mapping variable; the predictions "
            "will have no grid_mapping"
        )
        return None, None

    # Selecting a level attaches the scalar pressure coordinate to
    # every variable; a grid mapping container carries no coordinates.
    variable = source[name].reset_coords(drop=True).copy()
    attrs = dict(variable.attrs)

    # A grid mapping variable describes the projection, it does not
    # reference one.
    attrs.pop("grid_mapping", None)

    wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    if wkt:
        # Whatever the source already states wins over the WKT.
        attrs = {**_cf_from_wkt(wkt), **attrs}

    if "grid_mapping_name" not in attrs:
        logger.warning(
            f"'{name}' has no grid_mapping_name and no usable WKT; "
            "leaving its attributes as they are"
        )

    variable.attrs = attrs
    # The projection is a container variable: it carries no data and so
    # no coordinates either.
    variable.encoding = {"coordinates": None}
    return name, variable


def pressure_units(predictions, source=None):
    """The units of the pressure coordinate, as the source states them.

    Falls back to GRIB's `typeOfLevel` and finally to the magnitude of
    the value, so the units always match what is actually stored.
    """
    pressure = predictions["pressure"]
    datasets = [d for d in (predictions, source) if d is not None]

    # What the source states itself, if anything.
    for dataset in datasets:
        units = dataset["pressure"].attrs.get("units")
        if units:
            return units

    # GRIB names the unit of the level in `typeOfLevel`.
    for dataset in datasets:
        for var in dataset.data_vars.values():
            units = _LEVEL_TYPE_UNITS.get(var.attrs.get("typeOfLevel"))
            if units:
                return units

    # Pressure levels of NWP interest are at most ~1100 hPa, while the
    # same levels in Pa are at least 1000.
    units = "hPa" if float(np.max(pressure.values)) < 2000 else "Pa"
    logger.warning(
        f"The source does not state the pressure units; inferring "
        f"'{units}' from the value {pressure.values}"
    )
    return units


def _history_entry(source_url=None):
    """A CF `history` line for this inference run."""
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    installed = _package_version()
    tool = f"leewavenet {installed}" if installed else "leewavenet"
    entry = f"{stamp}: {tool} inference"
    return f"{entry} on {source_url}" if source_url else entry


def _global_attrs(predictions, source=None, source_url=None):
    """The global attributes to write on the root group."""
    attrs = {
        "Conventions": CONVENTIONS,
        "title": TITLE,
        "institution": INSTITUTION,
        "source": SOURCE,
        "references": REFERENCES,
        "model_source": MODELS_URL,
    }

    installed = _package_version()
    if installed:
        attrs["leewavenet_version"] = installed

    if source_url:
        attrs["source_dataset"] = str(source_url)

    # Carry the input's own provenance through, marked as the source's.
    for key in SOURCE_PROVENANCE:
        value = (source.attrs if source is not None else {}).get(key)
        if value is not None:
            attrs[f"source_{key}"] = value

    history = [_history_entry(source_url)]
    existing = predictions.attrs.get("history")
    if existing:
        history.insert(0, existing)
    attrs["history"] = "\n".join(history)

    return {**predictions.attrs, **attrs}


def _with_attrs(variable, new_attrs):
    """Replace a variable's attributes without mutating the original."""
    variable.attrs = {**variable.attrs, **new_attrs}


def add_cf_metadata(predictions, source=None, source_url=None):
    """Return `predictions` with CF metadata, values untouched.

    `source` is the dataset the predictions were computed from; it
    supplies the grid mapping and the pressure units. `source_url` is
    recorded in the provenance.
    """
    predictions = predictions.copy(deep=False)

    name, projection = (None, None)
    if source is not None:
        name, projection = grid_mapping(source)
    if projection is not None:
        predictions[name] = projection
        logger.info(f"Added the grid mapping variable '{name}'")

    for variable, attrs in PREDICTION_ATTRS.items():
        if variable not in predictions:
            logger.warning(f"The predictions have no '{variable}'")
            continue

        attrs = dict(attrs)
        if variable == "wave_mask":
            # CF asks a flag variable's flag_values to have the same
            # type as the variable itself.
            attrs["flag_values"] = np.array(
                WAVE_FLAG_VALUES, dtype=predictions[variable].dtype
            )
        if name is not None:
            attrs["grid_mapping"] = name

        _with_attrs(predictions[variable], attrs)

    if "pressure" in predictions.coords:
        _with_attrs(
            predictions["pressure"],
            {
                "standard_name": "air_pressure",
                "long_name": "Pressure level",
                "units": pressure_units(predictions, source),
                "positive": "down",
            },
        )

    predictions.attrs = _global_attrs(predictions, source, source_url)
    return predictions
