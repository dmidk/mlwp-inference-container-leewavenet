import __main__
import argparse
import os

import torch
import xarray as xr
from fastai.vision.all import load_learner
from loguru import logger

from .inference import amplitude, orientation, segmentation, wavelength
from .read_source import read_source
from .write_output import write_predictions

# Define custom functions for fastai models (without them code crashes!)


def open_xarray(file):
    return xr.open_dataset(file)["upward_air_velocity"]


def label_func():
    return None


def open_np():
    return None


def label_func2():
    return None


def _register_custom_functions():
    """Make the custom functions above reachable from ``__main__``."""
    for func in (open_xarray, label_func, open_np, label_func2):
        setattr(__main__, func.__name__, func)


# ----------------- Model Utilities ----------------- #


def process_time_step(index, ds, gpu):
    _register_custom_functions()

    # Load models
    logger.debug(
        "Loading the following pkl files: segmentation_model, "
        "wavelength_model, amplitude_model, orientation_model"
    )
    segmentation_model = load_learner(
        "storage/models/segmentation_model.pkl", cpu=not gpu
    )
    wavelength_model = load_learner(
        "storage/models/wavelength_model.pkl", cpu=not gpu
    )
    amplitude_model = load_learner(
        "storage/models/amplitude_model.pkl", cpu=not gpu
    )
    orientation_model = load_learner(
        "storage/models/orientation_model.pkl", cpu=not gpu
    )
    logger.info("All the models have been loaded!")

    print(f"Processing Time Step: {index}")
    da_time = ds.wz.isel(time=index, pressure=8)

    print("Segmenting...")
    wave_mask = segmentation(segmentation_model, da_time)

    print("Computing amplitude...")
    amp = amplitude(amplitude_model, da_time) / wave_mask

    print("Computing wavelength...")
    wl = wavelength(wavelength_model, da_time) / wave_mask

    print("Computing orientation...")
    orient = orientation(orientation_model, da_time) / wave_mask

    print("Finished Time Step")

    return xr.Dataset(
        {
            "upward_air_velocity": da_time,
            "wave_mask": (("y", "x"), wave_mask),
            "wavelength": (("y", "x"), wl),
            "orientation": (("y", "x"), orient),
            "amplitude": (("y", "x"), amp),
        },
        coords={
            "y": ds.y,
            "x": ds.x,
            "time": ds.time[index],
        },
    )


DEFAULT_SOURCE = "storage/ig-recent/pressure_levels.zarr"
DEFAULT_DESTINATION = "storage/output"


def compute_and_plot_leewaves(
    source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION, plot=False
):
    """Apply ML models to compute lee waves and optionally plot each time step.

    `source` and `destination` may be local paths or any fsspec URL,
    e.g. "s3://bucket/path.zarr". Predictions are always written to
    `destination`; plots are written there too when `plot` is set.
    """
    ds = read_source(source)
    results = []

    # Run on the GPU if there is one. The models were saved with their data
    # loaders on cuda:0, so cpu=False would crash on a machine without a GPU
    gpu = torch.cuda.is_available()
    if gpu:
        logger.info(f"Running the models on {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU found, running the models on the CPU")

    if plot:
        # Imported lazily so the plotting stays optional.
        from .plotting import characteristics_plot, segmentation_plot

    for i in range(ds.sizes["time"]):
        leewaves = process_time_step(i, ds, gpu)
        results.append(leewaves)

        if plot:
            print(f"Plotting results for time step {i}")
            segmentation_plot(leewaves, destination, plot=True)
            characteristics_plot(leewaves, destination, plot=True)

    predictions = xr.concat(results, dim="time")
    write_predictions(predictions, destination)

    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run LeeWaveNet inference.")
    parser.add_argument(
        "--source",
        default=os.environ.get("LEEWAVENET_SOURCE", DEFAULT_SOURCE),
        help="Input Zarr dataset, local path or fsspec URL "
        "(env: LEEWAVENET_SOURCE)",
    )
    parser.add_argument(
        "--destination",
        default=os.environ.get("LEEWAVENET_DESTINATION", DEFAULT_DESTINATION),
        help="Output location for predictions and plots, local path or "
        "fsspec URL (env: LEEWAVENET_DESTINATION)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Also write and show plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compute_and_plot_leewaves(
        source=args.source, destination=args.destination, plot=args.plot
    )


if __name__ == "__main__":
    main()
