import datetime
import gc

import cartopy.crs as ccrs
import matplotlib.colors as colours
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .write_output import save_figure


def quiver_orient(dataset, coords, sep=32):
    """Convert raw orientations into suitable ones for plotting as quivers"""

    # How many values are not finite
    orientations = dataset["orientation"].values
    finites = np.isfinite(orientations).sum()
    print(
        f"Number of finite elements in orientation: {finites} "
        f"out of {orientations.size}"
    )

    # Print first finite value
    finite_orientations = orientations[np.isfinite(orientations)]
    print(f"First finite value: {finite_orientations[0]}")

    # Convert to radians
    angle_rad = dataset["orientation"].values * (np.pi / 180)

    # Determine sizes for the new coarser grid
    new_y_size = (coords["y"].size + sep - 1) // sep
    new_x_size = (coords["x"].size + sep - 1) // sep

    # Initialize new coordinate arrays and angle array
    new_x = np.zeros(new_x_size)
    new_y = np.zeros(new_y_size)
    angle_rad2 = np.zeros((new_y_size, new_x_size))

    for i in range(new_y_size):
        for j in range(new_x_size):
            # Safely access elements ensuring indices do not exceed bounds
            original_y_idx = min(sep * i, coords["y"].size - 1)
            original_x_idx = min(sep * j, coords["x"].size - 1)

            new_angle_rad = (
                np.pi / 2 - angle_rad[original_y_idx, original_x_idx]
            )
            new_y[i] = coords["y"][original_y_idx].values
            new_x[j] = coords["x"][original_x_idx].values
            angle_rad2[i, j] = new_angle_rad

    # Create new dataset on more coarse grid, with wave orientations as
    # u and v components
    alt_dataframe = xr.Dataset(
        data_vars={"angle_rad": (("y", "x"), angle_rad2)},
        coords={"x": new_x, "y": new_y},
    )
    alt_dataframe["orient_u"] = (("y", "x"), np.cos(angle_rad2))
    alt_dataframe["orient_v"] = (("y", "x"), np.sin(angle_rad2))
    alt_dataframe["-orient_u"] = (("y", "x"), -np.cos(angle_rad2))
    alt_dataframe["-orient_v"] = (("y", "x"), -np.sin(angle_rad2))

    return alt_dataframe


def _close_figure(fig):
    """Free a figure's memory."""
    plt.close(fig)
    gc.collect()


def characteristics_plot(leewaves, destination, plot):
    """Plot characteristics"""
    cmap = "viridis"
    vv_cmap = colours.ListedColormap(
        [
            (0 / 255, 56 / 255, 116 / 255),
            (0 / 255, 101 / 255, 206 / 255),
            (0 / 255, 128 / 255, 189 / 255),
            (0 / 255, 151 / 255, 154 / 255),
            (0 / 255, 174 / 255, 119 / 255),
            (0 / 255, 197 / 255, 85 / 255),
            (0 / 255, 220 / 255, 50 / 255),
            (0 / 255, 244 / 255, 16 / 255),
            (255 / 255, 253 / 255, 255 / 255),
            (238 / 255, 212 / 255, 0 / 255),
            (225 / 255, 182 / 255, 0 / 255),
            (213 / 255, 151 / 255, 0 / 255),
            (201 / 255, 121 / 255, 0 / 255),
            (189 / 255, 91 / 255, 0 / 255),
            (177 / 255, 60 / 255, 0 / 255),
            (165 / 255, 30 / 255, 0 / 255),
            (153 / 255, 0 / 255, 0 / 255),
        ]
    )
    vv_cmap2 = colours.ListedColormap(
        [
            (255 / 255, 253 / 255, 255 / 255),
            (238 / 255, 212 / 255, 0 / 255),
            (238 / 255, 212 / 255, 0 / 255),
            (225 / 255, 182 / 255, 0 / 255),
            (225 / 255, 182 / 255, 0 / 255),
            (213 / 255, 151 / 255, 0 / 255),
            (213 / 255, 151 / 255, 0 / 255),
            (201 / 255, 121 / 255, 0 / 255),
            (201 / 255, 121 / 255, 0 / 255),
            (189 / 255, 91 / 255, 0 / 255),
            (189 / 255, 91 / 255, 0 / 255),
            (177 / 255, 60 / 255, 0 / 255),
            (177 / 255, 60 / 255, 0 / 255),
            (165 / 255, 30 / 255, 0 / 255),
            (165 / 255, 30 / 255, 0 / 255),
            (153 / 255, 0 / 255, 0 / 255),
            (153 / 255, 0 / 255, 0 / 255),
        ]
    )

    proj = ccrs.LambertConformal(
        central_longitude=-40.0, central_latitude=70.0
    )
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(50, 50))
    ax1 = fig.add_subplot(221, projection=proj)
    ax2 = fig.add_subplot(222, projection=proj)
    ax3 = fig.add_subplot(223, projection=proj)
    ax4 = fig.add_subplot(224, projection=proj)

    leewaves["wave_mask"].plot.contour(
        ax=ax1,
        x="lon",
        y="lat",
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        linewidths=0.5,
    )
    leewaves["upward_air_velocity"].plot.pcolormesh(
        ax=ax1,
        x="lon",
        y="lat",
        cmap=vv_cmap,
        transform=ccrs.PlateCarree(),
        vmin=-1.25,
        vmax=1.25,
        cbar_kwargs={
            "label": "Upward Air velocity (ms$^{-1}$)",
            "shrink": 0.8,
            "extend": "neither",
        },
    )
    leewaves["wavelength"].plot.pcolormesh(
        ax=ax2,
        x="lon",
        y="lat",
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        alpha=1,
        levels=np.arange(5, 50, 5),
        cbar_kwargs={
            "label": "Wavelength (km)",
            "shrink": 0.6,
            "extend": "neither",
        },
    )
    leewaves["amplitude"].plot.pcolormesh(
        cmap=vv_cmap2,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=4.25,
        robust=False,
        rasterized=True,
        ax=ax3,
        add_colorbar=True,
        cbar_kwargs={
            "label": "Amplitude Prediction (m s $^{-1}$)",
            "shrink": 0.6,
            "ticks": np.arange(0, 4.5, 0.5),
            "extend": "neither",
        },
    )

    leewaves["upward_air_velocity"].plot.pcolormesh(
        cmap=vv_cmap,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        robust=False,
        rasterized=True,
        ax=ax4,
        vmin=-4.25,
        vmax=4.25,
        add_colorbar=True,
        alpha=1,
        cbar_kwargs={
            "label": "Upward Air velocity (ms$^{-1}$)",
            "shrink": 0.6,
            "extend": "neither",
        },
    )

    # Calculate U,V components of orientation
    u = np.cos(np.deg2rad(leewaves["orientation"].values))
    v = np.sin(np.deg2rad(leewaves["orientation"].values))

    leewaves["u"] = (("y", "x"), u)
    leewaves["v"] = (("y", "x"), v)
    leewaves["-u"] = (("y", "x"), -u)
    leewaves["-v"] = (("y", "x"), -v)

    # Quiver plot
    headlength = 3
    headaxislength = 2
    width = 0.004

    lats = leewaves["lat"].values
    lons = leewaves["lon"].values

    thinfactor = 16
    tf = thinfactor
    ax4.quiver(
        lons[::tf, ::tf],
        lats[::tf, ::tf],
        u[::tf, ::tf],
        v[::tf, ::tf],
        transform=data_crs,
        width=width,
        pivot="tail",
        headlength=headlength,
        headaxislength=headaxislength,
    )
    ax4.quiver(
        lons[::tf, ::tf],
        lats[::tf, ::tf],
        -u[::tf, ::tf],
        -v[::tf, ::tf],
        transform=data_crs,
        width=width,
        pivot="tail",
        headlength=headlength,
        headaxislength=headaxislength,
    )

    ax1.set_title("(a) Data & Wave Location Prediction")
    ax2.set_title("(b) Wavelength Prediction")
    ax3.set_title("(c) Amplitude Prediction")
    ax4.set_title("(d) Orientation Prediction")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.coastlines()

    valid_time = str(leewaves["time"].values)[:-10] + "Z"
    fig.suptitle(
        "Lee Wave Forecast & Characteristics \nValid "
        + valid_time
        + " (T+ "
        + ")"
    )

    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    filename = f"characteristics_{timestamp}.png"
    save_figure(fig, destination, filename)
    if plot:
        plt.show()
    _close_figure(fig)


def segmentation_plot(leewaves, destination, plot):
    """Plot segmentation model output"""
    fig, ax = plt.subplots(
        subplot_kw={
            "projection": ccrs.LambertConformal(
                central_longitude=-40.0, central_latitude=70.0
            )
        },
        figsize=(10, 10),
    )
    vv_cmap = colours.ListedColormap(
        [
            (0 / 255, 56 / 255, 116 / 255),
            (0 / 255, 101 / 255, 206 / 255),
            (0 / 255, 128 / 255, 189 / 255),
            (0 / 255, 151 / 255, 154 / 255),
            (0 / 255, 174 / 255, 119 / 255),
            (0 / 255, 197 / 255, 85 / 255),
            (0 / 255, 220 / 255, 50 / 255),
            (0 / 255, 244 / 255, 16 / 255),
            (255 / 255, 253 / 255, 255 / 255),
            (238 / 255, 212 / 255, 0 / 255),
            (225 / 255, 182 / 255, 0 / 255),
            (213 / 255, 151 / 255, 0 / 255),
            (201 / 255, 121 / 255, 0 / 255),
            (189 / 255, 91 / 255, 0 / 255),
            (177 / 255, 60 / 255, 0 / 255),
            (165 / 255, 30 / 255, 0 / 255),
            (153 / 255, 0 / 255, 0 / 255),
        ]
    )
    if (
        leewaves["wave_mask"].values == np.zeros_like(leewaves["wave_mask"])
    ).all():
        cmap = colours.ListedColormap(["white"])
    else:
        cmap = colours.ListedColormap(["white", "black"])

    leewaves["wave_mask"].plot.contour(
        ax=ax,
        x="lon",
        y="lat",
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        linewidths=0.5,
    )
    leewaves["upward_air_velocity"].plot.pcolormesh(
        ax=ax,
        x="lon",
        y="lat",
        cmap=vv_cmap,
        transform=ccrs.PlateCarree(),
        vmin=-1.25,
        vmax=1.25,
        cbar_kwargs={
            "label": "Upward Air velocity (ms$^{-1}$)",
            "shrink": 0.8,
            "extend": "neither",
        },
    )

    ax.coastlines()

    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    filename = f"segmentation_{timestamp}.png"
    save_figure(fig, destination, filename)
    if plot:
        plt.show()
    _close_figure(fig)
