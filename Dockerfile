# Use Pytorch base image from DockerHub
#FROM dockerhub.dmi.dk/anibali/pytorch
FROM docker.io/pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Links the published image to its repository on ghcr.io.
LABEL org.opencontainers.image.source="https://github.com/dmidk/mlwp-inference-container-leewavenet"
LABEL org.opencontainers.image.description="LeeWaveNet inference container for DMI MLWP"

ENV DEBIAN_FRONTEND=noninteractive

USER 0

RUN apt-get update -y && \
    apt-get install -y \
    build-essential \
    libgeos-dev \
    libeccodes-dev \
    libeccodes-tools \
  && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel
RUN pip install uv

# copy files
COPY pyproject.toml README.md /project/
COPY leewavenet/ /project/leewavenet
COPY scripts/ /project/scripts

# Set working directory
WORKDIR /project

RUN uv sync --no-dev --extra plot

# Stop `uv run` from syncing again: during the build it would add the dev dependencies,
# and at start-up it would rebuild the package,
# which needs PyPI and fails without network access.
ENV UV_NO_SYNC=1

# Download the .pkl files from Nexus into /project/storage/models
RUN uv run python scripts/download_models.py

# Coastlines for --plot: cartopy would download them on first use, and the
# Airflow dev server has no internet access. cartopy looks in CARTOPY_DATA_DIR
# first; coastlines() picks the scale from the map extent, so all three
ENV CARTOPY_DATA_DIR=/usr/local/share/cartopy
RUN uv run python -c 'import cartopy, cartopy.io.shapereader as shp; \
    cartopy.config["data_dir"] = cartopy.config["pre_existing_data_dir"]; \
    [print(shp.natural_earth(scale, "physical", "coastline")) \
     for scale in ["110m", "50m", "10m"]]'

# Set the entry point. Arguments given to `podman run` are passed
# straight to the leewavenet command.
ENTRYPOINT ["uv", "run", "leewavenet"]
