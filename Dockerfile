# uv installs torch together with its CUDA libraries (the nvidia-* packages
# in uv.lock), so the image needs only Python; a PyTorch base image would
# add a second, unused torch.
FROM docker.io/library/python:3.12-slim

# Links the published image to its repository on ghcr.io.
LABEL org.opencontainers.image.source="https://github.com/dmidk/mlwp-inference-container-leewavenet"
LABEL org.opencontainers.image.description="LeeWaveNet inference container for DMI MLWP"

# Kept from the former pytorch/pytorch base image: they tell the NVIDIA
# container runtime to give the container the GPU driver.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

USER 0

RUN pip install -U pip setuptools wheel
RUN pip install uv

# copy files
COPY pyproject.toml uv.lock README.md /project/
COPY leewavenet/ /project/leewavenet
COPY scripts/ /project/scripts

# Set working directory
WORKDIR /project

# Install the exact versions in uv.lock; fail if it doesn't match pyproject.toml.
# Done in three steps so that no image layer is much larger than ~1 GB:
# big layers time out when the cluster pulls the image.
# 1) everything except the largest packages
RUN uv sync --locked --no-dev --extra plot --no-install-project \
    --no-install-package torch --no-install-package triton \
    --no-install-package nvidia-cudnn-cu12 --no-install-package nvidia-cublas-cu12
# 2) the two largest CUDA libraries
RUN uv sync --locked --no-dev --extra plot --no-install-project \
    --no-install-package torch --no-install-package triton
# 3) torch, triton and the leewavenet package
RUN uv sync --locked --no-dev --extra plot

# Stop `uv run` from syncing again: during the build it would add the dev dependencies,
# and at start-up it would rebuild the package,
# which needs PyPI and fails without network access.
ENV UV_NO_SYNC=1

# Download the .pkl files from Zenodo into /project/storage/models
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
