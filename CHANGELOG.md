# Changelog

## [Unreleased]

### Changed
- Updated documentation.
- Bumped `requires-python` from `>=3.10,<3.11` to `<3.13`.
- Bumped `torch` from `1.13.1` to `2.2.0` (required for Python 3.12 compatibility; verified against the existing `.pkl` models).
- `read_source.py` now reads pre-converted zarr data from local `storage/` instead of indexing raw GRIB files live via `dmidc`/`gribscan` (much faster: seconds instead of ~21 min).
- Run pre-commit to satisfy general linting [!3](https://gitlab.dmi.dk/nwp/leewavenet/-/merge_requests/3)
- Fixed the container build and simplified the Dockerfile. [!4](https://gitlab.dmi.dk/nwp/leewavenet/-/merge_requests/4)
- Made plotting an optional dependency group. [!4](https://gitlab.dmi.dk/nwp/leewavenet/-/merge_requests/4)
- Verified that models downloaded from Nexus and from Zenodo are byte-identical, and that running the full inference pipeline with each source gives identical predictions (max abs diff = 0 across all output variables).
- Run the models on the GPU when one is available, otherwise on the CPU

### Removed
- Dropped `dmidc[harmonie-grib]` dependency (no longer used by `read_source.py`).
- Removed redundant/unnecessary pins: `shapely`, `eccodes`, `torchvision`, `pandas`, `pillow`, `scipy`, `fastcore`, `fastdownload`, `fastprogress`, `scikit-learn`, `netcdf4`, `h5py` — all either transitively satisfied by other dependencies or unused after the `dmidc` removal.

### Fixed
- Fix LeeWaveNet container exit code after successful inference
- Run the container without network access: `uv run` no longer re-syncs the environment on start-up, and the dev dependencies are no longer installed in the image
- Select the 700 hPa level by value instead of by position; on zarr-creator's IG stores position 8 is 500 hPa

### Added
- `scripts/download_models.py` — downloads the model `.pkl` files into `storage/models/`, using paths relative to the script itself (portable across machines/checkouts).
- `ModelDownloader` abstract base class in `scripts/download_models.py`, with `NexusDownloader` and `ZenodoDownloader` subclasses; `storage_dir` is configurable via the constructor. Defaults to `ZenodoDownloader`.
- Read and write data directly from and to S3 buckets using fsspec, so it can write to both filesystem and S3 [!3](https://gitlab.dmi.dk/nwp/leewavenet/-/merge_requests/3)
