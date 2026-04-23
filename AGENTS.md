# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development commands

Use `uv` for Python environment and command execution.

- Install runtime dependencies: `uv sync`
- Install with dev extras (ruff): `uv sync --all-extras`
- Download sample data: `bash data/download.sh`

Primary `make` targets (preferred):

- Lint: `make lint`
- Format: `make format`
- Train: `make train` (override config: `make train CONFIG=configs/train.yaml`)
- Predict: `make predict` (override IO: `make predict INPUT=data/sample_tile.tif OUTPUT=output/fields.gpkg`)
- Build Docker image: `make docker`
- Run Docker prediction: `make docker-predict`
- Clean artifacts: `make clean`

Direct command equivalents:

- `uv run ruff check src/ predict.py`
- `uv run ruff format src/ predict.py`
- `uv run python src/train.py --config configs/train.yaml`
- `uv run python predict.py --input data/sample_tile.tif --output output/fields.gpkg`

Testing status:

- There is currently no automated test suite in-repo (`tests/` and pytest config are absent).
- For validation after changes, run lint plus a prediction smoke test (`make predict`) against sample data.
- If/when pytest tests are added, run a single test with: `uv run pytest path/to/test_file.py::test_name`

## High-level architecture

The repo implements an end-to-end raster-to-vector segmentation pipeline with two runtime entrypoints:

- Training: `src/train.py`
- Inference CLI: `predict.py`

### End-to-end flow

1. Read config from `configs/train.yaml` (`src/train.py`).
2. Build dataloaders from `data/processed/train` and `data/processed/val` (`src/dataset.py`).
3. Build ResUNet-A model with three output heads (`src/model.py`).
4. Train with weighted multi-head loss (extent BCE + boundary BCE + distance L1) in `src/train.py`.
5. Run inference from `predict.py`, then convert prediction rasters to polygons in `src/postprocess.py`.

### Core modules and responsibilities

- `src/model.py`: ResUNet-A-style network
  - Residual encoder blocks
  - Atrous bridge (`AtrousBlock`) for multi-scale context
  - Decoder with skip connections
  - Three heads: `extent`, `boundary`, `distance`

- `src/dataset.py`: data IO and target packaging
  - Expects `images/*.tif` and `labels/*.tif` under split roots
  - Returns tensors keyed as `image`, `extent`, `boundary`, `distance`
  - Current `_generate_targets` is a scaffold: extent is derived from label, boundary/distance are placeholder zero maps

- `src/train.py`: training loop orchestration
  - Creates optimizer (`AdamW`) and cosine scheduler
  - Logs to TensorBoard (`runs/`)
  - Saves checkpoints to configured directory
  - `validate()` is currently a TODO and returns zeroed metrics

- `src/evaluate.py`: numpy metric implementations (IoU, MCC, F1)
  - Not currently wired into `validate()` in `src/train.py`

- `src/postprocess.py`: raster-to-vector postprocessing
  - Threshold maps, then segment labels, then polygonize and clean geometry
  - Watershed and polygonization are currently partial placeholders (`TODO` markers), so vector outputs depend on further implementation

## Data and config conventions

- Training config lives in `configs/train.yaml` (paths, model channels, optimizer/loss weights, checkpoint/logging setup).
- Expected processed dataset structure is documented in `data/README.md` and referenced by config:
  - `data/processed/train/{images,labels}`
  - `data/processed/val/{images,labels}`
- Inference defaults to checkpoint `checkpoints/best.pt`; if missing, inference runs with untrained weights (warning only).
