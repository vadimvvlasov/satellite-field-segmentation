# Satellite Field Boundary Segmentation

End-to-end pipeline for agricultural field boundary detection from satellite imagery.
Sentinel-2 / multispectral input → ResUNet-A segmentation → geospatial postprocessing → GeoPackage output.

![Example output](assets/example_output.png)

---

## Results

| Metric | Value |
|---|---|
| MCC (val) | 0.91 |
| Strict IoU (val) | 0.917 |
| Countries tested | USA, Brazil, Russia, Europe |
| Training data | 16.9 GB, auto-collected (no manual labeling) |

---

## Pipeline

```
Sentinel-2 L2A (STAC)
    ↓
Preprocessing (cloud mask, normalization)
    ↓
ResUNet-A (extent + boundary + distance maps)
    ↓
Watershed vectorization (Higra)
    ↓
Geometry cleanup (GeoPandas)
    ↓
GeoPackage output
```

---

## Quickstart

```bash
# Install (requires uv)
uv sync --all-extras

# Download sample data
bash data/download.sh

# Train
uv run python src/train.py --config configs/train.yaml

# Predict
uv run python predict.py --input data/sample_tile.tif --output output/fields.gpkg
```

### Lint & format

```bash
uv run ruff check .
uv run ruff format .
```

Or with Docker:
```bash
docker build -t field-seg .
docker run -v $(pwd)/data:/data field-seg python predict.py --input /data/tile.tif --output /data/out.gpkg
```

---

## Architecture

**ResUNet-A** — Residual U-Net with Atrous Convolutions.
Multi-head output: extent map + boundary map + distance transform.
Input: 256×256×4 (RGB + NIR). Trained with AdamW, 5-fold CV.

Key design decisions:
- Multi-head outputs improve boundary sharpness vs single-head segmentation
- Atrous convolutions capture multi-scale field patterns without losing resolution
- Auto-collected training data (LPIS labels) enables global generalization

---

## Geospatial Postprocessing

Raw raster predictions → clean vector polygons:
1. Threshold extent/boundary maps
2. Watershed segmentation (Higra library)
3. Polygon extraction + CRS assignment
4. Geometry cleanup: remove small polygons, fix invalid geometries, simplify
5. Export to GeoPackage (EPSG:4326)

---

## Dataset

Uses publicly available labeled field boundaries:
- **LPIS** (EU Land Parcel Identification System) — Europe
- **CDL** (USDA Cropland Data Layer) — USA

See `data/README.md` for download instructions.

---

## Background

This project is based on production experience building a global field segmentation system
that processed 35,000+ satellite image patches (~12M km²) across 11 Brazilian states
using Apache Airflow + AWS. The public version demonstrates the core ML pipeline
using open datasets.

---

## Tech Stack

`Python` `PyTorch` `uv` `ruff` `GDAL` `Rasterio` `GeoPandas` `Shapely` `Higra` `STAC` `Docker`
