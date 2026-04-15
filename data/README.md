# Dataset

## Sources

### LPIS (EU Land Parcel Identification System)
- **Region:** Europe
- **Format:** GeoPackage / Shapefile
- **License:** Open data (EU member states)
- **Download:** <https://data.europa.eu/>

### CDL (USDA Cropland Data Layer)
- **Region:** USA (CONUS)
- **Format:** GeoTIFF
- **License:** Public domain
- **Download:** <https://nassgeodata.gmu.edu/CropScape/>

### Sentinel-2 Imagery
- **Source:** STAC catalog (AWS, Google Earth Engine, Copernicus Data Space)
- **Product:** L2A (surface reflectance)
- **Bands used:** B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)

## Download script

Run `bash data/download.sh` to fetch sample data.
The script downloads a small sample tile and label for testing the pipeline.

## Data layout

After running the download script:

```
data/
├── raw/
│   ├── images/        # Sentinel-2 tiles (.tif)
│   └── labels/        # LPIS / CDL field boundaries (.gpkg / .tif)
├── processed/
│   ├── train/         # Training splits
│   └── val/           # Validation splits
└── sample_tile.tif    # Quick example for predict.py
```
