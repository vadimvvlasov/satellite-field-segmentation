#!/usr/bin/env bash
# Download sample Sentinel-2 tile and field boundary labels for testing.
# Usage: bash data/download.sh

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="${DATA_DIR}/raw"
SAMPLE="${DATA_DIR}/sample_tile.tif"

mkdir -p "${RAW_DIR}/images" "${RAW_DIR}/labels"

echo "==> Downloading sample data..."

# TODO: Replace with actual STAC-based download or direct URLs.
# Example placeholder — download a small test tile.
# In production this would query a STAC catalog (e.g. planetarycomputer.microsoft.com)
# and fetch Sentinel-2 L2A tiles + corresponding LPIS/CDL labels.

# Placeholder: create a dummy file so predict.py can run in demo mode
if [ ! -f "${SAMPLE}" ]; then
    echo "  No real data source configured. Creating placeholder file."
    echo "  Replace this script section with actual download URLs."
    python3 -c "
import numpy as np, rasterio
from rasterio.transform import from_origin

# Dummy 256x256 4-band tile (RGB+NIR)
data = np.random.randint(0, 255, (4, 256, 256), dtype=np.uint8)
with rasterio.open(
    '${SAMPLE}', 'w',
    driver='GTiff', height=256, width=256,
    count=4, dtype='uint8',
    crs='EPSG:4326',
    transform=from_origin(-54.0, -4.0, 10.0, 10.0)
) as ds:
    ds.write(data)
print(f'  Created placeholder: ${SAMPLE}')
"
fi

echo "==> Done. Data layout:"
ls -lh "${DATA_DIR}"
