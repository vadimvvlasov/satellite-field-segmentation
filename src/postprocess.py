"""Geospatial postprocessing: raster predictions → vector polygons.

Pipeline:
1. Threshold extent/boundary maps
2. Watershed segmentation (Higra)
3. Polygon extraction + CRS assignment
4. Geometry cleanup
5. Export to GeoPackage (EPSG:4326)
"""

from pathlib import Path

import geopandas as gpd
import numpy as np


def threshold_maps(
    extent: np.ndarray,
    boundary: np.ndarray,
    extent_thresh: float = 0.5,
    boundary_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholds to extent and boundary prediction maps."""
    extent_bin = (extent > extent_thresh).astype(np.uint8)
    boundary_bin = (boundary > boundary_thresh).astype(np.uint8)
    return extent_bin, boundary_bin


def watershed_segmentation(
    extent_bin: np.ndarray,
    boundary_bin: np.ndarray,
) -> np.ndarray:
    """Run watershed to separate touching field polygons.

    TODO: Integrate Higra watershed for proper field separation.
    Currently uses scipy.ndimage as placeholder.
    """
    from scipy.ndimage import label as nd_label

    # Placeholder: simple connected-component labeling
    # In production: use Higra watershed on distance transform
    labeled, _ = nd_label(extent_bin)
    return labeled


def polygons_from_labels(
    labeled: np.ndarray,
    transform: object | None = None,
    crs: str = "EPSG:4326",
    min_area: float = 1e-6,
) -> gpd.GeoDataFrame:
    """Extract polygons from labeled raster and build GeoDataFrame."""
    geometries = []

    for _idx in range(1, labeled.max() + 1):
        # TODO: use rasterio.features.shapes for proper polygonization
        # Placeholder: skip actual polygonization for stub
        pass

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    return gdf


def cleanup_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove invalid geometries, simplify, drop too-small polygons."""
    if gdf.empty:
        return gdf

    # Fix invalid geometries
    gdf["geometry"] = gdf.geometry.make_valid()

    # Simplify
    gdf["geometry"] = gdf.geometry.simplify(tolerance=1e-6)

    # Remove tiny polygons
    gdf["area"] = gdf.geometry.area
    gdf = gdf[gdf["area"] > gdf["area"].quantile(0.01)]
    gdf = gdf.drop(columns=["area"])

    return gdf.reset_index(drop=True)


def raster_to_geopackage(
    extent: np.ndarray,
    boundary: np.ndarray,
    output_path: str | Path,
    transform: object | None = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Full pipeline: raw prediction maps → GeoPackage file.

    Args:
        extent: Predicted extent map (H, W).
        boundary: Predicted boundary map (H, W).
        output_path: Destination .gpkg file path.
        transform: Rasterio Affine transform for georeferencing.
        crs: Output CRS.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: threshold
    extent_bin, boundary_bin = threshold_maps(extent, boundary)

    # Step 2: watershed
    labeled = watershed_segmentation(extent_bin, boundary_bin)

    # Step 3: polygons
    gdf = polygons_from_labels(labeled, transform=transform, crs=crs)

    # Step 4: cleanup
    gdf = cleanup_geometries(gdf)

    # Step 5: export
    gdf.to_file(output_path, driver="GPKG")
    print(f"Saved {len(gdf)} field polygons to {output_path}")

    return gdf
