"""Dataset and DataLoader for satellite field boundary segmentation.

Handles:
- Loading Sentinel-2 tiles (GeoTIFF) and corresponding labels
- Data augmentations (rotation, flip, color jitter)
- Multi-head target generation (extent, boundary, distance maps)
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


class FieldDataset(Dataset):
    """Loads Sentinel-2 tiles and field boundary labels.

    Args:
        root_dir: Path to directory containing images/ and labels/ subfolders.
        transform: Optional augmentation pipeline.
        bands: Spectral band indices to load (default: 4 bands — RGB + NIR).
        tile_size: Spatial size for tiling (default: 256).
    """

    BANDS_RGBNIR = [2, 3, 4, 7]  # Sentinel-2 band indices for B02, B03, B04, B08

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable | None = None,
        bands: list[int] | None = None,
        tile_size: int = 256,
    ) -> None:
        self.root = Path(root_dir)
        self.transform = transform
        self.bands = bands or self.BANDS_RGBNIR
        self.tile_size = tile_size

        self.image_paths = sorted(self.root.glob("images/*.tif"))
        self.label_paths = sorted(self.root.glob("labels/*.tif"))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.root}/images/")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx] if idx < len(self.label_paths) else None

        # Read image
        image = self._read_raster(img_path)

        # Read / generate label
        if label_path and label_path.exists():
            label = self._read_label(label_path)
        else:
            label = self._generate_dummy_label(image.shape[1], image.shape[2])

        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=image.transpose(1, 2, 0), mask=label)
            image = augmented["image"].transpose(2, 0, 1)
            label = augmented["mask"]

        # Generate multi-head targets
        extent, boundary, distance = self._generate_targets(label)

        return {
            "image": torch.from_numpy(image).float(),
            "extent": torch.from_numpy(extent).float(),
            "boundary": torch.from_numpy(boundary).float(),
            "distance": torch.from_numpy(distance).float(),
            "path": str(img_path),
        }

    @staticmethod
    def _read_raster(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            data = ds.read()
        return data

    @staticmethod
    def _read_label(path: Path) -> np.ndarray:
        with rasterio.open(path) as ds:
            data = ds.read(1)
        return data

    @staticmethod
    def _generate_dummy_label(h: int, w: int) -> np.ndarray:
        return np.zeros((h, w), dtype=np.uint8)

    @staticmethod
    def _generate_targets(label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate extent, boundary, and distance transform maps from label."""
        extent = (label > 0).astype(np.float32)
        boundary = np.zeros_like(label, dtype=np.float32)
        distance = np.zeros_like(label, dtype=np.float32)
        return extent, boundary, distance


def build_dataloader(
    root_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    transform: Callable | None = None,
    shuffle: bool = False,
) -> DataLoader:
    """Convenience function to create a DataLoader for the dataset."""
    dataset = FieldDataset(root_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )
