"""CLI entry point for inference.

Usage:
    python predict.py --input data/sample_tile.tif --output output/fields.gpkg
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load ResUNet-A model from checkpoint."""
    from src.model import build_model

    model = build_model(in_channels=4)
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}, using untrained model.")
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    tile_size: int = 256,
) -> None:
    """Run inference on a single GeoTIFF tile and save as GeoPackage."""
    from src.postprocess import raster_to_geopackage

    input_path = Path(input_path)
    output_path = Path(output_path)

    with rasterio.open(input_path) as ds:
        transform = ds.transform
        crs = ds.crs
        image = ds.read()

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Pad or crop to tile_size
    _, h, w = image.shape
    if h != tile_size or w != tile_size:
        padded = np.zeros((image.shape[0], tile_size, tile_size), dtype=np.float32)
        padded[:, :min(h, tile_size), :min(w, tile_size)] = image[
            :, :min(h, tile_size), :min(w, tile_size)
        ]
        image = padded

    # Inference
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)

    extent = outputs["extent"].squeeze().cpu().numpy()
    boundary = outputs["boundary"].squeeze().cpu().numpy()

    # Postprocess → GeoPackage
    raster_to_geopackage(
        extent=extent,
        boundary=boundary,
        output_path=output_path,
        transform=transform,
        crs=str(crs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict field boundaries from satellite imagery")
    parser.add_argument("--input", required=True, help="Input GeoTIFF tile path")
    parser.add_argument("--output", required=True, help="Output GeoPackage path")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Model checkpoint path (default: checkpoints/best.pt)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    print(f"Running inference on {args.input} → {args.output}")
    run_inference(model, args.input, args.output, device)
    print("Done.")


if __name__ == "__main__":
    main()
