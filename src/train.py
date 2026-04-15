"""Training loop for ResUNet-A field boundary segmentation.

Usage:
    python src/train.py --config configs/train.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.dataset import build_dataloader
from src.model import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_loss(outputs: dict, targets: dict, loss_weights: dict) -> torch.Tensor:
    """Combined multi-head loss (BCE + L1 for distance)."""
    loss_extent = F.binary_cross_entropy(
        outputs["extent"], targets["extent"]
    )
    loss_boundary = F.binary_cross_entropy(
        outputs["boundary"], targets["boundary"]
    )
    loss_distance = F.l1_loss(
        outputs["distance"], targets["distance"]
    )
    return (
        loss_weights["extent"] * loss_extent
        + loss_weights["boundary"] * loss_boundary
        + loss_weights["distance"] * loss_distance
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_weights: dict,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = {
            "extent": batch["extent"].to(device),
            "boundary": batch["boundary"].to(device),
            "distance": batch["distance"].to(device),
        }

        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, targets, loss_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return {"loss": total_loss / len(dataloader)}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and compute metrics."""
    model.eval()
    # TODO: Implement IoU, MCC, F1 computation
    return {"val_iou": 0.0, "val_mcc": 0.0, "val_f1": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResUNet-A")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg["seed"])

    # Build model
    model = build_model(in_channels=cfg["model"]["input_channels"])
    model = model.to(device)

    # Build dataloaders
    train_cfg = cfg["training"]
    train_loader = build_dataloader(
        cfg["data"]["train_dir"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        cfg["data"]["val_dir"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=False,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    # Logging
    writer = SummaryWriter(log_dir=cfg["logging"]["log_dir"])
    checkpoint_dir = Path(cfg["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training on {device} for {train_cfg['epochs']} epochs...")

    for epoch in range(1, train_cfg["epochs"] + 1):
        metrics = train_one_epoch(model, train_loader, optimizer, cfg["loss"], device)
        val_metrics = validate(model, val_loader, device)

        # Log
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        scheduler.step()
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}")

        # Save checkpoint
        if epoch % cfg["checkpoint"]["save_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
