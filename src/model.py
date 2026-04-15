"""ResUNet-A model for field boundary segmentation.

Residual U-Net with Atrous Convolutions.
Multi-head output:
  - Extent map  (field interior)
  - Boundary map (field edges)
  - Distance transform (distance to nearest boundary)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class AtrousBlock(nn.Module):
    """Atrous (dilated) convolution block for multi-scale context."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        rates = [1, 2, 4, 8]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.fusion = nn.Conv2d(out_channels * len(rates), out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.convs]
        fused = torch.cat(features, dim=1)
        return self.fusion(fused)


class ResUNet_A(nn.Module):
    """ResUNet-A with multi-head output.

    Architecture:
        Encoder: Residual blocks + downsampling
        Bridge: Atrous Spatial Pyramid Pooling
        Decoder: Upsampling + skip connections
        Heads: extent, boundary, distance

    Args:
        in_channels: Number of input bands (default 4: RGB + NIR).
        backbone_depths: Number of residual blocks per encoder stage.
        num_filters: Base number of filters.
    """

    def __init__(
        self,
        in_channels: int = 4,
        backbone_depths: list[int] | None = None,
        num_filters: int = 32,
    ) -> None:
        super().__init__()
        if backbone_depths is None:
            backbone_depths = [2, 2, 2, 2]

        # Encoder
        self.encoder = nn.ModuleList()
        for i, depth in enumerate(backbone_depths):
            in_ch = in_channels if i == 0 else num_filters * (2 ** (i - 1))
            out_ch = num_filters * (2**i)
            blocks = []
            for j in range(depth):
                blocks.append(ResidualBlock(
                    in_ch if j == 0 else out_ch, out_ch,
                    stride=2 if j == 0 else 1,
                ))
            self.encoder.append(nn.Sequential(*blocks))

        # Bridge — ASPP
        bridge_in = num_filters * (2 ** (len(backbone_depths) - 1))
        self.bridge = AtrousBlock(bridge_in, num_filters * 4)

        # Decoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(backbone_depths))):
            skip_ch = num_filters * (2 ** i)
            dec_in = num_filters * (2 ** (i + 1))
            if i == len(backbone_depths) - 1:
                dec_in = num_filters * 4
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(dec_in, skip_ch, 2, stride=2),
                ResidualBlock(skip_ch * 2, skip_ch),
            ))

        # Output heads
        self.head_extent = nn.Conv2d(num_filters, 1, 1)
        self.head_boundary = nn.Conv2d(num_filters, 1, 1)
        self.head_distance = nn.Conv2d(num_filters, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encoder pass
        encoder_outs = []
        for block in self.encoder:
            x = block(x)
            encoder_outs.append(x)

        # Bridge
        x = self.bridge(x)

        # Decoder with skip connections
        for i, dec_block in enumerate(self.decoder):
            skip = encoder_outs[-(i + 1)]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        return {
            "extent": torch.sigmoid(self.head_extent(x)),
            "boundary": torch.sigmoid(self.head_boundary(x)),
            "distance": torch.relu(self.head_distance(x)),
        }


def build_model(in_channels: int = 4, **kwargs) -> ResUNet_A:
    """Factory function to build the model."""
    return ResUNet_A(in_channels=in_channels, **kwargs)
