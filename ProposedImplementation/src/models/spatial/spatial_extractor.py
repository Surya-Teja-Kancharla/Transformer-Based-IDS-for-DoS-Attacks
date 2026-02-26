"""
spatial_extractor.py  —  ProposedImplementation
================================================
Lightweight Spatial Feature Extractor using Depthwise Separable Convolutions.

MOTIVATION (Phase 2 contribution):
  The original Res-TranBiLSTM uses a full ResNet with 512 filters in the
  final layer.  For deployment on resource-constrained IoT/fog nodes, we
  replace it with a lightweight architecture using:
    1. Depthwise Separable Convolutions (DSConv) — factorises each 3×3 Conv
       into a depthwise + pointwise op, reducing parameters ~8–9× per layer.
    2. Inverted Residual Blocks (inspired by MobileNetV2) — keeps skip
       connections for gradient flow with far fewer parameters.
    3. Squeeze-and-Excitation (SE) channel attention — recovers accuracy
       lost from reduced capacity by adaptively re-weighting channels.

ARCHITECTURE:
  Input:  (B, 1, 28, 28)
  stem:   DSConv 3×3, 32ch, stride=1       → (B,  32, 28, 28)
  stage1: InvertedResidual 32→64, stride=2  → (B,  64, 14, 14)
  stage2: InvertedResidual 64→128, stride=2 → (B, 128,  7,  7)
  stage3: InvertedResidual 128→128, stride=2→ (B, 128,  4,  4)
  stage4: InvertedResidual 128→256, stride=2→ (B, 256,  2,  2)
  GlobalAvgPool → Flatten → FC(256→128) → ReLU → Dropout
  Output: (B, 128)   ← same as existing model, fully compatible

PARAMETER COMPARISON (approximate):
  Existing (ResNet):  ~11.2 M parameters in spatial branch
  Proposed (LightCNN): ~0.9 M parameters in spatial branch  (~92% reduction)

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: depthwise(3×3) + pointwise(1×1).
    Reduces parameters from C_in*C_out*k*k to C_in*k*k + C_in*C_out.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=padding, groups=in_ch, bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.relu(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block — channel attention with reduction ratio r.
    Recalibrates channel responses to recover accuracy lost from fewer params.
    """
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class InvertedResidualBlock(nn.Module):
    """
    Lightweight inverted residual block (MobileNetV2-style) + SE attention.

    Structure:
      expand (pointwise) → DSConv → SE → project (pointwise) → skip (if same dim)

    Parameters
    ----------
    in_ch   : input channels
    out_ch  : output channels
    stride  : spatial downsampling (1 or 2)
    expand  : expansion ratio for inner dimension
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand: int = 4,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand
        self.use_skip = (stride == 1 and in_ch == out_ch)

        layers = []

        # Expand (pointwise)
        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True),
            ]

        # Depthwise
        layers += [
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        ]

        # SE attention
        layers.append(SqueezeExcitation(mid_ch))

        # Project (pointwise, no activation — "linear bottleneck")
        layers += [
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]

        self.block = nn.Sequential(*layers)

        # Downsample for skip when channels differ
        if not self.use_skip and stride == 1:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip:
            return out + x
        elif self.skip_proj is not None:
            return out + self.skip_proj(x)
        return out


# ---------------------------------------------------------------------------
# Full Lightweight Spatial Extractor
# ---------------------------------------------------------------------------

class LightweightSpatialExtractor(nn.Module):
    """
    Lightweight spatial feature extractor for ProposedImplementation.

    Produces same (B, 128) output as the existing ResNet branch,
    so the classifier head is 100% compatible.

    Parameters
    ----------
    in_channels : int   — 1 for grayscale
    output_dim  : int   — 128 (matches existing model)
    dropout     : float — 0.5
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Stem: standard DSConv (not inverted residual for first layer)
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32, stride=1),  # (B, 32, 28, 28)
        )

        # Stages: progressive channel expansion + spatial downsampling
        self.stage1 = InvertedResidualBlock(32,  64,  stride=2, expand=4)   # (B, 64,  14, 14)
        self.stage2 = InvertedResidualBlock(64,  128, stride=2, expand=4)   # (B, 128,  7,  7)
        self.stage3 = InvertedResidualBlock(128, 128, stride=2, expand=4)   # (B, 128,  4,  4)
        self.stage4 = InvertedResidualBlock(128, 256, stride=2, expand=4)   # (B, 256,  2,  2)

        # Global pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d(1)   # (B, 256, 1, 1)
        self.flatten = nn.Flatten()          # (B, 256)
        self.fc = nn.Linear(256, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, 28, 28)

        Returns
        -------
        (B, 128)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_output_dim(self) -> int:
        return self.output_dim

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = LightweightSpatialExtractor()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")          # (4, 128)
    print(f"Params: {model.count_parameters():,}")