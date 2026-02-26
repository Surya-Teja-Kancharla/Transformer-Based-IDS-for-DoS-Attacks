"""
resnet_block.py
===============
Implements the residual building blocks for the Spatial Feature Extraction
Module in Res-TranBiLSTM.

Paper reference (Wang et al., 2023):
  - Section 3.3 and Fig. 3: "a residual block, a batch normalization (BN)
    layer is added after each convolution layer"
  - Table 6: 3×3 conv kernels throughout, 4 residual blocks (conv2_x to
    conv5_x), each block has 2 conv layers → 16 total conv layers
  - Eq.(3): X[l+c] = f(h(X[l]) + F(x))
             F(x) = sum_{i=l}^{l+c-1} F(xi)

Architecture per Table 6:
  conv1:   28×28, [3×3, 64, stride=1]
  conv2_x: 14×14, [3×3, 64; 3×3, 64] × 2  (with 3×3 maxpool stride=1)
  conv3_x:  7×7,  [3×3, 128; 3×3, 128] × 2
  conv4_x:  4×4,  [3×3, 256; 3×3, 256] × 2
  conv5_x:  2×2,  [3×3, 512; 3×3, 512] × 2
  FC-1(128) → spatial feature vector

Author: FYP Implementation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# Basic Residual Block (He et al., 2016)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Basic residual block with two Conv-BN-ReLU layers and a skip connection.

    Structure (per paper Fig. 3):
        Input x
          │
          ├──→ Conv(3×3) → BN → ReLU → Conv(3×3) → BN → F(x)
          │
          └──→ (identity or projection shortcut) → h(x)
          │
          └──→ ReLU(F(x) + h(x)) → Output

    For same-dimension skip: shortcut = identity (no parameters)
    For dimension change: shortcut = 1×1 Conv → BN (projection)

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int
        Stride for the first conv layer. stride=2 halves spatial dims.
    """

    expansion = 1  # BasicBlock has no bottleneck expansion

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        # -- Main path: Conv-BN-ReLU → Conv-BN --
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # -- Skip connection (shortcut / projection) --
        # If dimensions change (stride ≠ 1 or channel count changes),
        # use a 1×1 conv projection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Eq.(3) from the paper:
            X[l+c] = f(h(X[l]) + F(x))

        Parameters
        ----------
        x : torch.Tensor, shape (B, C_in, H, W)

        Returns
        -------
        torch.Tensor, shape (B, C_out, H', W')
        """
        # Main path: F(x) — residual mapping
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # Skip connection: h(x)
        identity = self.shortcut(x)

        # Combine: X[l+c] = f(F(x) + h(X[l]))
        out = self.relu(residual + identity)
        return out


# ---------------------------------------------------------------------------
# Residual Layer (stack of N residual blocks)
# ---------------------------------------------------------------------------

class ResidualLayer(nn.Module):
    """
    A stack of N basic residual blocks with the same output channel count.

    The first block handles stride/channel transitions;
    subsequent blocks use stride=1 and same channels.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    num_blocks : int
        Number of residual blocks in this layer.
    stride : int
        Stride for the FIRST block only.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        stride: int = 1,
    ) -> None:
        super().__init__()

        layers = [
            ResidualBlock(in_channels, out_channels, stride=stride)
        ]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test basic block
    block = ResidualBlock(64, 64, stride=1)
    x = torch.randn(4, 64, 14, 14)
    out = block(x)
    print(f"ResidualBlock (same dim)  : {x.shape} → {out.shape}")

    # Test with dimension change
    block2 = ResidualBlock(64, 128, stride=2)
    x2 = torch.randn(4, 64, 14, 14)
    out2 = block2(x2)
    print(f"ResidualBlock (dim change) : {x2.shape} → {out2.shape}")

    # Test layer
    layer = ResidualLayer(64, 128, num_blocks=2, stride=2)
    x3 = torch.randn(4, 64, 14, 14)
    out3 = layer(x3)
    print(f"ResidualLayer              : {x3.shape} → {out3.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"ResidualBlock parameters   : {total_params:,}")