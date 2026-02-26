"""
spatial_extractor.py
====================
Full ResNet-based Spatial Feature Extraction Module.

Paper (Wang et al., 2023) Table 6 — exact architecture:
  Layer     | Output Size | Content
  ----------|-------------|--------------------------------------------
  conv1     | 28×28       | 3×3, 64, stride=1
  conv2_x   | 14×14       | 3×3 maxpool stride=1; [3×3,64; 3×3,64]×2
  conv3_x   |  7×7        | [3×3,128; 3×3,128] × 2
  conv4_x   |  4×4        | [3×3,256; 3×3,256] × 2
  conv5_x   |  2×2        | [3×3,512; 3×3,512] × 2
  FC-1(128) |  output dim 128

Input:  (B, 1, 28, 28)  — grayscale 28×28 image (bicubic upsampled from 8×8)
Output: (B, 128)        — spatial feature vector

Author: FYP Implementation
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .resnet_block import ResidualBlock, ResidualLayer


class SpatialFeatureExtractor(nn.Module):
    """
    ResNet spatial branch of Res-TranBiLSTM.

    Faithfully implements Table 6 from the paper:
      - Pre-conv: 3×3, 64 channels, stride=1, padding=1
      - conv2_x: MaxPool then 2 residual blocks of [64,64]  → 14×14
      - conv3_x: 2 residual blocks of [128,128], stride=2   →  7×7
      - conv4_x: 2 residual blocks of [256,256], stride=2   →  4×4
      - conv5_x: 2 residual blocks of [512,512], stride=2   →  2×2
      - Global MaxPool → Flatten → FC(512 → 128) → output

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale image).
    output_dim : int
        Spatial feature vector dimension (128 per paper FC-1).
    dropout : float
        Dropout rate applied after FC layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # -- conv1: pre-convolution layer (28×28 → 28×28) --
        # Paper: "3×3, 64, stride:1"
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -- conv2_x: maxpool + 2 residual blocks, [64,64]×2 (28→14) --
        # Paper: "3×3 max pooling, stride:1" followed by residual blocks
        # The maxpool halves spatial dim: 28×28 → 14×14
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = ResidualLayer(64, 64, num_blocks=2, stride=1)

        # -- conv3_x: [128,128]×2, stride=2 (14→7) --
        self.conv3_x = ResidualLayer(64, 128, num_blocks=2, stride=2)

        # -- conv4_x: [256,256]×2, stride=2 (7→4) --
        self.conv4_x = ResidualLayer(128, 256, num_blocks=2, stride=2)

        # -- conv5_x: [512,512]×2, stride=2 (4→2) --
        self.conv5_x = ResidualLayer(256, 512, num_blocks=2, stride=2)

        # -- Global max pooling → Flatten → FC-1(128) --
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))   # (B,512,1,1)
        self.fc = nn.Sequential(
            nn.Flatten(),                        # (B, 512)
            nn.Linear(512, output_dim),          # (B, 128)
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Weight initialization (He initialization for conv layers)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 28, 28)
            Batch of grayscale 28×28 network traffic images.

        Returns
        -------
        torch.Tensor, shape (B, 128)
            Spatial feature vectors.
        """
        # Pre-conv
        out = self.conv1(x)            # (B, 64, 28, 28)

        # conv2_x: pool first, then residual
        out = self.maxpool(out)        # (B, 64, 14, 14)
        out = self.conv2_x(out)        # (B, 64, 14, 14)

        # conv3_x
        out = self.conv3_x(out)        # (B, 128,  7,  7)

        # conv4_x
        out = self.conv4_x(out)        # (B, 256,  4,  4)

        # conv5_x
        out = self.conv5_x(out)        # (B, 512,  2,  2)

        # Global pool + FC
        out = self.global_pool(out)    # (B, 512,  1,  1)
        out = self.fc(out)             # (B, 128)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def get_output_dim(self) -> int:
        return self.output_dim

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SpatialFeatureExtractor(in_channels=1, output_dim=128)
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # expect (8, 128)
    print(f"Params: {model.count_parameters():,}")