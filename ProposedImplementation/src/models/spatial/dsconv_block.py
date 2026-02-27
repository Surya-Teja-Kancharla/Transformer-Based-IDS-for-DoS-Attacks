"""
dsconv_block.py
===============
MobileNetV2 and EfficientNet-B0 building blocks for the Phase 2
Ensemble Spatial Feature Extractor (EnsembleSpatialExtractor).

Phase 2 replacement for ExistingImplementation/src/models/spatial/resnet_block.py.

PHASE 1 vs PHASE 2 SPATIAL BUILDING BLOCKS:
  +---------------------------+--------------------------------------------+
  | Phase 1 (resnet_block)    | Phase 2 (dsconv_block)                     |
  +---------------------------+--------------------------------------------+
  | nn.Conv2d (standard 3x3)  | MBConvBlock (depthwise separable + expand) |
  | (no equivalent)           | SqueezeExcitation (SE channel attention)   |
  | ResidualBlock             | MBConvBlock with skip connection           |
  | ResidualLayer             | MobileNetV2Branch / EfficientNetB0Branch   |
  +---------------------------+--------------------------------------------+

28x28 ADAPTATION (critical implementation detail per New_Idea.txt lines 269-274):
  Standard MobileNetV2 and EfficientNet-B0 are designed for 224x224 input and
  use stride-2 in their stem convolution, which would immediately halve 28x28
  feature maps before any meaningful features are learned.

  Modifications applied to both branches:
    1. Stem convolution: stride=1 (was stride=2) -- preserves 28x28 spatial dims
    2. First MBConv stages: stride=1 (was stride=2) -- defers early downsampling
    3. Later stages: stride=2 applied selectively (28->14->7->4 where needed)
    4. No early pooling that would collapse small feature maps

  This mirrors exactly what the paper (Wang et al. 2023) did for ResNet-18:
  "conv1: 3x3 filter, stride=1; conv2_x: 3x3 max pooling, stride=1"
  (New_Idea.txt lines 108-113).

BUILDING BLOCKS:

  SqueezeExcitation (SE):
    EfficientNet-B0 uses SE by design (Tan & Le, 2019).
    MobileNetV2 does not use SE by default (Sandler et al., 2018).
    Per New_Idea.txt line 247: the two models must have distinct inductive
    biases -- MobileNetV2 captures local spatial patterns, EfficientNetB0
    captures channel-wise relationships via SE. Therefore SE is enabled
    only in EfficientNetB0Branch, keeping architectures distinct.

  MBConvBlock (Mobile Inverted Bottleneck Convolution):
    Shared building block for both architectures.
    Structure: [expand(1x1) ->] dw(3x3) -> [SE ->] project(1x1) -> [+skip]
    - MobileNetV2Branch: use_se=False, activation=relu6
    - EfficientNetB0Branch: use_se=True,  activation=silu

  MobileNetV2Branch  (~700K params adapted for 28x28):
    Depthwise separable convs, no SE, ReLU6 activation.

  EfficientNetB0Branch (~1.0M params adapted for 28x28):
    Depthwise separable convs + SE blocks, SiLU activation.

PARAMETER COMPARISON (spatial branch total):
  Phase 1 ResNet-18 spatial:      ~11,200,000 parameters
  Phase 2 MobileNetV2 branch:        ~700,000 parameters
  Phase 2 EfficientNetB0 branch:   ~1,000,000 parameters
  Phase 2 Ensemble combined:       ~1,700,000 parameters  (~85% reduction)
  Combined still lighter than one ResNet-18 (New_Idea.txt lines 240-245).

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation Block
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation channel attention block (Hu et al., 2018).

    Used by EfficientNetB0Branch to capture channel-wise relationships
    (New_Idea.txt line 247: "EfficientNet-B0 captures channel-wise
    relationships via SE attention blocks").

    NOT used by MobileNetV2Branch to maintain architectural diversity
    between the two ensemble members (New_Idea.txt line 248:
    "These two models look at the same 28x28 traffic matrix differently
    -- exactly the diversity needed for ensemble to work").

    Parameters
    ----------
    channels  : int -- number of channels to recalibrate
    reduction : int -- squeeze ratio for bottleneck FC (EfficientNet uses 4)
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc1     = nn.Linear(channels, mid, bias=True)
        self.act     = nn.SiLU(inplace=True)   # SiLU (Swish) matches EfficientNet
        self.fc2     = nn.Linear(mid, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: (B, C, H, W) -> (B, C)
        scale = self.pool(x).view(x.size(0), -1)
        # Excite: FC -> SiLU -> FC -> Sigmoid
        scale = self.sigmoid(self.fc2(self.act(self.fc1(scale))))
        # Scale: broadcast back to (B, C, H, W)
        return x * scale.view(x.size(0), x.size(1), 1, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# MBConv Block (Mobile Inverted Bottleneck Convolution)
# ---------------------------------------------------------------------------

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution block.

    Shared building block for MobileNetV2Branch and EfficientNetB0Branch.

    Differs from Phase 1's ResidualBlock:
      Phase 1 ResidualBlock: conv(3x3)->BN->ReLU->conv(3x3)->BN + skip
      MBConvBlock:           expand(1x1)->[dw(3x3)]->[SE]->project(1x1) + skip

    "Inverted" because channels are EXPANDED in the middle (not compressed
    like in Phase 1's bottleneck-style residual blocks).

    Skip connection rule (standard MobileNetV2/EfficientNet):
      stride=1 AND in_ch==out_ch -> identity skip (same as Phase 1)
      otherwise                  -> no skip

    Parameters
    ----------
    in_ch        : int   -- input channels
    out_ch       : int   -- output channels
    stride       : int   -- spatial stride (1 = same size, 2 = halve)
    expand       : int   -- channel expansion multiplier (default 6)
    use_se       : bool  -- include SE block (True for EfficientNet, False for MobileNetV2)
    se_reduction : int   -- SE reduction ratio (4 for EfficientNet-B0)
    activation   : str   -- 'relu6' for MobileNetV2, 'silu' for EfficientNet
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand: int = 6,
        use_se: bool = False,
        se_reduction: int = 4,
        activation: str = "relu6",
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand
        self.use_skip = (stride == 1 and in_ch == out_ch)

        def _act() -> nn.Module:
            return nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU6(inplace=True)

        layers: list = []

        # Expand: pointwise 1x1 to increase channel depth (skip if expand==1)
        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                _act(),
            ]

        # Depthwise: 3x3 spatial filtering (groups=mid_ch makes it channel-wise)
        layers += [
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, stride=stride,
                padding=1, groups=mid_ch, bias=False,
            ),
            nn.BatchNorm2d(mid_ch),
            _act(),
        ]

        # SE channel attention (EfficientNet only -- kept absent for MobileNetV2
        # to maintain the architectural diversity required by New_Idea.txt line 247-248)
        if use_se:
            layers.append(SqueezeExcitation(mid_ch, reduction=se_reduction))

        # Project: pointwise 1x1, NO activation (linear bottleneck per MobileNetV2)
        layers += [
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip:
            return out + x    # identity skip (stride=1, same channels)
        return out            # no skip when stride=2 or channels change

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# MobileNetV2 Branch (adapted for 28x28 input)
# ---------------------------------------------------------------------------

class MobileNetV2Branch(nn.Module):
    """
    MobileNetV2-style spatial branch adapted for 28x28 input.

    New_Idea.txt (lines 236, 247):
      "MobileNetV2 (~2.3M) captures local spatial patterns via
       depthwise separable convolutions"

    Standard MobileNetV2 (Sandler et al., 2018) bottleneck schedule:
      Operator     | c   | n | s
      Conv2d 3x3   | 32  | 1 | 2   <- stride=1 for 28x28
      MBConv1 3x3  | 16  | 1 | 1
      MBConv6 3x3  | 24  | 2 | 2   <- stride=1 for 28x28
      MBConv6 3x3  | 32  | 3 | 2
      MBConv6 3x3  | 64  | 4 | 2
      MBConv6 3x3  | 96  | 3 | 1
      MBConv6 3x3  | 160 | 3 | 2
      MBConv6 3x3  | 320 | 1 | 1
      Conv2d 1x1   | 1280| 1 | 1

    Adapted spatial map progression for 28x28:
      28x28 -> stem(s=1) -> 28x28 -> s1(s=1) -> 28x28 -> s2(s=1) ->
      28x28 -> s3(s=2) -> 14x14 -> s4(s=2) -> 7x7 -> s5(s=2) -> 4x4 ->
      s6(s=1) -> 4x4 -> head_conv -> pool -> (B,256,1,1) -> fc -> (B,128)

    Parameters
    ----------
    in_channels : int   -- input image channels (1 for grayscale)
    output_dim  : int   -- output feature vector dimension (128)
    dropout     : float -- dropout after final FC
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Stem: Conv2d(1->32, 3x3, stride=1) [standard uses stride=2]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # MBConv stages (t=expand, c=out_ch, n=blocks, s=stride)
        # stage1: t=1, c=16,  n=1, s=1  -> (B, 16, 28, 28)
        self.stage1 = self._make_stage(32,  16,  n=1, stride=1, expand=1)
        # stage2: t=6, c=24,  n=2, s=1  -> (B, 24, 28, 28)  [standard s=2 -> s=1]
        self.stage2 = self._make_stage(16,  24,  n=2, stride=1, expand=6)
        # stage3: t=6, c=32,  n=3, s=2  -> (B, 32, 14, 14)
        self.stage3 = self._make_stage(24,  32,  n=3, stride=2, expand=6)
        # stage4: t=6, c=64,  n=2, s=2  -> (B, 64,  7,  7)
        self.stage4 = self._make_stage(32,  64,  n=2, stride=2, expand=6)
        # stage5: t=6, c=96,  n=2, s=2  -> (B, 96,  4,  4)
        self.stage5 = self._make_stage(64,  96,  n=2, stride=2, expand=6)
        # stage6: t=6, c=160, n=1, s=1  -> (B, 160, 4,  4)  [standard s=2 -> s=1]
        self.stage6 = self._make_stage(96,  160, n=1, stride=1, expand=6)

        # Head: Conv(160->256, 1x1) -> GlobalAvgPool -> FC(256->128)
        self.head_conv = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self._initialize_weights()

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n: int,
        stride: int,
        expand: int,
    ) -> nn.Sequential:
        """Build n MBConvBlocks. First handles stride; rest use stride=1."""
        layers = [
            MBConvBlock(in_ch, out_ch, stride=stride,
                        expand=expand, use_se=False, activation="relu6")
        ]
        for _ in range(1, n):
            layers.append(
                MBConvBlock(out_ch, out_ch, stride=1,
                            expand=expand, use_se=False, activation="relu6")
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 28, 28)

        Returns
        -------
        torch.Tensor, shape (B, 128)
        """
        out = self.stem(x)           # (B, 32, 28, 28)
        out = self.stage1(out)       # (B, 16, 28, 28)
        out = self.stage2(out)       # (B, 24, 28, 28)
        out = self.stage3(out)       # (B, 32, 14, 14)
        out = self.stage4(out)       # (B, 64,  7,  7)
        out = self.stage5(out)       # (B, 96,  4,  4)
        out = self.stage6(out)       # (B, 160, 4,  4)
        out = self.head_conv(out)    # (B, 256, 4,  4)
        out = self.global_pool(out)  # (B, 256, 1,  1)
        out = self.fc(out)           # (B, 128)
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


# ---------------------------------------------------------------------------
# EfficientNet-B0 Branch (adapted for 28x28 input)
# ---------------------------------------------------------------------------

class EfficientNetB0Branch(nn.Module):
    """
    EfficientNet-B0-style spatial branch adapted for 28x28 input.

    New_Idea.txt (lines 238, 247):
      "EfficientNet-B0 (~5.3M) captures channel-wise relationships
       via SE attention blocks"

    Standard EfficientNet-B0 (Tan & Le, 2019) MBConv schedule:
      Stage | Operator       | c   | n | s
      1     | Conv2d 3x3     | 32  | 1 | 2  <- stride=1 for 28x28
      2     | MBConv1 3x3    | 16  | 1 | 1
      3     | MBConv6 3x3    | 24  | 2 | 2  <- stride=1 for 28x28
      4     | MBConv6 3x3    | 40  | 2 | 2
      5     | MBConv6 3x3    | 80  | 3 | 2
      6     | MBConv6 3x3    | 112 | 3 | 1
      7     | MBConv6 3x3    | 192 | 4 | 2  <- stride=1 for 28x28
      8     | MBConv6 3x3    | 320 | 1 | 1
      9     | Conv2d 1x1     |1280 | 1 | 1

    All stages use SE with reduction=4 (EfficientNet-B0 default = se_ratio 0.25).

    Adapted spatial map progression for 28x28:
      28x28 -> stem(s=1) -> 28x28 -> s1(s=1) -> 28x28 -> s2(s=1) ->
      28x28 -> s3(s=2) -> 14x14 -> s4(s=2) -> 7x7 -> s5(s=1) -> 7x7 ->
      s6(s=1) -> 7x7 -> head_conv -> pool -> (B,256,1,1) -> fc -> (B,128)

    Parameters
    ----------
    in_channels : int   -- input image channels (1 for grayscale)
    output_dim  : int   -- output feature vector dimension (128)
    dropout     : float -- dropout after final FC
    """

    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Stem: Conv2d(1->32, 3x3, stride=1) [standard uses stride=2]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # MBConv stages with SE (EfficientNet always uses SE).
        # stage1: t=1, c=16,  n=1, s=1  -> (B, 16, 28, 28)
        self.stage1 = self._make_stage(32,  16,  n=1, stride=1, expand=1)
        # stage2: t=6, c=24,  n=2, s=1  -> (B, 24, 28, 28)  [standard s=2 -> s=1]
        self.stage2 = self._make_stage(16,  24,  n=2, stride=1, expand=6)
        # stage3: t=6, c=40,  n=2, s=2  -> (B, 40, 14, 14)
        self.stage3 = self._make_stage(24,  40,  n=2, stride=2, expand=6)
        # stage4: t=6, c=80,  n=2, s=2  -> (B, 80,  7,  7)
        self.stage4 = self._make_stage(40,  80,  n=2, stride=2, expand=6)
        # stage5: t=6, c=112, n=2, s=1  -> (B, 112, 7,  7)
        self.stage5 = self._make_stage(80,  112, n=2, stride=1, expand=6)
        # stage6: t=6, c=192, n=2, s=1  -> (B, 192, 7,  7)  [standard s=2 -> s=1]
        self.stage6 = self._make_stage(112, 192, n=2, stride=1, expand=6)

        # Head: Conv(192->256, 1x1) -> GlobalAvgPool -> FC(256->128)
        self.head_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self._initialize_weights()

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n: int,
        stride: int,
        expand: int,
    ) -> nn.Sequential:
        """Build n MBConvBlocks with SE. First block handles stride; rest stride=1."""
        layers = [
            MBConvBlock(
                in_ch, out_ch, stride=stride,
                expand=expand, use_se=True, se_reduction=4, activation="silu",
            )
        ]
        for _ in range(1, n):
            layers.append(
                MBConvBlock(
                    out_ch, out_ch, stride=1,
                    expand=expand, use_se=True, se_reduction=4, activation="silu",
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, 28, 28)

        Returns
        -------
        torch.Tensor, shape (B, 128)
        """
        out = self.stem(x)           # (B, 32, 28, 28)
        out = self.stage1(out)       # (B, 16, 28, 28)
        out = self.stage2(out)       # (B, 24, 28, 28)
        out = self.stage3(out)       # (B, 40, 14, 14)
        out = self.stage4(out)       # (B, 80,  7,  7)
        out = self.stage5(out)       # (B, 112, 7,  7)
        out = self.stage6(out)       # (B, 192, 7,  7)
        out = self.head_conv(out)    # (B, 256, 7,  7)
        out = self.global_pool(out)  # (B, 256, 1,  1)
        out = self.fc(out)           # (B, 128)
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


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    B = 4
    x = torch.randn(B, 1, 28, 28)

    # MBConvBlock tests
    print("=== MBConvBlock ===")
    block_same = MBConvBlock(32, 32, stride=1, expand=6, use_se=False)
    out = block_same(torch.randn(B, 32, 14, 14))
    print(f"  MBConv(32->32, s=1, SE=F): {(B,32,14,14)} -> {tuple(out.shape)}"
          f"  skip={block_same.use_skip}")

    block_se = MBConvBlock(40, 40, stride=1, expand=6, use_se=True)
    out = block_se(torch.randn(B, 40, 14, 14))
    print(f"  MBConv(40->40, s=1, SE=T): {(B,40,14,14)} -> {tuple(out.shape)}"
          f"  skip={block_se.use_skip}")

    block_down = MBConvBlock(32, 64, stride=2, expand=6, use_se=False)
    out = block_down(torch.randn(B, 32, 14, 14))
    print(f"  MBConv(32->64, s=2, SE=F): {(B,32,14,14)} -> {tuple(out.shape)}"
          f"  skip={block_down.use_skip}")

    # MobileNetV2Branch
    print("\n=== MobileNetV2Branch ===")
    mv2 = MobileNetV2Branch(in_channels=1, output_dim=128)
    mv2.eval()
    out_mv2 = mv2(x)
    print(f"  Input:   {tuple(x.shape)}")
    print(f"  Output:  {tuple(out_mv2.shape)}")   # expect (4, 128)
    print(f"  Params:  {mv2.count_parameters():,}")

    # EfficientNetB0Branch
    print("\n=== EfficientNetB0Branch ===")
    efn = EfficientNetB0Branch(in_channels=1, output_dim=128)
    efn.eval()
    out_efn = efn(x)
    print(f"  Input:   {tuple(x.shape)}")
    print(f"  Output:  {tuple(out_efn.shape)}")   # expect (4, 128)
    print(f"  Params:  {efn.count_parameters():,}")

    # Combined vs Phase 1 comparison
    total_p2 = mv2.count_parameters() + efn.count_parameters()
    phase1_spatial = 11_200_000
    print("\n=== Phase 1 vs Phase 2 Spatial Comparison ===")
    print(f"  Phase 1 ResNet-18 (single)    : ~{phase1_spatial:,} params")
    print(f"  Phase 2 MobileNetV2Branch     :  {mv2.count_parameters():,} params")
    print(f"  Phase 2 EfficientNetB0Branch  :  {efn.count_parameters():,} params")
    print(f"  Phase 2 Ensemble combined     :  {total_p2:,} params")
    print(f"  Reduction vs Phase 1          :  {1 - total_p2/phase1_spatial:.1%} fewer")
    print(f"  New_Idea.txt claim valid      :  "
          f"{'YES (combined < ResNet-18)' if total_p2 < phase1_spatial else 'FAIL'}")