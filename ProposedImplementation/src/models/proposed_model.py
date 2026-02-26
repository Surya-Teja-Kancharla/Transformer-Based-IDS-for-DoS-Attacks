"""
proposed_model.py  —  ProposedImplementation
=============================================
Full Lightweight IDS Model — assembles the three proposed modules.

ARCHITECTURE OVERVIEW:
  Input features
    ├── 1D→2D reshaping → LightweightSpatialExtractor (DSConv+SE) → (B,128)
    └── MLP encoding → EfficientTemporalExtractor (LinearAttn+BiGRU) → (B,128)
                              ↓
              Concat [spatial, temporal] → (B, 256)
                              ↓
              ClassificationHead: FC-4(128)→FC-5(64)→FC-6(N_cls)

IMPROVEMENTS OVER EXISTING MODEL (Res-TranBiLSTM):
  ┌──────────────────────┬─────────────────┬───────────────────┐
  │ Component            │ Existing        │ Proposed          │
  ├──────────────────────┼─────────────────┼───────────────────┤
  │ Spatial              │ ResNet (11.2M)  │ DSConv+SE (~0.9M) │
  │ Temporal attention   │ Softmax O(n²)   │ Linear O(n)       │
  │ Temporal recurrence  │ BiLSTM          │ BiGRU (~33% fewer)│
  │ Total params (approx)│ ~11.4M          │ ~1.1M             │
  │ Param reduction      │ baseline        │ ~90%              │
  └──────────────────────┴─────────────────┴───────────────────┘
  Target accuracy: within 0.5% of Res-TranBiLSTM on DoS subset.

I/O CONTRACT (identical to ExistingImplementation):
  x_img : (B, 1, 28, 28)
  x_seq : (B, 64)
  output: (B, num_classes) logits

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple

from models.spatial.spatial_extractor   import LightweightSpatialExtractor
from models.temporal.temporal_extractor import EfficientTemporalExtractor
from models.classification.classifier   import ClassificationHead


class LightweightIDSModel(nn.Module):
    """
    Proposed lightweight IDS model for resource-constrained IoT/fog deployment.

    Parameters
    ----------
    num_classes   : int   — 5 for DoS subset
    seq_len       : int   — 64
    in_channels   : int   — 1 (grayscale)
    output_dim    : int   — 128 (both branches)
    d_model       : int   — 32
    n_heads       : int   — 4
    ff_hidden     : int   — 64
    gru_hidden    : int   — 64
    mlp_hidden    : int   — 16
    dropout       : float — 0.5
    """

    def __init__(
        self,
        num_classes: int = 5,
        seq_len: int = 64,
        in_channels: int = 1,
        output_dim: int = 128,
        d_model: int = 32,
        n_heads: int = 4,
        ff_hidden: int = 64,
        gru_hidden: int = 64,
        mlp_hidden: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seq_len     = seq_len

        # Lightweight spatial branch
        self.spatial = LightweightSpatialExtractor(
            in_channels=in_channels,
            output_dim=output_dim,
            dropout=dropout,
        )

        # Efficient temporal branch
        self.temporal = EfficientTemporalExtractor(
            seq_len=seq_len,
            mlp_input_dim=1,
            mlp_hidden_dim=mlp_hidden,
            d_model=d_model,
            n_heads=n_heads,
            ff_hidden=ff_hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
        )

        concat_dim = self.spatial.get_output_dim() + self.temporal.get_output_dim()
        assert concat_dim == 256, f"Expected 256, got {concat_dim}"

        # Classification head (same as existing)
        self.classifier = ClassificationHead(
            input_dim=concat_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_img: (B, 1, 28, 28)
        x_seq: (B, 64)
        returns logits: (B, num_classes)
        """
        spatial_feat  = self.spatial(x_img)
        temporal_feat = self.temporal(x_seq)
        return self.classifier(spatial_feat, temporal_feat)

    def predict(
        self,
        x_img: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_img, x_seq)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)
        return preds, probs

    def get_feature_vectors(
        self,
        x_img: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        s = self.spatial(x_img)
        t = self.temporal(x_seq)
        return {"spatial": s, "temporal": t, "concat": torch.cat([s, t], dim=1)}

    def count_parameters(self) -> Dict[str, int]:
        sp = sum(p.numel() for p in self.spatial.parameters()    if p.requires_grad)
        tp = sum(p.numel() for p in self.temporal.parameters()   if p.requires_grad)
        cp = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return {"spatial": sp, "temporal": tp, "classification": cp, "total": sp+tp+cp}

    def summary(self) -> str:
        pc = self.count_parameters()
        lines = [
            "=" * 60,
            "  LightweightIDSModel (ProposedImplementation)",
            "=" * 60,
            f"  num_classes : {self.num_classes}",
            f"  seq_len     : {self.seq_len}",
            "",
            "  [Spatial — DSConv + SE + InvertedResidual]",
            f"    Parameters : {pc['spatial']:>10,}",
            "    Input       : (B, 1, 28, 28)",
            "    Output      : (B, 128)",
            "",
            "  [Temporal — LinearAttn + BiGRU]",
            f"    Parameters : {pc['temporal']:>10,}",
            "    Input       : (B, 64)",
            "    Output      : (B, 128)",
            "",
            "  [Classification Head]",
            f"    Parameters : {pc['classification']:>10,}",
            "    Input       : (B, 256)",
            f"    Output      : (B, {self.num_classes})",
            "",
            f"  TOTAL Parameters: {pc['total']:>10,}",
            "=" * 60,
        ]
        return "\n".join(lines)


def build_proposed_dos_model(num_classes: int = 5, dropout: float = 0.5) -> LightweightIDSModel:
    """Build proposed model for CIC-IDS2017 DoS-only subset."""
    return LightweightIDSModel(num_classes=num_classes, seq_len=64, dropout=dropout)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_proposed_dos_model()
    model.eval()

    x_img = torch.randn(4, 1, 28, 28)
    x_seq = torch.randn(4, 64)
    logits = model(x_img, x_seq)

    print(model.summary())
    print(f"\nx_img:  {x_img.shape}")
    print(f"x_seq:  {x_seq.shape}")
    print(f"logits: {logits.shape}")   # (4, 5)