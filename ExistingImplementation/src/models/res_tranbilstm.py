"""
res_tranbilstm.py
=================
Full Res-TranBiLSTM model — assembles all three modules in parallel per Fig. 2.

Paper (Wang et al., 2023) Fig. 2 — Overall workflow:
  Input features
    ├── 1D→2D reshaping → ResNet (Spatial) → spatial_feat (128,)
    └── MLP encoding → TranBiLSTM (Temporal) → temporal_feat (128,)
                              ↓
              Concat [spatial_feat, temporal_feat] → (256,)
                              ↓
                FC-4(128) → FC-5(64) → FC-6(N_cls) → softmax

Key design:
  - Dual-branch parallel feature extraction
  - Spatial: ResNet processes 28×28 grayscale image
  - Temporal: TranBiLSTM processes 64-dim feature sequence
  - Both branches output 128-dim vectors → concat → 256-dim
  - Classification head: 256 → 128 → 64 → N_classes

Configuration for CIC-IDS2017 DoS subset (our study):
  - num_classes = 5  (BENIGN + 4 DoS types)
  - seq_len     = 64 (features after selection)
  - image_size  = 28 (bicubic upsampled from 8×8)

Author: FYP Implementation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from models.spatial.spatial_extractor import SpatialFeatureExtractor
from models.temporal.temporal_extractor import TemporalFeatureExtractor
from models.classification.classifier import ClassificationHead


class ResTranBiLSTM(nn.Module):
    """
    Full Res-TranBiLSTM intrusion detection model.

    Takes dual-branch inputs and produces class logits.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
          DoS subset (our study): 5
          Full CIC-IDS2017:        7
          NSL-KDD:                 5
          MQTTset:                 6
    seq_len : int
        Number of features / sequence length (64 for CIC-IDS2017).
    in_channels : int
        Number of image channels (1 = grayscale).
    spatial_output_dim : int
        ResNet FC-1 output dimension (128 per paper).
    temporal_output_dim : int
        TranBiLSTM output dimension (128 = 64+64 BiLSTM).
    mlp_hidden_dim : int
        MLP FC-2 hidden size (16 per Table 7).
    d_model : int
        Transformer/BiLSTM input dimension (32 per Table 8).
    n_heads : int
        Number of attention heads (4 per Table 8).
    ff_hidden : int
        Transformer feed-forward hidden size (64 per Table 8).
    bilstm_hidden : int
        BiLSTM hidden size per direction (64 per Table 8).
    dropout : float
        Dropout rate (0.5 per Table 10).
    """

    def __init__(
        self,
        num_classes: int = 5,
        seq_len: int = 64,
        in_channels: int = 1,
        spatial_output_dim: int = 128,
        d_model: int = 32,
        n_heads: int = 4,
        ff_hidden: int = 64,
        bilstm_hidden: int = 64,
        mlp_hidden_dim: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.seq_len = seq_len

        # -- Spatial Branch: ResNet --
        self.spatial = SpatialFeatureExtractor(
            in_channels=in_channels,
            output_dim=spatial_output_dim,
            dropout=dropout,
        )

        # -- Temporal Branch: TranBiLSTM --
        self.temporal = TemporalFeatureExtractor(
            seq_len=seq_len,
            mlp_input_dim=1,
            mlp_hidden_dim=mlp_hidden_dim,
            d_model=d_model,
            n_heads=n_heads,
            ff_hidden=ff_hidden,
            bilstm_hidden=bilstm_hidden,
            dropout=dropout,
        )

        # Verify concat dimension
        concat_dim = self.spatial.get_output_dim() + self.temporal.get_output_dim()
        assert concat_dim == 256, (
            f"Expected concat_dim=256, got {concat_dim}. "
            f"spatial={self.spatial.get_output_dim()}, "
            f"temporal={self.temporal.get_output_dim()}"
        )

        # -- Classification Head: FC-4 → FC-5 → FC-6 --
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
        Parallel dual-branch forward pass.

        Parameters
        ----------
        x_img : torch.Tensor, shape (B, 1, 28, 28)
            Spatial input: bicubic-upsampled grayscale image.
        x_seq : torch.Tensor, shape (B, 64) or (B, 64, 1)
            Temporal input: normalized feature sequence.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Raw class logits (use softmax for probabilities).
        """
        # Parallel extraction
        spatial_feat  = self.spatial(x_img)    # (B, 128)
        temporal_feat = self.temporal(x_seq)   # (B, 128)

        # Classification
        logits = self.classifier(spatial_feat, temporal_feat)  # (B, N)

        return logits

    def predict(
        self,
        x_img: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper: returns (predicted_class, probabilities).

        Returns
        -------
        preds  : torch.Tensor, shape (B,) — predicted class indices
        probs  : torch.Tensor, shape (B, num_classes) — softmax probabilities
        """
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
        """
        Returns intermediate feature vectors for analysis/visualization.

        Returns
        -------
        dict with keys: 'spatial', 'temporal', 'concat'
        """
        spatial_feat  = self.spatial(x_img)
        temporal_feat = self.temporal(x_seq)
        concat_feat   = torch.cat([spatial_feat, temporal_feat], dim=1)

        return {
            "spatial":  spatial_feat,
            "temporal": temporal_feat,
            "concat":   concat_feat,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Returns parameter counts for each module and total."""
        spatial_params  = sum(p.numel() for p in self.spatial.parameters() if p.requires_grad)
        temporal_params = sum(p.numel() for p in self.temporal.parameters() if p.requires_grad)
        cls_params      = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total           = spatial_params + temporal_params + cls_params

        return {
            "spatial":       spatial_params,
            "temporal":      temporal_params,
            "classification": cls_params,
            "total":         total,
        }

    def summary(self) -> str:
        """Print model architecture summary."""
        param_counts = self.count_parameters()
        lines = [
            "=" * 60,
            "  Res-TranBiLSTM Model Summary",
            "=" * 60,
            f"  num_classes   : {self.num_classes}",
            f"  seq_len       : {self.seq_len}",
            "",
            "  [Spatial Branch — ResNet]",
            f"    Parameters  : {param_counts['spatial']:>10,}",
            "    Input        : (B, 1, 28, 28)",
            "    Output       : (B, 128)",
            "",
            "  [Temporal Branch — TranBiLSTM]",
            f"    Parameters  : {param_counts['temporal']:>10,}",
            "    Input        : (B, 64) or (B, 64, 1)",
            "    Output       : (B, 128)",
            "",
            "  [Classification Head]",
            f"    Parameters  : {param_counts['classification']:>10,}",
            "    Input        : (B, 256) — concat(spatial, temporal)",
            f"    Output       : (B, {self.num_classes})",
            "",
            f"  TOTAL Parameters: {param_counts['total']:>10,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_dos_model(num_classes: int = 5, dropout: float = 0.5) -> ResTranBiLSTM:
    """
    Build model for CIC-IDS2017 DoS-only subset (our study).

    5 classes: BENIGN, DoS_slowloris, DoS_Slowhttptest, DoS_Hulk, DoS_GoldenEye
    """
    return ResTranBiLSTM(
        num_classes=num_classes,
        seq_len=64,
        dropout=dropout,
    )


def build_full_cicids_model(dropout: float = 0.5) -> ResTranBiLSTM:
    """Build model for full 7-class CIC-IDS2017 (paper Table 9)."""
    return ResTranBiLSTM(num_classes=7, seq_len=64, dropout=dropout)


def build_nslkdd_model(dropout: float = 0.5) -> ResTranBiLSTM:
    """Build model for NSL-KDD 5-class problem (paper Table 9)."""
    return ResTranBiLSTM(num_classes=5, seq_len=121, dropout=dropout)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build DoS model
    model = build_dos_model(num_classes=5)
    model.eval()

    # Dummy inputs
    B = 4
    x_img = torch.randn(B, 1, 28, 28)
    x_seq = torch.randn(B, 64)

    # Forward pass
    logits = model(x_img, x_seq)
    preds, probs = model.predict(x_img, x_seq)

    print(model.summary())
    print(f"\nForward pass:")
    print(f"  x_img:   {x_img.shape}")
    print(f"  x_seq:   {x_seq.shape}")
    print(f"  logits:  {logits.shape}")   # (4, 5)
    print(f"  preds:   {preds.shape}")    # (4,)
    print(f"  probs:   {probs.shape}")    # (4, 5)
    print(f"  prob sum: {probs.sum(dim=1)}")  # all ~1.0

    # Feature vectors
    feats = model.get_feature_vectors(x_img, x_seq)
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")