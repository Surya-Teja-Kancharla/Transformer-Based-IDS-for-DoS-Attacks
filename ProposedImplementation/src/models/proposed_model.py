"""
proposed_model.py
=================
Full Ensemble-TranBiLSTM model -- assembles all three proposed modules.

Phase 2 equivalent of ExistingImplementation/src/models/res_tranbilstm.py.

Paper (Wang et al., 2023) Fig. 2 -- Overall workflow (same dual-branch structure):
  Input features
    |-- 1D->2D reshaping -> EnsembleSpatialExtractor              -> (B, 128)
    |     MobileNetV2 + EfficientNet-B0 soft feature averaging
    |
    +-- MLP encoding -> EfficientTemporalExtractor (LinearAttn+BiLSTM) -> (B, 128)
                              |
              Concat [spatial_feat, temporal_feat] -> (B, 256)
                              |
                FC-4(128) -> FC-5(64) -> FC-6(N_cls) -> softmax

IMPROVEMENTS OVER PHASE 1 (Res-TranBiLSTM), from New_Idea.txt:
  +----------------------+--------------------+-------------------------------+
  | Component            | Phase 1            | Phase 2                       |
  +----------------------+--------------------+-------------------------------+
  | Spatial branch       | ResNet-18 (11.2M)  | MobileNetV2 + EfficientNet-B0 |
  |                      | Single model       | Soft ensemble (~3.2M combined)|
  | Ensemble method      | N/A                | Soft feature averaging        |
  |                      |                    | (New_Idea.txt lines 276-280)  |
  | Temporal attention   | Softmax O(n^2)     | Linear kernel O(n)            |
  | Temporal recurrence  | BiLSTM             | BiLSTM (IDENTICAL)            |
  | Data augmentation    | SMOTE-ENN          | CTGAN (generative model)      |
  | Spatial params       | ~11.2M             | ~3.2M (~71% reduction)        |
  | Total params (approx)| ~11.4M             | ~3.4M (~70% reduction)        |
  | Accuracy target      | 98.90% (baseline)  | >=98.5% (within 0.5%)        |
  +----------------------+--------------------+-------------------------------+

ISOLATION PRINCIPLE:
  BiLSTM is kept identical to Phase 1 (paper architecture preserved) so
  that any accuracy or efficiency difference is attributable solely to:
    1. Spatial branch: ResNet-18 -> MobileNetV2 + EfficientNet-B0 ensemble
    2. Data augmentation: SMOTE-ENN -> CTGAN
    3. Attention: softmax O(n^2) -> linear kernel O(n)

RESEARCH CONTRIBUTION (New_Idea.txt lines 282-285):
  "We replace the single heavy ResNet-18 spatial branch with an ensemble
   of two lightweight models -- MobileNetV2 and EfficientNet-B0 -- connected
   via soft voting, reducing total parameters while improving prediction
   reliability for DoS sub-type detection in fog-deployed IoT IDS."

I/O CONTRACT (identical to ExistingImplementation ResTranBiLSTM):
  x_img : (B, 1, 28, 28)
  x_seq : (B, 64) or (B, 64, 1)
  output: (B, num_classes) logits

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple

from models.spatial.spatial_extractor   import EnsembleSpatialExtractor
from models.temporal.temporal_extractor import EfficientTemporalExtractor
from models.classification.classifier   import ClassificationHead


class LightweightIDSModel(nn.Module):
    """
    Full Ensemble-TranBiLSTM intrusion detection model.

    Phase 2 replacement for Phase 1's ResTranBiLSTM.

    Spatial branch: EnsembleSpatialExtractor
      Runs MobileNetV2Branch and EfficientNetB0Branch in parallel on the
      same 28x28 input. Both produce (B, 128). Soft ensemble (average)
      gives (B, 128) to the concat layer.

    Temporal branch: EfficientTemporalExtractor
      LinearAttention O(n) + BiLSTM (identical to Phase 1). Produces (B, 128).

    Both branches run in parallel (same as Phase 1 architecture).
    Concat (B, 256) -> ClassificationHead (same as Phase 1).

    Parameters
    ----------
    num_classes : int
        Number of output classes.
          DoS subset (our study): 5
          Full CIC-IDS2017:        7
          NSL-KDD:                 5
    seq_len : int
        Number of features / sequence length (64 for CIC-IDS2017).
    in_channels : int
        Number of image channels (1 = grayscale).
    output_dim : int
        Both branch output dimensions (128 per paper FC-1).
    d_model : int
        Attention / BiLSTM input dimension (32 per Table 8).
    n_heads : int
        Number of attention heads (4 per Table 8).
    ff_hidden : int
        Feed-forward hidden size (64 per Table 8).
    bilstm_hidden : int
        BiLSTM hidden size per direction (64 per Table 8).
    mlp_hidden_dim : int
        MLP FC-2 hidden size (16 per Table 7).
    dropout : float
        Dropout rate (0.5 per Table 10).
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
        bilstm_hidden: int = 64,
        mlp_hidden_dim: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.seq_len     = seq_len

        # -- Spatial Branch: EnsembleSpatialExtractor --
        # MobileNetV2Branch + EfficientNetB0Branch, soft feature averaging
        # Replaces Phase 1's SpatialFeatureExtractor (ResNet-18)
        self.spatial = EnsembleSpatialExtractor(
            in_channels=in_channels,
            output_dim=output_dim,
            dropout=dropout,
        )

        # -- Temporal Branch: EfficientTemporalExtractor (LinearAttn + BiLSTM) --
        # LinearAttention replaces softmax attention (O(n) vs O(n^2)).
        # BiLSTM is IDENTICAL to Phase 1 -- paper architecture preserved.
        self.temporal = EfficientTemporalExtractor(
            seq_len=seq_len,
            mlp_input_dim=1,
            mlp_hidden_dim=mlp_hidden_dim,
            d_model=d_model,
            n_heads=n_heads,
            ff_hidden=ff_hidden,
            bilstm_hidden=bilstm_hidden,
            dropout=dropout,
        )

        # Verify concat dimension (same assertion as Phase 1)
        concat_dim = self.spatial.get_output_dim() + self.temporal.get_output_dim()
        assert concat_dim == 256, (
            f"Expected concat_dim=256, got {concat_dim}. "
            f"spatial={self.spatial.get_output_dim()}, "
            f"temporal={self.temporal.get_output_dim()}"
        )

        # -- Classification Head: FC-4 -> FC-5 -> FC-6  (same as Phase 1) --
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

        The spatial ensemble (MobileNetV2 + EfficientNetB0) and temporal
        branch (LinearAttn + BiLSTM) run in parallel. Both produce (B, 128).
        Concat -> ClassificationHead -> logits.

        Parameters
        ----------
        x_img : torch.Tensor, shape (B, 1, 28, 28)
            Spatial input: bicubic-upsampled grayscale traffic image.
        x_seq : torch.Tensor, shape (B, 64) or (B, 64, 1)
            Temporal input: normalised feature sequence.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Raw class logits (use softmax for probabilities).
        """
        # Parallel extraction (same structure as Phase 1)
        spatial_feat  = self.spatial(x_img)    # (B, 128) -- ensemble average
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
        preds : torch.Tensor, shape (B,)             -- predicted class indices
        probs : torch.Tensor, shape (B, num_classes)  -- softmax probabilities
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
        Returns intermediate feature vectors for analysis/visualisation.

        Returns
        -------
        dict with keys:
          'spatial'  -- (B, 128) ensemble-averaged spatial features
          'temporal' -- (B, 128) temporal features
          'concat'   -- (B, 256) concatenated features
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
        spatial_params  = sum(p.numel() for p in self.spatial.parameters()    if p.requires_grad)
        temporal_params = sum(p.numel() for p in self.temporal.parameters()   if p.requires_grad)
        cls_params      = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total           = spatial_params + temporal_params + cls_params

        # Per-branch spatial breakdown
        branch_params = self.spatial.count_parameters_by_branch()

        return {
            "spatial":              spatial_params,
            "spatial_mobilenet":    branch_params["mobilenet"],
            "spatial_efficientnet": branch_params["efficientnet"],
            "temporal":             temporal_params,
            "classification":       cls_params,
            "total":                total,
        }

    def summary(self) -> str:
        """Returns a formatted architecture summary string."""
        param_counts = self.count_parameters()
        lines = [
            "=" * 65,
            "  LightweightIDSModel -- Ensemble-TranBiLSTM (ProposedImplementation)",
            "=" * 65,
            f"  num_classes   : {self.num_classes}",
            f"  seq_len       : {self.seq_len}",
            "",
            "  [Spatial Branch -- MobileNetV2 + EfficientNet-B0 Ensemble]",
            f"    MobileNetV2Branch    : {param_counts['spatial_mobilenet']:>10,} params",
            f"    EfficientNetB0Branch : {param_counts['spatial_efficientnet']:>10,} params",
            f"    Ensemble total       : {param_counts['spatial']:>10,} params",
            "    Ensemble method      : soft feature averaging (avg of branch outputs)",
            "    Input                : (B, 1, 28, 28) -- each branch sees same input",
            "    Output               : (B, 128)       -- averaged feature vector",
            "",
            "  [Temporal Branch -- LinearAttn + BiLSTM]",
            f"    Parameters           : {param_counts['temporal']:>10,}",
            "    Attention            : LinearAttentionBlock  O(n)",
            "    Recurrence           : BiLSTM (nn.LSTM, identical to Phase 1)",
            "    Input                : (B, 64) or (B, 64, 1)",
            "    Output               : (B, 128)",
            "",
            "  [Classification Head -- identical to Phase 1]",
            f"    Parameters           : {param_counts['classification']:>10,}",
            "    Input                : (B, 256) -- concat(spatial, temporal)",
            f"    Output               : (B, {self.num_classes})",
            "",
            f"  TOTAL Parameters     : {param_counts['total']:>10,}",
            f"  Phase 1 (ResNet-18)  : ~11,334,117",
            f"  Reduction            : ~{1 - param_counts['total']/11_334_117:.0%} fewer parameters",
            "=" * 65,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_proposed_dos_model(
    num_classes: int = 5, dropout: float = 0.5
) -> LightweightIDSModel:
    """
    Build Phase 2 model for CIC-IDS2017 DoS-only subset (our study).

    5 classes: BENIGN, DoS_slowloris, DoS_Slowhttptest, DoS_Hulk, DoS_GoldenEye
    Phase 2 twin of: build_dos_model() in ExistingImplementation.
    """
    return LightweightIDSModel(
        num_classes=num_classes,
        seq_len=64,
        dropout=dropout,
    )


def build_proposed_full_cicids_model(dropout: float = 0.5) -> LightweightIDSModel:
    """Build Phase 2 model for full 7-class CIC-IDS2017 (paper Table 9)."""
    return LightweightIDSModel(num_classes=7, seq_len=64, dropout=dropout)


def build_proposed_nslkdd_model(dropout: float = 0.5) -> LightweightIDSModel:
    """Build Phase 2 model for NSL-KDD 5-class problem (paper Table 9)."""
    return LightweightIDSModel(num_classes=5, seq_len=121, dropout=dropout)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_proposed_dos_model(num_classes=5)
    model.eval()

    B = 4
    x_img = torch.randn(B, 1, 28, 28)
    x_seq = torch.randn(B, 64)

    # Forward pass
    logits = model(x_img, x_seq)
    preds, probs = model.predict(x_img, x_seq)

    print(model.summary())
    print(f"\nForward pass:")
    print(f"  x_img:    {tuple(x_img.shape)}")
    print(f"  x_seq:    {tuple(x_seq.shape)}")
    print(f"  logits:   {tuple(logits.shape)}")     # (4, 5)
    print(f"  preds:    {tuple(preds.shape)}")      # (4,)
    print(f"  probs:    {tuple(probs.shape)}")      # (4, 5)
    print(f"  prob sum: {probs.sum(dim=1)}")        # all ~1.0

    # Feature vectors
    feats = model.get_feature_vectors(x_img, x_seq)
    print(f"\nFeature vectors:")
    for k, v in feats.items():
        print(f"  {k}: {tuple(v.shape)}")

    # Full parameter breakdown
    p = model.count_parameters()
    print(f"\nParameter breakdown:")
    print(f"  MobileNetV2Branch    : {p['spatial_mobilenet']:>10,}")
    print(f"  EfficientNetB0Branch : {p['spatial_efficientnet']:>10,}")
    print(f"  Spatial ensemble     : {p['spatial']:>10,}")
    print(f"  Temporal (LinAttn+BiLSTM) : {p['temporal']:>10,}")
    print(f"  Classification       : {p['classification']:>10,}")
    print(f"  TOTAL Phase 2        : {p['total']:>10,}")
    print(f"  Phase 1 baseline     : ~11,334,117")
    print(f"  Reduction            :  {1 - p['total']/11_334_117:.1%} fewer params")
    