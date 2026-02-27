"""
classifier.py
=============
Implements the Classification Module of Lightweight-CTGAN-IDS.

Phase 2 note:
  This file is STRUCTURALLY IDENTICAL to ExistingImplementation's
  classifier.py. Both spatial (LightweightSpatialExtractor) and temporal
  (EfficientTemporalExtractor) branches still output (B, 128), so the
  concatenated input is still (B, 256). The classification head is
  therefore completely unchanged between phases — any accuracy difference
  is attributable entirely to the spatial/temporal architecture and
  augmentation (CTGAN), not the classifier.

Paper (Wang et al., 2023) Table 9 — Network structure (unchanged):
  Layer   | Output  | Processing Function
  --------|---------|---------------------
  Concat  | (256,)  | Concat [spatial(128) + temporal(128)]
  FC-4    | (128,)  | ReLU + Dropout
  FC-5    | (64,)   | ReLU + Dropout
  FC-6    | (N_cls,)| softmax
    NSL-KDD:     5 classes
    CIC-IDS2017: 7 classes  ← our focus (DoS subset: 5 classes)
    MQTTset:     6 classes

Table 10 — Hyperparameters (unchanged for fair comparison):
  Batch_Size      = 256
  Learning_Rate   = 0.0001
  Dropout         = 0.5
  Input_dimension = 256   (= 128 spatial + 128 temporal)

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Fully connected classification head for Lightweight-CTGAN-IDS.

    Takes the concatenated spatiotemporal feature vector (256,) and
    produces class logits via FC-4 → FC-5 → FC-6 (Table 9).

    Identical to Phase 1 ClassificationHead — both phases produce
    (B, 128) spatial and (B, 128) temporal features, so this head
    is 100% compatible and unchanged for fair Phase 1 vs Phase 2
    comparison.

    Parameters
    ----------
    input_dim : int
        Input dimension after concat (128 + 128 = 256 per Table 9).
    num_classes : int
        Number of output classes (5 for DoS-only subset).
    dropout : float
        Dropout rate (0.5 per Table 10).
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # FC-4: 256 → 128  (ReLU + Dropout)
        self.fc4 = nn.Linear(input_dim, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout(p=dropout)

        # FC-5: 128 → 64  (ReLU + Dropout)
        self.fc5 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop5 = nn.Dropout(p=dropout)

        # FC-6: 64 → num_classes  (softmax applied during inference/loss)
        # Note: nn.CrossEntropyLoss applies log-softmax internally,
        # so we return raw logits during training. Softmax is applied
        # explicitly only for inference probability outputs.
        self.fc6 = nn.Linear(64, num_classes)

        self._initialize_weights()

    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_feat  : torch.Tensor, shape (B, 128)
            Output from LightweightSpatialExtractor.
        temporal_feat : torch.Tensor, shape (B, 128)
            Output from EfficientTemporalExtractor (BiGRU last hidden).

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Raw class logits (apply softmax for probabilities).
        """
        # Concat spatiotemporal features: (B, 128) + (B, 128) → (B, 256)
        x = torch.cat([spatial_feat, temporal_feat], dim=1)   # (B, 256)

        # FC-4
        x = self.fc4(x)     # (B, 128)
        x = self.relu4(x)
        x = self.drop4(x)

        # FC-5
        x = self.fc5(x)     # (B, 64)
        x = self.relu5(x)
        x = self.drop5(x)

        # FC-6 — raw logits
        logits = self.fc6(x)   # (B, num_classes)

        return logits

    def predict_proba(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns softmax probabilities (for inference only).

        Returns
        -------
        torch.Tensor, shape (B, num_classes) — probability distribution.
        """
        logits = self.forward(spatial_feat, temporal_feat)
        return torch.softmax(logits, dim=1)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    head = ClassificationHead(input_dim=256, num_classes=5)

    spatial = torch.randn(8, 128)
    temporal = torch.randn(8, 128)

    logits = head(spatial, temporal)
    probs = head.predict_proba(spatial, temporal)

    print(f"Spatial feat:  {spatial.shape}")
    print(f"Temporal feat: {temporal.shape}")
    print(f"Logits:        {logits.shape}")   # expect (8, 5)
    print(f"Probs:         {probs.shape}")    # expect (8, 5)
    print(f"Prob sum:      {probs.sum(dim=1).mean():.4f}")  # expect ~1.0
    print(f"Parameters:    {head.count_parameters():,}")