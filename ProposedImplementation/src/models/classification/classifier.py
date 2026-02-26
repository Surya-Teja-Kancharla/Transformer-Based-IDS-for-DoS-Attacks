"""
classifier.py  —  ProposedImplementation
=========================================
Classification head — identical structure to ExistingImplementation.

Both branches (spatial/temporal) still output (B, 128), so the
concatenated input is still (B, 256) and this head is unchanged.
Keeping a separate copy here ensures ProposedImplementation is a
self-contained codebase.

Table 9 (paper) — unchanged:
  Concat → FC-4(128) → FC-5(64) → FC-6(N_cls) → softmax

Author: FYP ProposedImplementation
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    FC classification head — same as ExistingImplementation.

    Parameters
    ----------
    input_dim   : int   — 256 (128 spatial + 128 temporal)
    num_classes : int   — 5 for DoS subset
    dropout     : float — 0.5
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fc4   = nn.Linear(input_dim, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout(p=dropout)
        self.fc5   = nn.Linear(128, 64)
        self.relu5 = nn.ReLU(inplace=True)
        self.drop5 = nn.Dropout(p=dropout)
        self.fc6   = nn.Linear(64, num_classes)

        self._init_weights()

    def forward(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([spatial_feat, temporal_feat], dim=1)
        x = self.drop4(self.relu4(self.fc4(x)))
        x = self.drop5(self.relu5(self.fc5(x)))
        return self.fc6(x)

    def predict_proba(
        self,
        spatial_feat: torch.Tensor,
        temporal_feat: torch.Tensor,
    ) -> torch.Tensor:
        return torch.softmax(self.forward(spatial_feat, temporal_feat), dim=1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)