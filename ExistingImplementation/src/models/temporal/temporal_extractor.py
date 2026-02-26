"""
temporal_extractor.py
=====================
Implements the Temporal Feature Extraction Module (TranBiLSTM) from
Res-TranBiLSTM (Wang et al., 2023).

Paper Table 7 — Network structure:
  Block              | Layer       | Output          | Processing
  -------------------|-------------|-----------------|------------------
  MLP encoding       | FC-2        | (len_seq, 16)   | ReLU + Dropout
                     | FC-3        | (len_seq, 32)   | ReLU + Dropout
  Transformer encoder| Encoder     | (len_seq, 32)   | ReLU + Dropout
  BiLSTM block       | BiLSTMCell  | (128,)          | ReLU + Dropout

Paper Table 8 — Hyperparameters:
  Input_dimension        = 32
  Attention_head         = 4
  FeedForward_hidden_size = 64
  BiLSTM_hidden_size     = 64

Paper Section 3.4:
  "TranBiLSTM consists of a Transformer encoder and a BiLSTM block.
   The introduction of multi-head self-attention increases the attention
   among different features, as well as between local and global features,
   and explores the internal relevance among features."

  "We use the last output of the forward LSTM and the last output of the
   backward LSTM for the concatenation and as the input to the next layer,
   and sets an input dimension that is twice the output dimension to
   minimize the complexity of the model."

Input:  (B, 64)       — flat normalized feature vector (CIC-IDS2017: 64 features)
        OR (B, 64, 1) — pre-formatted sequence (one token per feature)
Output: (B, 128)      — temporal feature vector (concat of forward+backward LSTM)

The 128-dim output matches FC-1(128) spatial output → concat → (256,) for classifier.

Author: FYP Implementation
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# MLP Encoding Block (Table 7: FC-2, FC-3)
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """
    Per-feature MLP encoding: maps each feature token from dim=1 to dim=32.

    Paper: "similar to the word embedding layer in NLP, an MLP layer is
    used to encode data for each feature. Thereafter, the features are
    amplified to map to different subspaces, thereby extracting richer
    features and meeting the input dimensions required by the model."

    Table 7:
      FC-2: (len_seq, 1) → (len_seq, 16)   ReLU + Dropout
      FC-3: (len_seq, 16) → (len_seq, 32)  ReLU + Dropout

    Parameters
    ----------
    input_dim : int
        Input token dimension (1 per feature by default).
    hidden_dim : int
        Intermediate MLP dimension (16 per Table 7 FC-2).
    output_dim : int
        Final token embedding dimension (32 per Table 8).
    dropout : float
        Dropout rate (0.5 per Table 10).
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # FC-2: input_dim → hidden_dim (applied per-token via Linear)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=dropout)

        # FC-3: hidden_dim → output_dim
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, len_seq, input_dim) — sequence of feature tokens.

        Returns
        -------
        torch.Tensor
            Shape (B, len_seq, output_dim=32) — MLP-encoded sequence.
        """
        # FC-2
        out = self.fc2(x)        # (B, len_seq, 16)
        out = self.relu2(out)
        out = self.drop2(out)

        # FC-3
        out = self.fc3(out)      # (B, len_seq, 32)
        out = self.relu3(out)
        out = self.drop3(out)

        return out


# ---------------------------------------------------------------------------
# Transformer Encoder Block (Paper Section 3.4.1)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block per paper Fig. 4 and Eqs. (4-6).

    Structure:
      Input → MultiHeadAttention → Add&Norm → FeedForward → Add&Norm → Output

    Paper Eqs:
      MultiHeadAtt(Q,K,V) = Θ(h1,h2,...,hi) * Wo          [Eq.4]
      Attention(Q,K,V) = softmax(QK^T / sqrt(dk)) * V      [Eq.5]
      FNN(x) = max(0, xW1+b1) * W2 + b2                    [Eq.6]

    Table 8:
      Input_dimension        = 32  (d_model)
      Attention_head         = 4
      FeedForward_hidden_size = 64

    Parameters
    ----------
    d_model : int
        Model dimension (32 per Table 8).
    n_heads : int
        Number of attention heads (4 per Table 8).
    ff_hidden : int
        Feed-forward hidden size (64 per Table 8).
    dropout : float
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        ff_hidden: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Multi-head self-attention (Eq. 4-5)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # (B, seq, dim) format
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=dropout)

        # Feed-forward network (Eq. 6): two-layer FC
        # First layer: ReLU activation; second: no activation
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(ff_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, d_model)

        Returns
        -------
        torch.Tensor, shape (B, seq_len, d_model)
        """
        # Multi-head self-attention with residual + LayerNorm
        attn_out, _ = self.attn(x, x, x)   # Q=K=V=x (self-attention)
        x = self.norm1(x + self.drop1(attn_out))

        # Feed-forward with residual + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop2(ff_out))

        return x


# ---------------------------------------------------------------------------
# BiLSTM Block (Paper Section 3.4.2, Eqs. 7-8)
# ---------------------------------------------------------------------------

class BiLSTMBlock(nn.Module):
    """
    BiLSTM block per paper Fig. 4 and Eqs. (7-8).

    Paper: "each training sequence consists of a forward-propagating LSTM
    network and a backward-propagating LSTM network. These two LSTM
    networks are simultaneously connected to an output layer, providing
    complete contextual information."

    Eq. (8): h_t = [h→_t, h←_t]  (concatenation of forward+backward)

    "We use the last output of the forward LSTM and the last output of
    the backward LSTM for the concatenation."

    Table 7: BiLSTMCell → output (128,)
    Table 8: BiLSTM_hidden_size = 64
      → 64 (forward) + 64 (backward) = 128 total ✓

    Note: "sets an input dimension that is twice the output dimension"
    means: input_dim = 2 * hidden_size, i.e., BiLSTM input = 32 (d_model)
    and output = 64+64 = 128.

    Parameters
    ----------
    input_dim : int
        Input size to LSTM (= d_model = 32).
    hidden_size : int
        Hidden size per direction (64 per Table 8).
    dropout : float
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = hidden_size * 2   # 128 (forward + backward)

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, input_dim=32)

        Returns
        -------
        torch.Tensor, shape (B, 128)
            Concatenation of last forward and last backward LSTM outputs.
        """
        # output: (B, seq_len, 2*hidden_size)
        # h_n:    (2, B, hidden_size) — 2 = num_directions
        output, (h_n, _) = self.bilstm(x)

        # Extract last forward hidden state: h_n[0] → (B, hidden_size)
        h_forward = h_n[0]    # forward direction last hidden
        h_backward = h_n[1]   # backward direction last hidden

        # Concatenate: [h→, h←] → (B, 128) per Eq.(8)
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (B, 128)

        h_concat = self.relu(h_concat)
        h_concat = self.dropout(h_concat)

        return h_concat

    def get_output_dim(self) -> int:
        return self.output_dim


# ---------------------------------------------------------------------------
# Full TranBiLSTM Module
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(nn.Module):
    """
    Full TranBiLSTM temporal feature extraction module.

    Pipeline (Table 7):
      1. Reshape flat (B,64) → sequence (B,64,1) — one token per feature
      2. MLP Encoding: (B,64,1) → FC-2 → (B,64,16) → FC-3 → (B,64,32)
      3. Transformer Encoder: (B,64,32) → (B,64,32)  [with self-attention]
      4. BiLSTM Block: (B,64,32) → (B,128)  [last fwd+bkwd hidden states]

    Output: (B, 128) temporal feature vector, ready for concat with spatial.

    Parameters
    ----------
    seq_len : int
        Length of input sequence (number of features = 64 for CIC-IDS2017).
    mlp_input_dim : int
        Token input dim (1: each feature is a single scalar).
    mlp_hidden_dim : int
        MLP hidden dim (16 per Table 7 FC-2).
    d_model : int
        Transformer/BiLSTM input dim (32 per Table 8).
    n_heads : int
        Attention heads (4 per Table 8).
    ff_hidden : int
        Feed-forward hidden (64 per Table 8).
    bilstm_hidden : int
        BiLSTM hidden per direction (64 per Table 8).
    dropout : float
        Dropout rate (0.5 per Table 10).
    """

    def __init__(
        self,
        seq_len: int = 64,
        mlp_input_dim: int = 1,
        mlp_hidden_dim: int = 16,
        d_model: int = 32,
        n_heads: int = 4,
        ff_hidden: int = 64,
        bilstm_hidden: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = bilstm_hidden * 2  # 128

        # Step 1: MLP encoding (FC-2, FC-3)
        self.mlp_encoder = MLPEncoder(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        # Step 2: Transformer encoder block
        self.transformer = TransformerEncoderBlock(
            d_model=d_model,
            n_heads=n_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
        )

        # Step 3: BiLSTM block
        self.bilstm = BiLSTMBlock(
            input_dim=d_model,
            hidden_size=bilstm_hidden,
            dropout=dropout,
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, seq_len) or (B, seq_len, 1) — input feature sequence.

        Returns
        -------
        torch.Tensor, shape (B, 128)
            Temporal feature vector.
        """
        # Ensure (B, seq_len, 1) shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)   # (B, 64) → (B, 64, 1)

        # MLP encoding: (B, 64, 1) → (B, 64, 32)
        x = self.mlp_encoder(x)

        # Transformer encoder: (B, 64, 32) → (B, 64, 32)
        x = self.transformer(x)

        # BiLSTM: (B, 64, 32) → (B, 128)
        x = self.bilstm(x)

        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def get_output_dim(self) -> int:
        return self.output_dim

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = TemporalFeatureExtractor(seq_len=64)
    model.eval()

    # Test with flat input
    x_flat = torch.randn(8, 64)
    out = model(x_flat)
    print(f"Input (flat):     {x_flat.shape}")
    print(f"Output:           {out.shape}")   # expect (8, 128)

    # Test with sequence input
    x_seq = torch.randn(8, 64, 1)
    out2 = model(x_seq)
    print(f"Input (sequence): {x_seq.shape}")
    print(f"Output:           {out2.shape}")  # expect (8, 128)

    print(f"Parameters: {model.count_parameters():,}")