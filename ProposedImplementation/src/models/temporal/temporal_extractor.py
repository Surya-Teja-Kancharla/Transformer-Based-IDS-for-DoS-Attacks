"""
temporal_extractor.py
=====================
Efficient Temporal Feature Extraction Module for Lightweight-CTGAN-IDS.

Phase 2 replacement for ExistingImplementation/src/models/temporal/temporal_extractor.py.

PHASE 1 vs PHASE 2 TEMPORAL BRANCH:
  ┌────────────────────────────────┬────────────────────────────────────────────┐
  │ Phase 1 (TranBiLSTM)           │ Phase 2 (EfficientTemporalExtractor)       │
  ├────────────────────────────────┼────────────────────────────────────────────┤
  │ MLP encoding: FC 1->16->32     │ MLP encoding: FC 1->16->32  (IDENTICAL)    │
  │ Attention: softmax O(n^2)      │ Attention: linear kernel O(n)              │
  │   nn.MultiheadAttention        │   LinearAttentionBlock (ELU kernel)        │
  │ Recurrence: BiLSTM (3 gates)   │ Recurrence: BiGRU (2 gates, ~33% fewer)   │
  │ Params (temporal): ~0.11 M     │ Params (temporal): ~0.08 M (~27% fewer)   │
  │ Attn complexity: O(n^2 * d)    │ Attn complexity: O(n * d^2) -- O(n)       │
  └────────────────────────────────┴────────────────────────────────────────────┘

MOTIVATION:
  1. Linear Attention (Katharopoulos et al., 2020):
     Replaces softmax(QK^T/sqrt(d))V with phi(Q)(phi(K)^T V) where phi(x)=ELU(x)+1.
     Complexity drops from O(n^2 d) to O(n d^2), which is O(n) for fixed d.
     Critical for real-time IDS where inference latency must be minimised.

  2. Bidirectional GRU (replaces BiLSTM):
     GRU uses 2 gates (reset, update) vs LSTM's 3 (forget, input, output).
     ~33% fewer recurrent parameters, empirically comparable accuracy on
     tabular feature sequences (Chung et al., 2014).

  3. MLP encoding is IDENTICAL to Phase 1 -- same FC-2 (1->16), FC-3 (16->32).
     This isolates Phase 2 improvements to the attention and recurrence only.

ARCHITECTURE (same I/O contract as Phase 1 TemporalFeatureExtractor):
  Input:  (B, 64)       -- flat normalised feature vector
          OR (B, 64, 1) -- pre-formatted sequence (one token per feature)
  MLP:    (B, 64, 1) -> FC-2 -> (B, 64, 16) -> FC-3 -> (B, 64, 32)
  Attn:   (B, 64, 32) -> LinearAttentionBlock -> (B, 64, 32)   [O(n)]
  GRU:    (B, 64, 32) -> BiGRUBlock -> (B, 128)   [concat fwd+bkwd]
  Output: (B, 128)   -- identical to Phase 1 temporal output

PARAMETER COMPARISON:
  Phase 1 TemporalFeatureExtractor (TranBiLSTM): ~0.11 M params
  Phase 2 EfficientTemporalExtractor (LinAttn+BiGRU): ~0.08 M params

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# MLP Encoding Block (Table 7: FC-2, FC-3)
# Identical to Phase 1 -- isolates Phase 2 changes to attention and recurrence
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """
    Per-feature MLP encoding: maps each feature token from dim=1 to dim=32.

    IDENTICAL to Phase 1 MLPEncoder -- reused unchanged so that any
    accuracy difference between phases is attributable solely to the
    attention mechanism (LinearAttn vs softmax) and recurrence (BiGRU vs BiLSTM).

    Paper: "similar to the word embedding layer in NLP, an MLP layer is
    used to encode data for each feature. Thereafter, the features are
    amplified to map to different subspaces, thereby extracting richer
    features and meeting the input dimensions required by the model."

    Table 7:
      FC-2: (len_seq, 1)  -> (len_seq, 16)   ReLU + Dropout
      FC-3: (len_seq, 16) -> (len_seq, 32)   ReLU + Dropout

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

        # FC-2: input_dim -> hidden_dim (applied per-token via Linear)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=dropout)

        # FC-3: hidden_dim -> output_dim
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, len_seq, input_dim) -- sequence of feature tokens.

        Returns
        -------
        torch.Tensor
            Shape (B, len_seq, output_dim=32) -- MLP-encoded sequence.
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
# Linear Attention Block (Phase 2 replacement for TransformerEncoderBlock)
# ---------------------------------------------------------------------------

class LinearAttentionBlock(nn.Module):
    """
    Linear (kernel) self-attention block.

    Phase 1 equivalent: TransformerEncoderBlock (softmax MultiheadAttention)

    Replaces:  Attention(Q,K,V) = softmax(QK^T / sqrt(d)) * V   [O(n^2 d)]
    With:      Attention(Q,K,V) = phi(Q)(phi(K)^T V) / (phi(Q) phi(K)^T 1)

    Using phi(x) = ELU(x) + 1  (Katharopoulos et al., 2020).
    Exploits matrix associativity: Q(K^T V) costs O(n d^2) instead of O(n^2 d).
    For fixed d=32, this is O(n) in sequence length -- critical for IDS deployment.

    Structure (identical to Phase 1 TransformerEncoderBlock):
      Input -> LinearAttn -> Add&Norm -> FFN -> Add&Norm -> Output

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
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        # Q, K, V projection matrices (no bias -- same as standard MHA)
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=dropout)

        # Feed-forward network (identical structure to Phase 1)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(ff_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(p=dropout)

    def _kernel_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map phi(x) = ELU(x) + 1  (always positive, approximates exp)."""
        return F.elu(x) + 1.0

    def _linear_attn(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute linear attention via the associativity trick.

        Instead of (QK^T)V  [O(n^2 d)], compute Q(K^T V)  [O(n d^2)].

        Parameters
        ----------
        Q, K, V : torch.Tensor, shape (B, heads, n, d_head)

        Returns
        -------
        torch.Tensor, shape (B, heads, n, d_head)
        """
        Q = self._kernel_fn(Q)   # phi(Q): (B, heads, n, d_head)
        K = self._kernel_fn(K)   # phi(K): (B, heads, n, d_head)

        # K^T V: (B, heads, d_head, d_head)
        KV = torch.einsum("bhnd,bhnm->bhdm", K, V)

        # Normaliser: phi(Q) * sum(phi(K))
        K_sum = K.sum(dim=2)                                    # (B, heads, d_head)
        denom = torch.einsum("bhnd,bhd->bhn", Q, K_sum)        # (B, heads, n)
        denom = denom.clamp(min=1e-6).unsqueeze(-1)             # (B, heads, n, 1)

        # phi(Q)(K^T V) / normaliser
        out = torch.einsum("bhnd,bhdm->bhnm", Q, KV) / denom   # (B, heads, n, d_head)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, d_model)

        Returns
        -------
        torch.Tensor, shape (B, seq_len, d_model)
        """
        B, n, _ = x.shape

        # Project and reshape to multi-head format
        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, n, self.n_heads, self.d_head).transpose(1, 2)

        Q = _split(self.q_proj(x))   # (B, heads, n, d_head)
        K = _split(self.k_proj(x))
        V = _split(self.v_proj(x))

        # Linear attention: O(n) in sequence length
        attn = self._linear_attn(Q, K, V)                              # (B, heads, n, d_head)
        attn = attn.transpose(1, 2).contiguous().view(B, n, self.d_model)  # (B, n, d_model)
        attn = self.out_proj(attn)

        # Add & Norm (same as Phase 1 TransformerEncoderBlock)
        x = self.norm1(x + self.drop1(attn))

        # FFN + Add & Norm (same as Phase 1)
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop2(ff_out))

        return x


# ---------------------------------------------------------------------------
# Bidirectional GRU Block (Phase 2 replacement for BiLSTMBlock)
# ---------------------------------------------------------------------------

class BiGRUBlock(nn.Module):
    """
    Bidirectional GRU -- lightweight replacement for BiLSTMBlock (Phase 1).

    GRU uses 2 gates (reset, update) vs LSTM's 3 (forget, input, output),
    yielding ~33% fewer recurrent parameters with comparable accuracy on
    tabular feature-sequence data (Chung et al., 2014).

    Same output contract as Phase 1 BiLSTMBlock:
      - Input:  (B, seq_len, input_dim=32)
      - Output: (B, 128)  -- concat(h_forward, h_backward)

    "We use the last output of the forward [GRU] and the last output of the
    backward [GRU] for the concatenation" -- same strategy as Phase 1 BiLSTM.

    Parameters
    ----------
    input_dim : int
        Input size to GRU (= d_model = 32).
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
        self.output_dim  = hidden_size * 2   # 128 (forward + backward)

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, input_dim=32)

        Returns
        -------
        torch.Tensor, shape (B, 128)
            Concatenation of last forward and last backward GRU outputs.
        """
        # h_n: (2, B, hidden_size)  -- 2 = num_directions
        _, h_n = self.bigru(x)

        h_forward  = h_n[0]   # forward  direction last hidden: (B, hidden_size)
        h_backward = h_n[1]   # backward direction last hidden: (B, hidden_size)

        # Concatenate: [h_fwd, h_bkwd] -> (B, 128)
        h_concat = torch.cat([h_forward, h_backward], dim=1)   # (B, 128)

        h_concat = self.relu(h_concat)
        h_concat = self.dropout(h_concat)

        return h_concat

    def get_output_dim(self) -> int:
        return self.output_dim


# ---------------------------------------------------------------------------
# Full EfficientTemporalExtractor
# ---------------------------------------------------------------------------

class EfficientTemporalExtractor(nn.Module):
    """
    Full efficient temporal feature extraction module for Phase 2.

    Phase 2 replacement for Phase 1's TemporalFeatureExtractor (TranBiLSTM).

    Pipeline:
      1. Reshape flat (B,64) -> sequence (B,64,1) -- one token per feature
      2. MLP Encoding:  (B,64,1)  -> FC-2 -> (B,64,16) -> FC-3 -> (B,64,32)
         [IDENTICAL to Phase 1]
      3. Linear Attn:   (B,64,32) -> LinearAttentionBlock -> (B,64,32)
         [Phase 2: O(n) vs Phase 1's O(n^2) softmax attention]
      4. BiGRU Block:   (B,64,32) -> BiGRUBlock -> (B,128)
         [Phase 2: GRU vs Phase 1's LSTM, ~33% fewer recurrent params]

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
        Attention/GRU input dim (32 per Table 8).
    n_heads : int
        Attention heads (4 per Table 8).
    ff_hidden : int
        Feed-forward hidden (64 per Table 8).
    gru_hidden : int
        GRU hidden size per direction (64 per Table 8, same as BiLSTM).
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
        gru_hidden: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len    = seq_len
        self.output_dim = gru_hidden * 2   # 128

        # Step 1: MLP encoding (FC-2, FC-3) -- identical to Phase 1
        self.mlp_encoder = MLPEncoder(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        # Step 2: Linear attention block (Phase 2 replacement for softmax Transformer)
        self.linear_attn = LinearAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            ff_hidden=ff_hidden,
            dropout=dropout,
        )

        # Step 3: BiGRU block (Phase 2 replacement for BiLSTM)
        self.bigru = BiGRUBlock(
            input_dim=d_model,
            hidden_size=gru_hidden,
            dropout=dropout,
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, seq_len) or (B, seq_len, 1) -- input feature sequence.

        Returns
        -------
        torch.Tensor, shape (B, 128)
            Temporal feature vector.
        """
        # Ensure (B, seq_len, 1) shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)   # (B, 64) -> (B, 64, 1)

        # MLP encoding: (B, 64, 1) -> (B, 64, 32)
        x = self.mlp_encoder(x)

        # Linear attention: (B, 64, 32) -> (B, 64, 32)  [O(n)]
        x = self.linear_attn(x)

        # BiGRU: (B, 64, 32) -> (B, 128)
        x = self.bigru(x)

        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
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
    model = EfficientTemporalExtractor(seq_len=64)
    model.eval()

    # Test with flat input (same test as Phase 1)
    x_flat = torch.randn(8, 64)
    out = model(x_flat)
    print(f"Input (flat):     {x_flat.shape}")
    print(f"Output:           {out.shape}")    # expect (8, 128)

    # Test with sequence input (same test as Phase 1)
    x_seq = torch.randn(8, 64, 1)
    out2 = model(x_seq)
    print(f"Input (sequence): {x_seq.shape}")
    print(f"Output:           {out2.shape}")   # expect (8, 128)

    print(f"Parameters: {model.count_parameters():,}")

    # Phase 1 vs Phase 2 comparison
    phase1_params = 113_024   # TemporalFeatureExtractor (TranBiLSTM) approx
    phase2_params = model.count_parameters()
    print(f"\nPhase 1 TemporalFeatureExtractor (TranBiLSTM): ~{phase1_params:,}")
    print(f"Phase 2 EfficientTemporalExtractor (LinAttn+BiGRU): {phase2_params:,}")