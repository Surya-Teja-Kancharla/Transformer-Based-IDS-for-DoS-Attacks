"""
flops_counter.py  --  ProposedImplementation/src/evaluation/
=============================================================
Counts FLOPs (Multiply-Accumulate operations) and parameters for the
LightweightIDSModel (Phase 2), broken down by module and layer type.

Phase 2 twin of ExistingImplementation/src/evaluation/flops_counter.py.
Identical implementation -- the MAC calculators are model-agnostic and
cover every layer type present in LightweightIDSModel:

  Phase 1 layer types present       Phase 2 layer types present
  ──────────────────────────────── ────────────────────────────────────────────
  nn.Conv2d   (ResNet standard 3x3) nn.Conv2d   (MBConvBlock DW + PW + expand)
  nn.BatchNorm2d                    nn.BatchNorm2d
  nn.Linear   (MLP, FFN, FC head)   nn.Linear   (same + SE fc1/fc2 + attn proj)
  nn.MultiheadAttention             [replaced by LinearAttentionBlock,
                                     uses nn.Linear internally]
  nn.LSTM     (BiLSTM)              nn.LSTM     (BiLSTM -- identical to Phase 1)
  nn.AdaptiveMaxPool2d              nn.AdaptiveAvgPool2d (per-branch global pool)

Phase 2 spatial branch (EnsembleSpatialExtractor):
  Two parallel branches -- MobileNetV2Branch and EfficientNetB0Branch.
  Both use nn.Conv2d with groups=mid_ch (depthwise) and groups=1 (pointwise).
  EfficientNetB0Branch adds SqueezeExcitation (nn.Linear fc1/fc2 + AdaptiveAvgPool2d).
  All handled by existing _macs_conv2d and _macs_linear calculators.

  The by_module breakdown will show:
    spatial.mobilenet.*    -- MobileNetV2Branch MACs
    spatial.efficientnet.* -- EfficientNetB0Branch MACs
    temporal.*             -- LinearAttn + BiLSTM MACs
    classifier.*           -- FC-4, FC-5, FC-6 MACs

NOTE on LinearAttentionBlock (Phase 2):
  LinearAttentionBlock contains no nn.MultiheadAttention -- it is built
  entirely from nn.Linear (q_proj, k_proj, v_proj, out_proj, FFN layers).
  These are counted by _macs_linear automatically.
  The kernel attention itself (einsum operations) adds O(n*d^2) MACs;
  this is negligible vs the O(n^2*d) softmax alternative and is not
  separately counted (consistent with how most profilers treat manual
  matrix multiplications vs nn.MultiheadAttention).

NOTE on BiLSTM (Phase 2):
  Phase 2 uses nn.LSTM (bidirectional=True) -- identical to Phase 1.
  _macs_lstm handles both phases.

No external libraries required (no torchinfo, no thop, no fvcore).
Uses PyTorch forward hooks only.

Usage -- standalone:
  python src/evaluation/flops_counter.py

Usage -- in code:
  from evaluation.flops_counter import count_flops, print_flops_report

  model  = build_proposed_dos_model()
  x_img  = torch.randn(1, 1, 28, 28)
  x_seq  = torch.randn(1, 64)
  report = count_flops(model, x_img, x_seq)
  print_flops_report(report)

Definitions used throughout:
  MAC  = one multiply + one add  (standard for NN FLOPs counting)
  FLOPs ~= 2 x MACs  (some papers count each multiply and add separately)
  This file reports MACs. Use report["total_flops"] for 2xMACs if needed.

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Per-layer MAC calculators
# ---------------------------------------------------------------------------

def _macs_conv2d(module: nn.Conv2d, input: Tuple, output: torch.Tensor) -> int:
    """
    MACs for Conv2d.

    Formula:  MACs = C_out x H_out x W_out x (C_in/groups) x k_H x k_W x B

    groups > 1 handles Phase 2 depthwise convolutions (groups == in_channels).
    Bias adds C_out x H_out x W_out MACs if present (negligible, included).
    """
    B, C_out, H_out, W_out = output.shape
    k_h, k_w = (module.kernel_size if isinstance(module.kernel_size, tuple)
                 else (module.kernel_size, module.kernel_size))
    c_in_per_group = module.in_channels // module.groups
    macs = B * C_out * H_out * W_out * c_in_per_group * k_h * k_w
    if module.bias is not None:
        macs += B * C_out * H_out * W_out   # bias adds
    return int(macs)


def _macs_batchnorm2d(module: nn.BatchNorm2d, input: Tuple, output: torch.Tensor) -> int:
    """
    MACs for BatchNorm2d.
    Per element: subtract mean (1 add) + divide std (1 mul) + affine (2 ops) ~= 2 MACs.
    """
    B, C, H, W = output.shape
    return int(2 * B * C * H * W)


def _macs_linear(module: nn.Linear, input: Tuple, output: torch.Tensor) -> int:
    """
    MACs for Linear (fully connected).

    Formula: MACs = batch_elements x in_features x out_features
    where batch_elements = product of all dims except last.

    Covers Phase 2 uses: MLP encoder (FC-2, FC-3), LinearAttentionBlock
    projections (q/k/v/out/FFN), and ClassificationHead (FC-4, FC-5, FC-6).
    """
    in_tensor = input[0]
    # All dims except the last (feature) dim are batch dims
    batch_elems = in_tensor.numel() // in_tensor.shape[-1]
    macs = batch_elems * module.in_features * module.out_features
    if module.bias is not None:
        macs += batch_elems * module.out_features
    return int(macs)


def _macs_multihead_attention(
    module: nn.MultiheadAttention,
    input: Tuple,
    output: Tuple,
) -> int:
    """
    MACs for nn.MultiheadAttention (softmax dot-product attention).

    Present in Phase 1 (TransformerEncoderBlock).
    NOT present in Phase 2 (replaced by LinearAttentionBlock using nn.Linear).
    Kept in registry for completeness and Phase 1 compatibility.

    Components:
      1. Q, K, V projections: 3 x B x seq x d_model x d_model
      2. Scaled dot-product per head:
           QK^T:  B x h x seq x seq x d_head
           AttnV: B x h x seq x seq x d_head
      3. Output projection: B x seq x d_model x d_model
    """
    q = input[0]
    if module.batch_first:
        B, seq, d_model = q.shape
    else:
        seq, B, d_model = q.shape

    h = module.num_heads
    d_head = d_model // h

    macs_qkv    = 3 * B * seq * d_model * d_model
    macs_qkt    = B * h * seq * seq * d_head
    macs_attn_v = B * h * seq * seq * d_head
    macs_out    = B * seq * d_model * d_model

    return int(macs_qkv + macs_qkt + macs_attn_v + macs_out)


def _macs_lstm(module: nn.LSTM, input: Tuple, output: Tuple) -> int:
    """
    MACs for nn.LSTM (BiLSTM).

    Present in BOTH Phase 1 AND Phase 2 -- BiLSTM is identical across phases.

    LSTM has 4 gates (input, forget, cell, output). Per timestep per direction:
      MACs = 4 x (input_size + hidden_size) x hidden_size
    """
    x = input[0]
    if module.batch_first:
        B, seq, _ = x.shape
    else:
        seq, B, _ = x.shape

    directions  = 2 if module.bidirectional else 1
    input_size  = module.input_size
    hidden_size = module.hidden_size

    macs_per_step  = 4 * (input_size + hidden_size) * hidden_size
    macs_per_step += 3 * hidden_size   # elementwise ops per cell

    total_macs = B * seq * directions * module.num_layers * macs_per_step
    return int(total_macs)


def _macs_pooling(module: nn.Module, input: Tuple, output: torch.Tensor) -> int:
    """
    MACs for pooling layers (MaxPool, AvgPool, AdaptivePool).
    Phase 1 uses AdaptiveMaxPool2d; Phase 2 uses AdaptiveAvgPool2d.
    Both handled here. Approximated as output_elements comparisons (1 MAC each).
    """
    return int(output.numel())


# ---------------------------------------------------------------------------
# Registry: map layer type -> MAC calculator
# ---------------------------------------------------------------------------

_MAC_CALCULATORS = {
    nn.Conv2d:             _macs_conv2d,               # DW + PW convs (Phase 2)
    nn.BatchNorm2d:        _macs_batchnorm2d,
    nn.Linear:             _macs_linear,               # MLP, attn projections, FC head
    nn.MultiheadAttention: _macs_multihead_attention,  # Phase 1 only
    nn.LSTM:               _macs_lstm,                 # Both phases (BiLSTM)
    nn.MaxPool2d:          _macs_pooling,
    nn.AvgPool2d:          _macs_pooling,
    nn.AdaptiveMaxPool2d:  _macs_pooling,              # Phase 1 global pool
    nn.AdaptiveAvgPool2d:  _macs_pooling,              # Phase 2 global pool
}


# ---------------------------------------------------------------------------
# Core counter
# ---------------------------------------------------------------------------

class _LayerRecord:
    """Stores per-layer stats collected during a forward pass."""
    __slots__ = ("name", "layer_type", "macs", "params", "input_shape", "output_shape")

    def __init__(self, name, layer_type, macs, params, input_shape, output_shape):
        self.name         = name
        self.layer_type   = layer_type
        self.macs         = macs
        self.params       = params
        self.input_shape  = input_shape
        self.output_shape = output_shape


def count_flops(
    model: nn.Module,
    x_img: torch.Tensor,
    x_seq: torch.Tensor,
    verbose: bool = False,
) -> Dict:
    """
    Count MACs and parameters for one forward pass of LightweightIDSModel.

    Phase 2 twin of ExistingImplementation's count_flops() -- identical
    implementation, works for any dual-branch model with forward(x_img, x_seq).

    Parameters
    ----------
    model : nn.Module
        LightweightIDSModel (or any dual-input model with forward(x_img, x_seq)).
    x_img : torch.Tensor, shape (B, 1, 28, 28)
    x_seq : torch.Tensor, shape (B, 64) or (B, 64, 1)
    verbose : bool
        If True, print per-layer breakdown during counting.

    Returns
    -------
    dict with keys:
      "total_macs"        -- total multiply-accumulate operations
      "total_flops"       -- total FLOPs (= 2 x MACs, common convention)
      "total_gmacs"       -- GMACs (10^9 MACs)
      "total_gflops"      -- GFLOPs (10^9 FLOPs)
      "total_params"      -- total trainable parameters
      "total_params_mb"   -- parameter memory in MB (float32)
      "by_type"           -- MACs aggregated by layer type name
      "by_module"         -- MACs aggregated by top-level module name
      "layers"            -- list of per-layer dicts (name, type, macs, params, shapes)
      "batch_size"        -- B used for counting
    """
    model.eval()
    records: List[_LayerRecord] = []
    hooks = []

    def _make_hook(mod_name: str, mod: nn.Module, calc_fn):
        def hook(module, inp, out):
            # Some modules return tuples (LSTM, MHA)
            out_tensor = out[0] if isinstance(out, (tuple, list)) else out
            inp_shape  = tuple(inp[0].shape) if inp else ()
            out_shape  = tuple(out_tensor.shape) if isinstance(out_tensor, torch.Tensor) else ()

            try:
                macs = calc_fn(module, inp, out)
            except Exception:
                macs = 0

            params = sum(p.numel() for p in module.parameters(recurse=False)
                         if p.requires_grad)

            rec = _LayerRecord(
                name         = mod_name,
                layer_type   = type(module).__name__,
                macs         = macs,
                params       = params,
                input_shape  = inp_shape,
                output_shape = out_shape,
            )
            records.append(rec)

            if verbose:
                print(f"  {mod_name:<55} {type(module).__name__:<25} "
                      f"MACs: {macs:>12,}  Params: {params:>10,}")

        return hook

    # Register hooks only for leaf-level modules that have calculators
    for mod_name, mod in model.named_modules():
        for layer_cls, calc_fn in _MAC_CALCULATORS.items():
            if type(mod) is layer_cls:
                h = mod.register_forward_hook(_make_hook(mod_name, mod, calc_fn))
                hooks.append(h)
                break

    # Single forward pass with no gradient
    with torch.no_grad():
        model(x_img, x_seq)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate
    total_macs   = sum(r.macs   for r in records)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # By layer type
    by_type: Dict[str, int] = {}
    for r in records:
        by_type[r.layer_type] = by_type.get(r.layer_type, 0) + r.macs

    # By top-level module (first segment of dotted name: "spatial", "temporal", "classifier")
    by_module: Dict[str, int] = {}
    for r in records:
        top = r.name.split(".")[0] if "." in r.name else r.name
        by_module[top] = by_module.get(top, 0) + r.macs

    layers = [
        {
            "name":         r.name,
            "type":         r.layer_type,
            "macs":         r.macs,
            "params":       r.params,
            "input_shape":  r.input_shape,
            "output_shape": r.output_shape,
        }
        for r in records
    ]

    return {
        "total_macs":      total_macs,
        "total_flops":     total_macs * 2,           # FLOPs = 2 x MACs
        "total_gmacs":     round(total_macs / 1e9, 6),
        "total_gflops":    round(total_macs * 2 / 1e9, 6),
        "total_params":    total_params,
        "total_params_mb": round(total_params * 4 / 1e6, 3),  # float32 = 4 bytes
        "by_type":         by_type,
        "by_module":       by_module,
        "layers":          layers,
        "batch_size":      int(x_img.shape[0]),
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_flops_report(report: Dict, show_layers: bool = False) -> None:
    """
    Pretty-print a FLOPs report produced by count_flops().

    Parameters
    ----------
    report     : dict returned by count_flops()
    show_layers: if True, print every individual layer row
    """
    B = report["batch_size"]
    print()
    print("=" * 70)
    print(f"  FLOPs / Parameter Report  (batch_size={B})")
    print("=" * 70)

    print(f"\n  Total MACs  : {report['total_macs']:>15,}")
    print(f"  Total GMACs : {report['total_gmacs']:>15.6f}")
    print(f"  Total FLOPs : {report['total_flops']:>15,}  (= 2 x MACs)")
    print(f"  Total GFLOPs: {report['total_gflops']:>15.6f}")
    print(f"\n  Total Params: {report['total_params']:>15,}")
    print(f"  Params (MB) : {report['total_params_mb']:>15.3f}  (float32)")

    print("\n  --- MACs by Top-Level Module ---")
    total = max(report["total_macs"], 1)
    for mod_name, macs in sorted(report["by_module"].items(),
                                  key=lambda x: -x[1]):
        pct = macs / total * 100
        bar = "\u2588" * int(pct / 2)
        print(f"    {mod_name:<20} {macs:>14,} MACs  ({pct:5.1f}%)  {bar}")

    print("\n  --- MACs by Layer Type ---")
    for layer_type, macs in sorted(report["by_type"].items(),
                                    key=lambda x: -x[1]):
        pct = macs / total * 100
        print(f"    {layer_type:<28} {macs:>14,} MACs  ({pct:5.1f}%)")

    if show_layers:
        print(f"\n  --- Per-Layer Breakdown ({len(report['layers'])} layers) ---")
        print(f"  {'Name':<50} {'Type':<22} {'MACs':>12}  {'Params':>10}")
        print(f"  {'-'*50} {'-'*22} {'-'*12}  {'-'*10}")
        for lyr in report["layers"]:
            print(f"  {lyr['name']:<50} {lyr['type']:<22} "
                  f"{lyr['macs']:>12,}  {lyr['params']:>10,}")

    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Convenience wrapper: count + print in one call
# ---------------------------------------------------------------------------

def profile_model(
    model: nn.Module,
    x_img: torch.Tensor,
    x_seq: torch.Tensor,
    show_layers: bool = False,
) -> Dict:
    """
    Count FLOPs and immediately print the report.

    Returns the report dict for further use.
    """
    report = count_flops(model, x_img, x_seq)
    print_flops_report(report, show_layers=show_layers)
    return report


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Add src/ to path so imports work when run directly
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.proposed_model import build_proposed_dos_model

    print("Building Ensemble-TranBiLSTM (DoS subset, 5 classes)...")
    print("  Spatial:  MobileNetV2Branch + EfficientNetB0Branch (soft ensemble)")
    print("  Temporal: LinearAttentionBlock (O(n)) + BiLSTM (identical to Phase 1)")
    model = build_proposed_dos_model(num_classes=5, dropout=0.0)  # dropout=0 for clean FLOPs
    model.eval()

    # Batch size 1 (single inference -- typical IoT fog node deployment scenario)
    x_img = torch.randn(1, 1, 28, 28)
    x_seq = torch.randn(1, 64)

    report = profile_model(model, x_img, x_seq, show_layers=True)

    # Per-module parameter breakdown (includes per-branch spatial)
    print("\n  --- Parameter Count by Module ---")
    param_counts = model.count_parameters()
    total = max(param_counts["total"], 1)
    for name, params in param_counts.items():
        if name == "total":
            print(f"  {'TOTAL':<30} {params:>12,}")
        else:
            pct = params / total * 100
            label = name.replace("_", " ").title()
            print(f"  {label:<30} {params:>12,}  ({pct:.1f}%)")

    # Phase 1 vs Phase 2 comparison
    print("\n  --- Phase 1 vs Phase 2 Complexity ---")
    print(f"  {'':35} {'Phase 1':>14}  {'Phase 2':>14}  {'Reduction':>10}")
    print(f"  {'Architecture':35} {'ResNet-18':>14}  {'MV2+EfB0 Ens':>14}  {'':>10}")
    print(f"  {'Recurrent Unit':35} {'BiLSTM':>14}  {'BiLSTM':>14}  {'identical':>10}")
    print(f"  {'Total Params':35} {'~11,334,117':>14}  {report['total_params']:>14,}  "
          f"{1 - report['total_params']/11_334_117:>10.1%}")
    print(f"  {'Spatial Params':35} {'~11,200,000':>14}  {param_counts['spatial']:>14,}  "
          f"{1 - param_counts['spatial']/11_200_000:>10.1%}")
    print(f"  {'Total GMACs':35} {'~0.1266':>14}  {report['total_gmacs']:>14.6f}  "
          f"{'(see report)':>10}")
    print(f"\n  Ensemble < single ResNet-18:  "
          f"{'VALIDATED' if report['total_params'] < 11_334_117 else 'CHECK CONFIG'}")
    