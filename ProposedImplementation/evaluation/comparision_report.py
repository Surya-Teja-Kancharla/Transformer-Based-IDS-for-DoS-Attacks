"""
comparison_report.py  —  ProposedImplementation/src/evaluation/
================================================================
Generates a quantitative comparison report between ExistingImplementation
(Res-TranBiLSTM) and ProposedImplementation (LightweightIDSModel).

Compares:
  1. Parameter counts per module and total
  2. Theoretical FLOPs (multiply-accumulate ops) for one forward pass
  3. Inference time (wall-clock, averaged over N runs)
  4. Test metrics: Accuracy, Precision, Recall, F1

Usage (from ProposedImplementation/):
  python src/evaluation/comparison_report.py \
      --existing-ckpt  ../ExistingImplementation/results/checkpoints/res_tranbilstm_dos_best.pth \
      --proposed-ckpt  results/checkpoints/lightweight_ids_dos_best.pth \
      --test-img       ../ExistingImplementation/data/processed/X_test_img.npy \
      --test-seq       ../ExistingImplementation/data/processed/X_test_seq.npy \
      --test-labels    ../ExistingImplementation/data/processed/y_test.npy \
      --output         results/metrics/comparison_report.json

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running from ProposedImplementation/src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compute_metrics, DOS_CLASS_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FLOPs counter (manual, no external dependency)
# ---------------------------------------------------------------------------

def count_flops_conv2d(m: nn.Conv2d, input: torch.Tensor, output: torch.Tensor) -> int:
    """MACs for one Conv2d forward pass."""
    B, C_out, H_out, W_out = output.shape
    k_h, k_w = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
    C_in_per_group = m.in_channels // m.groups
    macs = B * C_out * H_out * W_out * C_in_per_group * k_h * k_w
    return int(macs)


def count_flops_linear(m: nn.Linear, input: torch.Tensor, output: torch.Tensor) -> int:
    """MACs for one Linear forward pass."""
    return int(input[0].numel() * m.out_features // (input[0].shape[-1] // m.in_features
               if input[0].shape[-1] != m.in_features else 1))


def measure_flops(model: nn.Module, x_img: torch.Tensor, x_seq: torch.Tensor) -> int:
    """
    Register forward hooks to count total MACs for a single forward pass.
    Returns integer MAC count.
    """
    total_macs = [0]
    hooks = []

    def hook_conv(m, inp, out):
        total_macs[0] += count_flops_conv2d(m, inp, out)

    def hook_linear(m, inp, out):
        n_elem = inp[0].numel()
        total_macs[0] += n_elem * m.out_features // m.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook_linear))

    with torch.no_grad():
        model(x_img, x_seq)

    for h in hooks:
        h.remove()

    return total_macs[0]


# ---------------------------------------------------------------------------
# Inference time measurement
# ---------------------------------------------------------------------------

def measure_inference_time(
    model: nn.Module,
    x_img: torch.Tensor,
    x_seq: torch.Tensor,
    n_runs: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Measure mean and std inference time over n_runs forward passes (ms).
    Includes GPU warm-up if CUDA available.
    """
    model.eval()
    model.to(device)
    x_img = x_img.to(device)
    x_seq = x_seq.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(x_img, x_seq)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x_img, x_seq)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
    }


# ---------------------------------------------------------------------------
# Model evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names=None,
) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    for x_img, x_seq, y in loader:
        x_img = x_img.to(device)
        x_seq = x_seq.to(device)
        preds = model(x_img, x_seq).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return compute_metrics(np.array(all_labels), np.array(all_preds), class_names=class_names)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    existing_model: nn.Module,
    proposed_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_path: Optional[str] = None,
    class_names=None,
) -> Dict:
    """
    Generate full comparison report dict.

    Parameters
    ----------
    existing_model : loaded ExistingImplementation model
    proposed_model : loaded ProposedImplementation model
    test_loader    : shared test DataLoader
    device         : torch device
    output_path    : if given, save JSON to this path
    class_names    : list of class name strings

    Returns
    -------
    report dict
    """
    if class_names is None:
        class_names = DOS_CLASS_NAMES

    # Dummy batch for FLOPs/timing
    dummy_img = torch.randn(1, 1, 28, 28)
    dummy_seq = torch.randn(1, 64)

    report = {}

    for name, model in [("existing", existing_model), ("proposed", proposed_model)]:
        model.eval()

        # Parameter counts
        param_counts = model.count_parameters() if hasattr(model, "count_parameters") else {
            "total": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

        # FLOPs
        flops = measure_flops(model.cpu(), dummy_img.cpu(), dummy_seq.cpu())

        # Inference time (CPU, batch=1)
        timing = measure_inference_time(model, dummy_img, dummy_seq, n_runs=100, device=torch.device("cpu"))

        # Test metrics
        metrics = evaluate(model, test_loader, device, class_names=class_names)

        report[name] = {
            "parameters":      param_counts,
            "flops_macs":      flops,
            "flops_gmacs":     round(flops / 1e9, 4),
            "inference_time":  timing,
            "test_accuracy":   round(metrics["accuracy"]  * 100, 2),
            "test_precision":  round(metrics["precision"] * 100, 2),
            "test_recall":     round(metrics["recall"]    * 100, 2),
            "test_f1":         round(metrics["f1"]        * 100, 2),
            "per_class":       metrics.get("per_class", {}),
        }

    # Summary ratios
    ep = report["existing"]["parameters"].get("total", 1)
    pp = report["proposed"]["parameters"].get("total", 1)
    ef = report["existing"]["flops_macs"]
    pf = report["proposed"]["flops_macs"]

    report["comparison"] = {
        "param_reduction_pct":   round((1 - pp / ep) * 100, 1),
        "flops_reduction_pct":   round((1 - pf / ef) * 100, 1),
        "accuracy_delta_pct":    round(
            report["proposed"]["test_accuracy"] - report["existing"]["test_accuracy"], 2
        ),
        "speedup_factor":        round(
            report["existing"]["inference_time"]["mean_ms"] /
            max(report["proposed"]["inference_time"]["mean_ms"], 1e-9), 2
        ),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON REPORT")
    print("=" * 60)
    for name in ("existing", "proposed"):
        r = report[name]
        print(f"\n  [{name.upper()}]")
        print(f"    Parameters  : {r['parameters'].get('total', 0):>12,}")
        print(f"    FLOPs (GMACs): {r['flops_gmacs']:>11.4f}")
        print(f"    Infer (ms)  : {r['inference_time']['mean_ms']:>11.3f} ± {r['inference_time']['std_ms']:.3f}")
        print(f"    Accuracy    : {r['test_accuracy']:>10.2f}%")
        print(f"    Precision   : {r['test_precision']:>10.2f}%")
        print(f"    Recall      : {r['test_recall']:>10.2f}%")
        print(f"    F1-Score    : {r['test_f1']:>10.2f}%")

    c = report["comparison"]
    print(f"\n  [COMPARISON]")
    print(f"    Param reduction : {c['param_reduction_pct']:>8.1f}%")
    print(f"    FLOPs reduction : {c['flops_reduction_pct']:>8.1f}%")
    print(f"    Accuracy delta  : {c['accuracy_delta_pct']:>+8.2f}%")
    print(f"    Speedup factor  : {c['speedup_factor']:>8.2f}×")
    print("=" * 60)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Remove large per_class dicts for clean JSON
        save = {k: {sk: sv for sk, sv in v.items() if sk != "per_class"}
                for k, v in report.items()}
        with open(output_path, "w") as f:
            json.dump(save, f, indent=2)
        print(f"\nReport saved: {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare Existing vs Proposed IDS models")
    p.add_argument("--existing-ckpt",  required=True)
    p.add_argument("--proposed-ckpt",  required=True)
    p.add_argument("--test-img",       required=True)
    p.add_argument("--test-seq",       required=True)
    p.add_argument("--test-labels",    required=True)
    p.add_argument("--output",         default="results/metrics/comparison_report.json")
    p.add_argument("--batch-size",     type=int, default=256)
    p.add_argument("--num-classes",    type=int, default=5)
    p.add_argument("--device",         default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Import models
    from models.proposed_model import build_proposed_dos_model
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent /
                           "ExistingImplementation" / "src"))
    from models.res_tranbilstm import build_dos_model

    # Load checkpoints
    existing_model = build_dos_model(num_classes=args.num_classes)
    proposed_model = build_proposed_dos_model(num_classes=args.num_classes)

    from utils.checkpointer import load_checkpoint
    load_checkpoint(existing_model, args.existing_ckpt, device=device)
    load_checkpoint(proposed_model, args.proposed_ckpt, device=device)

    existing_model.to(device).eval()
    proposed_model.to(device).eval()

    # Build test loader
    X_img = np.load(args.test_img)
    X_seq = np.load(args.test_seq)
    y     = np.load(args.test_labels)

    t_img = torch.from_numpy(X_img).float()
    t_seq = torch.from_numpy(X_seq).float()
    if t_seq.dim() == 2:
        t_seq = t_seq.unsqueeze(-1)
    t_y   = torch.from_numpy(y).long()

    test_loader = DataLoader(
        TensorDataset(t_img, t_seq, t_y),
        batch_size=args.batch_size, shuffle=False,
    )

    generate_report(
        existing_model=existing_model,
        proposed_model=proposed_model,
        test_loader=test_loader,
        device=device,
        output_path=args.output,
        class_names=DOS_CLASS_NAMES,
    )