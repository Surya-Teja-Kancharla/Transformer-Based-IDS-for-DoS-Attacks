"""
comparison_report.py  --  ProposedImplementation/src/evaluation/
================================================================
Generates a quantitative Phase 1 vs Phase 2 comparison report.

Compares ExistingImplementation (Res-TranBiLSTM) against
ProposedImplementation (LightweightIDSModel) on four dimensions:

  1. Parameter counts     -- total and per-module breakdown
  2. FLOPs (MACs)         -- theoretical compute cost, batch_size=1
  3. Inference time       -- wall-clock latency (CPU, batch=1, 100 runs)
  4. Test metrics         -- Accuracy, Precision, Recall, F1 (macro)

WHAT THIS FILE PRODUCES:
  generate_report()        -- dict + console table
  plot_comparison_bars()   -- PNG: side-by-side bar chart of all metrics
  save_comparison_json()   -- JSON: full comparison report

CLI Usage (from ProposedImplementation/ directory):
  python src/evaluation/comparison_report.py \
      --existing-ckpt  ../ExistingImplementation/results/checkpoints/res_tranbilstm_dos_best.pth \
      --proposed-ckpt  results/checkpoints/lightweight_ctgan_dos_best.pth \
      --test-img       data/processed/X_test_img.npy \
      --test-seq       data/processed/X_test_seq.npy \
      --test-labels    data/processed/y_test.npy \
      --output         results/metrics/comparison_report.json

NOTE: Both models are evaluated on the SAME test set (Phase 2 test split)
to ensure a fair comparison on identical, unseen data.

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running as a standalone script from ProposedImplementation/
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.flops_counter import count_flops    # accurate, covers all layer types
from evaluation.metrics import compute_metrics, DOS_CLASS_NAMES

logger = logging.getLogger(__name__)


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
    Measure mean/std/min inference latency over n_runs forward passes.

    Includes warm-up runs to ensure GPU kernels are compiled and cache
    is warm before measurement begins.

    Parameters
    ----------
    model  : eval-mode model
    x_img  : (1, 1, 28, 28) -- single-sample input
    x_seq  : (1, 64)
    n_runs : number of timed forward passes (100 gives stable mean)
    device : measurement device (CPU for fair IoT-deployment comparison)

    Returns
    -------
    dict with "mean_ms", "std_ms", "min_ms"
    """
    model.eval()
    model.to(device)
    x_img = x_img.to(device)
    x_seq = x_seq.to(device)

    # Warm-up (10 passes, not measured)
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
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Run test-set evaluation; returns compute_metrics() result."""
    model.eval()
    all_preds, all_labels = [], []
    for x_img, x_seq, y in loader:
        x_img = x_img.to(device)
        x_seq = x_seq.to(device)
        preds = model(x_img, x_seq).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return compute_metrics(
        np.array(all_labels), np.array(all_preds), class_names=class_names
    )


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(
    existing_model: nn.Module,
    proposed_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    n_timing_runs: int = 100,
) -> Dict:
    """
    Generate full Phase 1 vs Phase 2 comparison report.

    Uses count_flops() from evaluation.flops_counter for accurate MACs
    (covers Conv2d groups, BatchNorm2d, Linear, LSTM, AdaptiveAvgPool2d).

    Parameters
    ----------
    existing_model  : loaded ExistingImplementation (ResTranBiLSTM)
    proposed_model  : loaded ProposedImplementation (LightweightIDSModel)
    test_loader     : shared test DataLoader (same split for both models)
    device          : evaluation device for test metrics
    output_path     : if given, save JSON report to this path
    class_names     : list of class name strings
    n_timing_runs   : number of timed passes for latency measurement

    Returns
    -------
    report dict with keys: "existing", "proposed", "comparison"
    """
    if class_names is None:
        class_names = DOS_CLASS_NAMES

    # Dummy single-sample inputs for FLOPs and timing (batch=1 simulates IoT deployment)
    dummy_img = torch.randn(1, 1, 28, 28)
    dummy_seq = torch.randn(1, 64)

    report: Dict = {}

    for tag, model in [("existing", existing_model), ("proposed", proposed_model)]:
        model.eval()
        cpu_model = model.cpu()

        # --- Parameter counts ---
        if hasattr(model, "count_parameters"):
            param_counts = model.count_parameters()
        else:
            total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts = {"total": total}

        # --- FLOPs (accurate, via forward hooks) ---
        flops_report = count_flops(cpu_model, dummy_img, dummy_seq)

        # --- Inference time (CPU, batch=1) ---
        timing = measure_inference_time(
            cpu_model, dummy_img, dummy_seq,
            n_runs=n_timing_runs,
            device=torch.device("cpu"),
        )

        # --- Restore to eval device for test metrics ---
        model.to(device)

        # --- Test metrics ---
        metrics = _evaluate(model, test_loader, device, class_names=class_names)

        report[tag] = {
            "parameters":      param_counts,
            "flops_macs":      flops_report["total_macs"],
            "flops_gmacs":     flops_report["total_gmacs"],
            "flops_gflops":    flops_report["total_gflops"],
            "params_mb":       flops_report["total_params_mb"],
            "inference_time":  timing,
            "test_accuracy":   round(metrics["accuracy"]  * 100, 2),
            "test_precision":  round(metrics["precision"] * 100, 2),
            "test_recall":     round(metrics["recall"]    * 100, 2),
            "test_f1":         round(metrics["f1"]        * 100, 2),
            "per_class":       metrics.get("per_class", {}),
        }

    # --- Summary ratios ---
    ep = report["existing"]["parameters"].get("total", 1)
    pp = report["proposed"]["parameters"].get("total", 1)
    ef = report["existing"]["flops_macs"]
    pf = report["proposed"]["flops_macs"]
    et = report["existing"]["inference_time"]["mean_ms"]
    pt = report["proposed"]["inference_time"]["mean_ms"]

    report["comparison"] = {
        "param_reduction_pct":  round((1 - pp / max(ep, 1)) * 100, 1),
        "flops_reduction_pct":  round((1 - pf / max(ef, 1)) * 100, 1),
        "accuracy_delta_pct":   round(
            report["proposed"]["test_accuracy"] - report["existing"]["test_accuracy"], 2
        ),
        "f1_delta_pct":         round(
            report["proposed"]["test_f1"] - report["existing"]["test_f1"], 2
        ),
        "speedup_factor":       round(et / max(pt, 1e-9), 2),
    }

    # --- Console table ---
    _print_report(report)

    # --- Save JSON ---
    if output_path:
        save_comparison_json(report, output_path)

    return report


def _print_report(report: Dict) -> None:
    """Print formatted console comparison table."""
    print("\n" + "=" * 65)
    print("  PHASE 1 vs PHASE 2 -- MODEL COMPARISON REPORT")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Phase 1 (Existing)':>16}  {'Phase 2 (Proposed)':>16}")
    print(f"  {'-'*30} {'-'*16}  {'-'*16}")

    e = report["existing"]
    p = report["proposed"]

    print(f"  {'Total Parameters':<30} {e['parameters'].get('total', 0):>16,}  {p['parameters'].get('total', 0):>16,}")
    print(f"  {'Params (MB)':<30} {e['params_mb']:>16.3f}  {p['params_mb']:>16.3f}")
    print(f"  {'FLOPs GMACs':<30} {e['flops_gmacs']:>16.6f}  {p['flops_gmacs']:>16.6f}")
    print(f"  {'Infer latency (ms, CPU)':<30} {e['inference_time']['mean_ms']:>15.3f}  {p['inference_time']['mean_ms']:>16.3f}")
    print(f"  {'Test Accuracy (%)':<30} {e['test_accuracy']:>16.2f}  {p['test_accuracy']:>16.2f}")
    print(f"  {'Test Precision (%)':<30} {e['test_precision']:>16.2f}  {p['test_precision']:>16.2f}")
    print(f"  {'Test Recall (%)':<30} {e['test_recall']:>16.2f}  {p['test_recall']:>16.2f}")
    print(f"  {'Test F1-Score (%)':<30} {e['test_f1']:>16.2f}  {p['test_f1']:>16.2f}")

    c = report["comparison"]
    print(f"\n  {'COMPARISON':}")
    print(f"  {'Parameter reduction':<30} {c['param_reduction_pct']:>+16.1f}%")
    print(f"  {'FLOPs reduction':<30} {c['flops_reduction_pct']:>+16.1f}%")
    print(f"  {'Accuracy delta':<30} {c['accuracy_delta_pct']:>+16.2f}%")
    print(f"  {'F1 delta':<30} {c['f1_delta_pct']:>+16.2f}%")
    print(f"  {'Inference speedup':<30} {c['speedup_factor']:>15.2f}x")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bars(
    report: Dict,
    save_path: Optional[str] = None,
    title: str = "Phase 1 vs Phase 2 -- Model Comparison",
) -> plt.Figure:
    """
    Side-by-side bar chart comparing key metrics between Phase 1 and Phase 2.

    Shows: Accuracy, Precision, Recall, F1 (as percentages).
    Parameter and FLOPs reduction are annotated as text labels.

    Parameters
    ----------
    report    : dict returned by generate_report()
    save_path : if given, save figure to this path
    """
    e = report["existing"]
    p = report["proposed"]
    c = report["comparison"]

    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    e_vals = [e["test_accuracy"], e["test_precision"], e["test_recall"], e["test_f1"]]
    p_vals = [p["test_accuracy"], p["test_precision"], p["test_recall"], p["test_f1"]]

    x     = np.arange(len(metric_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: classification metrics ---
    ax = axes[0]
    bars_e = ax.bar(x - width / 2, e_vals, width,
                    label="Phase 1 (Res-TranBiLSTM)", color="#2196F3", alpha=0.85)
    bars_p = ax.bar(x + width / 2, p_vals, width,
                    label="Phase 2 (LightweightIDS)",  color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(min(min(e_vals), min(p_vals)) - 2, 101)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("Classification Metrics", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Annotate bar values
    for bar in bars_e:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_p:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    # --- Right: efficiency metrics ---
    ax2 = axes[1]
    efficiency_labels = ["Param\nReduction (%)", "FLOPs\nReduction (%)", "Accuracy\nDelta (%)"]
    efficiency_vals   = [
        c["param_reduction_pct"],
        c["flops_reduction_pct"],
        c["accuracy_delta_pct"],
    ]
    colors = ["#FF9800" if v >= 0 else "#F44336" for v in efficiency_vals]
    bars_eff = ax2.bar(efficiency_labels, efficiency_vals, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Percentage", fontsize=10)
    ax2.set_title("Phase 2 Efficiency Gains", fontsize=11, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)

    for bar in bars_eff:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 h + (1 if h >= 0 else -2),
                 f"{h:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison bar chart saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def save_comparison_json(report: Dict, path: str) -> None:
    """
    Save the comparison report to a JSON file.

    Strips large 'per_class' dicts to keep the file concise.
    """
    def _strip(d):
        if isinstance(d, dict):
            return {k: _strip(v) for k, v in d.items() if k != "per_class"}
        return d

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_strip(report), f, indent=2)
    logger.info(f"Comparison report saved: {path}")
    print(f"\nReport saved: {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Phase 1 vs Phase 2 comparison report"
    )
    p.add_argument("--existing-ckpt", required=True,
                   help="Path to ExistingImplementation checkpoint .pth")
    p.add_argument("--proposed-ckpt", required=True,
                   help="Path to ProposedImplementation checkpoint .pth")
    p.add_argument("--test-img",    required=True,
                   help="Path to X_test_img.npy  (shape: N x 1 x 28 x 28)")
    p.add_argument("--test-seq",    required=True,
                   help="Path to X_test_seq.npy  (shape: N x 64)")
    p.add_argument("--test-labels", required=True,
                   help="Path to y_test.npy       (shape: N,)")
    p.add_argument("--output",      default="results/metrics/comparison_report.json")
    p.add_argument("--plots-dir",   default="results/plots",
                   help="Directory to save comparison bar chart PNG")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--timing-runs", type=int, default=100,
                   help="Number of forward passes for latency measurement")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Import Phase 2 model ---
    from models.proposed_model import build_proposed_dos_model

    # --- Import Phase 1 model (cross-directory) ---
    _phase1_src = str(
        Path(__file__).parent.parent.parent.parent
        / "ExistingImplementation" / "src"
    )
    sys.path.insert(0, _phase1_src)
    from models.res_tranbilstm import build_dos_model

    # --- Build and load checkpoints ---
    from utils.checkpointer import load_checkpoint

    existing_model = build_dos_model(num_classes=args.num_classes)
    proposed_model = build_proposed_dos_model(num_classes=args.num_classes)

    load_checkpoint(existing_model, args.existing_ckpt, device=device)
    load_checkpoint(proposed_model, args.proposed_ckpt, device=device)

    existing_model.to(device).eval()
    proposed_model.to(device).eval()

    # --- Build shared test loader ---
    X_img = np.load(args.test_img)
    X_seq = np.load(args.test_seq)
    y     = np.load(args.test_labels)

    t_img = torch.from_numpy(X_img).float()
    t_seq = torch.from_numpy(X_seq).float()
    if t_seq.dim() == 2:
        t_seq = t_seq.unsqueeze(-1)   # (N,64) -> (N,64,1)
    t_y = torch.from_numpy(y).long()

    test_loader = DataLoader(
        TensorDataset(t_img, t_seq, t_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,   # 0 required on Windows
    )

    # --- Generate report ---
    report = generate_report(
        existing_model = existing_model,
        proposed_model = proposed_model,
        test_loader    = test_loader,
        device         = device,
        output_path    = args.output,
        class_names    = DOS_CLASS_NAMES,
        n_timing_runs  = args.timing_runs,
    )

    # --- Plot comparison bars ---
    plot_comparison_bars(
        report,
        save_path = str(Path(args.plots_dir) / "comparison_bars.png"),
    )