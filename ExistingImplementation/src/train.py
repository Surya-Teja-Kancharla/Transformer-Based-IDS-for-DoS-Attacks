"""
train.py
========
Main training entry point for Res-TranBiLSTM (ExistingImplementation).

Usage:
  # From ExistingImplementation/ directory:
  python src/train.py --config configs/dos_config.yaml
  python src/train.py --config configs/dos_config.yaml --csv data/raw/Wednesday-workingHours.csv
  python src/train.py --config configs/dos_config.yaml --load-cache  # skip preprocessing

Author: FYP Implementation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure src/ is on path when running from ExistingImplementation/
sys.path.insert(0, str(Path(__file__).parent))

from utils.seed   import set_seed
from utils.logger import setup_logger

from preprocessing.data_pipeline import DOSPreprocessingPipeline, PIPELINE_CONFIG

from models.res_tranbilstm import build_dos_model
from training.trainer      import ResTranBiLSTMTrainer
from evaluation.flops_counter import count_flops, print_flops_report
from evaluation.metrics    import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_training_history,
    save_metrics_json,
    DOS_CLASS_NAMES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Res-TranBiLSTM on CIC-IDS2017 DoS subset")
    p.add_argument("--config",     type=str, default="configs/dos_config.yaml")
    p.add_argument("--csv",        type=str, default=None,
                   help="Path to Wednesday-workingHours CSV (overrides config)")
    p.add_argument("--load-cache", action="store_true",
                   help="Skip preprocessing, load cached .npy files")
    p.add_argument("--force-rerun", action="store_true",
                   help="Force re-run preprocessing even if cache exists")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override max_epochs from config")
    p.add_argument("--device",     type=str, default=None,
                   help="Override device (cuda/cpu)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    # --- Paths ---
    paths        = config["paths"]
    log_dir      = paths["log_dir"]
    ckpt_dir     = paths["checkpoint_dir"]
    metrics_dir  = paths["metrics_dir"]
    plots_dir    = paths["plots_dir"]

    # --- Logger ---
    logger = setup_logger("train", log_dir=log_dir)
    logger.info("=" * 60)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Config:     {args.config}")

    # --- Seed ---
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # --- Device ---
    device_str = args.device or config["training"]["device"]
    device     = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Preprocessing ---
    csv_path = args.csv or config["data"]["csv_path"]
    proc_cfg = config["data"]

    # DOSPreprocessingPipeline accepts: base_dir, config, verbose
    # csv_path is passed to run(), not to __init__
    # The base_dir should be set so that data/raw/ resolves correctly.
    # We pass base_dir as the directory containing data/ (i.e. ExistingImplementation/)
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ExistingImplementation/

    pipeline = DOSPreprocessingPipeline(
        base_dir = base_dir,
        config   = {
            "n_features":             proc_cfg["n_features"],
            "spatial_small":          proc_cfg["spatial_small"],
            "spatial_large":          proc_cfg["spatial_large"],
            "train_ratio":            proc_cfg["train_ratio"],
            "val_ratio":              proc_cfg["val_ratio"],
            "test_ratio":             proc_cfg["test_ratio"],
            "smote_target_per_class": config["smote_enn"].get("target_per_class", 15000),
            "smote_sampling_strategy": config["smote_enn"].get("sampling_strategy", "fixed"),
            "random_state":           seed,
        },
        verbose = True,
    )

    # run() accepts: csv_path, force_rerun
    # --load-cache: if True and cache exists, run() auto-loads from cache (force_rerun=False)
    # --force-rerun: explicitly ignore cache
    force_rerun = args.force_rerun or (not args.load_cache)

    logger.info("Running preprocessing pipeline...")
    dataset = pipeline.run(
        csv_path    = csv_path,
        force_rerun = force_rerun,
    )
    logger.info(
        f"Data shapes — "
        f"train_img: {dataset.X_train_img.shape}, "
        f"val_img: {dataset.X_val_img.shape}, "
        f"test_img: {dataset.X_test_img.shape}"
    )

    # --- Model ---
    num_classes = proc_cfg["num_classes"]
    dropout     = config["training"]["dropout"]
    model       = build_dos_model(num_classes=num_classes, dropout=dropout)
    logger.info(model.summary())

    # --- Trainer ---
    train_cfg   = config["training"]
    max_epochs  = args.epochs or train_cfg["max_epochs"]
    trainer     = ResTranBiLSTMTrainer(
        model          = model,
        device         = device_str,
        learning_rate  = train_cfg["learning_rate"],
        batch_size     = train_cfg["batch_size"],
        patience       = train_cfg["patience"],
        checkpoint_dir = ckpt_dir,
        log_dir        = log_dir,
    )

    # --- DataLoaders ---
    train_loader, val_loader = trainer.make_dataloaders(
        X_img_train = dataset.X_train_img,
        X_seq_train = dataset.X_train_seq,
        y_train     = dataset.y_train,
        X_img_val   = dataset.X_val_img,
        X_seq_val   = dataset.X_val_seq,
        y_val       = dataset.y_val,
    )

    # Also build test loader
    from torch.utils.data import DataLoader, TensorDataset
    t_img = torch.from_numpy(dataset.X_test_img).float()
    t_seq = torch.from_numpy(dataset.X_test_seq).float()
    if t_seq.dim() == 2:
        t_seq = t_seq.unsqueeze(-1)
    t_y  = torch.from_numpy(dataset.y_test).long()
    test_loader = DataLoader(
        TensorDataset(t_img, t_seq, t_y),
        batch_size  = train_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = 0,   # 0 required on Windows
    )

    # --- FLOPs / Parameter Profile ---
    # Measured at batch_size=1 on CPU to simulate IoT single-inference
    logger.info("\nProfiling model FLOPs and parameters...")
    _x_img = torch.zeros(1, 1, 28, 28)
    _x_seq = torch.zeros(1, 64)
    _model_cpu = model.cpu()
    flops_report = count_flops(_model_cpu, _x_img, _x_seq)
    print_flops_report(flops_report, show_layers=False)
    model = _model_cpu.to(device)  # restore to GPU for training

    # --- Train ---
    logger.info(f"\nStarting training for up to {max_epochs} epochs...")
    history = trainer.train(
        train_loader = train_loader,
        val_loader   = val_loader,
        n_epochs     = max_epochs,
        save_name    = config["experiment"]["name"],
    )

    # --- Load best model for evaluation ---
    trainer.load_best_checkpoint(config["experiment"]["name"])

    # --- Evaluate ---
    logger.info("\nEvaluating on test set...")
    class_names = proc_cfg.get("class_names", DOS_CLASS_NAMES)
    metrics     = evaluate_model(
        model       = model,
        test_loader = test_loader,
        device      = device,
        class_names = class_names,
    )

    # --- Save results ---
    exp_name = config["experiment"]["name"]

    save_metrics_json(metrics, f"{metrics_dir}/{exp_name}_test_metrics.json")

    plot_confusion_matrix(
        np.array(metrics["y_true"]),
        np.array(metrics["y_pred"]),
        class_names = class_names,
        save_path   = f"{plots_dir}/{exp_name}_confusion_matrix.png",
        normalize   = True,
        title       = "Res-TranBiLSTM — Confusion Matrix",
    )

    plot_per_class_metrics(
        metrics,
        class_names = class_names,
        save_path   = f"{plots_dir}/{exp_name}_per_class.png",
        title       = "Res-TranBiLSTM — Per-Class Performance",
    )

    plot_training_history(
        history,
        save_path = f"{plots_dir}/{exp_name}_training_history.png",
    )

    # --- Save FLOPs report ---
    import json as _json
    flops_save_path = f"{metrics_dir}/{exp_name}_flops_report.json"
    with open(flops_save_path, "w") as _f:
        _json.dump(
            {k: v for k, v in flops_report.items() if k != "layers"},
            _f, indent=2
        )
    flops_layers_path = f"{metrics_dir}/{exp_name}_flops_layers.json"
    with open(flops_layers_path, "w") as _f:
        _json.dump(flops_report["layers"], _f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS:")
    logger.info("FINAL TEST RESULTS:")
    logger.info(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"  Recall:    {metrics['recall']*100:.2f}%")
    logger.info(f"  F1-Score:  {metrics['f1']*100:.2f}%")
    logger.info("=" * 60)
    logger.info("\nMODEL COMPLEXITY:")
    logger.info(f"  Total Parameters : {flops_report['total_params']:,}")
    logger.info(f"  Params (MB)      : {flops_report['total_params_mb']:.3f} MB")
    logger.info(f"  Total MACs       : {flops_report['total_macs']:,}")
    logger.info(f"  Total GMACs      : {flops_report['total_gmacs']:.6f}")
    logger.info(f"  Total FLOPs      : {flops_report['total_flops']:,}  (= 2 x MACs)")
    logger.info(f"  Total GFLOPs     : {flops_report['total_gflops']:.6f}")
    logger.info(f"  Report saved     : {flops_save_path}")
    logger.info("=" * 60)
    logger.info(f"Target (paper): 99.15% accuracy on CIC-IDS2017")


if __name__ == "__main__":
    main()