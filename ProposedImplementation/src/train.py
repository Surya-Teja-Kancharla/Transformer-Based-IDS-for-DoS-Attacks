"""
train.py  —  ProposedImplementation
=====================================
Training entry point for the Lightweight IDS Model.

Usage (from ProposedImplementation/):
  python src/train.py --config configs/proposed_config.yaml
  python src/train.py --config configs/proposed_config.yaml --load-cache

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.seed   import set_seed
from utils.logger import setup_logger

from preprocessing.data_pipeline import DOSPreprocessingPipeline

from models.proposed_model import build_proposed_dos_model
from training.trainer      import ResTranBiLSTMTrainer   # same trainer, model-agnostic
from evaluation.metrics    import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_training_history,
    save_metrics_json,
    DOS_CLASS_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Lightweight IDS on CIC-IDS2017 DoS subset")
    p.add_argument("--config",      type=str, default="configs/proposed_config.yaml")
    p.add_argument("--csv",         type=str, default=None)
    p.add_argument("--load-cache",  action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--device",      type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths       = config["paths"]
    logger      = setup_logger("proposed_train", log_dir=paths["log_dir"])
    seed        = config["experiment"].get("seed", 42)
    set_seed(seed)

    device_str  = args.device or config["training"]["device"]
    device      = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Preprocessing (reuse ExistingImplementation pipeline)
    csv_path    = args.csv or config["data"]["csv_path"]
    proc_cfg    = config["data"]

    pipeline = DOSPreprocessingPipeline(
        csv_path      = csv_path,
        processed_dir = proc_cfg["processed_dir"],
        augmented_dir = proc_cfg.get("augmented_dir", "data/augmented"),
        config        = {
            "n_features":             proc_cfg["n_features"],
            "spatial_small":          proc_cfg["spatial_small"],
            "spatial_large":          proc_cfg["spatial_large"],
            "train_ratio":            proc_cfg["train_ratio"],
            "val_ratio":              proc_cfg["val_ratio"],
            "test_ratio":             proc_cfg["test_ratio"],
            "smote_target_per_class": config["smote_enn"].get("target_per_class", 20000),
            "random_state":           seed,
        },
    )

    dataset = pipeline.run(
        load_cache  = args.load_cache,
        force_rerun = args.force_rerun,
    )

    # Build proposed model
    num_classes = proc_cfg["num_classes"]
    model       = build_proposed_dos_model(
        num_classes=num_classes,
        dropout=config["training"]["dropout"],
    )
    logger.info(model.summary())

    # Trainer (model-agnostic — works for both existing and proposed)
    train_cfg = config["training"]
    trainer   = ResTranBiLSTMTrainer(
        model          = model,
        device         = device_str,
        learning_rate  = train_cfg["learning_rate"],
        batch_size     = train_cfg["batch_size"],
        patience       = train_cfg["patience"],
        checkpoint_dir = paths["checkpoint_dir"],
        log_dir        = paths["log_dir"],
    )

    train_loader, val_loader = trainer.make_dataloaders(
        X_img_train = dataset.X_train_img,
        X_seq_train = dataset.X_train_seq,
        y_train     = dataset.y_train,
        X_img_val   = dataset.X_val_img,
        X_seq_val   = dataset.X_val_seq,
        y_val       = dataset.y_val,
    )

    # Build test loader
    from torch.utils.data import DataLoader, TensorDataset
    t_img = torch.from_numpy(dataset.X_test_img).float()
    t_seq = torch.from_numpy(dataset.X_test_seq).float()
    if t_seq.dim() == 2:
        t_seq = t_seq.unsqueeze(-1)
    t_y   = torch.from_numpy(dataset.y_test).long()
    test_loader = DataLoader(
        TensorDataset(t_img, t_seq, t_y),
        batch_size=train_cfg["batch_size"] * 2, shuffle=False,
    )

    exp_name   = config["experiment"]["name"]
    max_epochs = args.epochs or train_cfg["max_epochs"]

    history = trainer.train(train_loader, val_loader, n_epochs=max_epochs, save_name=exp_name)
    trainer.load_best_checkpoint(exp_name)

    class_names = proc_cfg.get("class_names", DOS_CLASS_NAMES)
    metrics     = evaluate_model(model, test_loader, device, class_names=class_names)

    save_metrics_json(metrics, f"{paths['metrics_dir']}/{exp_name}_test_metrics.json")
    plot_confusion_matrix(
        np.array(metrics["y_true"]), np.array(metrics["y_pred"]),
        class_names=class_names,
        save_path=f"{paths['plots_dir']}/{exp_name}_confusion_matrix.png",
    )
    plot_per_class_metrics(metrics, class_names=class_names,
                           save_path=f"{paths['plots_dir']}/{exp_name}_per_class.png")
    plot_training_history(history, save_path=f"{paths['plots_dir']}/{exp_name}_history.png")

    logger.info(f"\nTest Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info(f"Test F1-Score  : {metrics['f1']*100:.2f}%")


if __name__ == "__main__":
    main()