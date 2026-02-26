"""
trainer.py
==========
Training loop for Res-TranBiLSTM with early stopping.

Paper (Wang et al., 2023) Section 4.2:
  "We use the Adam optimizer to train our model."
  "To avoid model overfitting, we use early-stopping criteria to train
   Res-TranBiLSTM — that is, we stop the training process once the
   validation accuracy begins decreasing."

Table 10 hyperparameters:
  Batch_Size     = 256
  Learning_Rate  = 0.0001
  Dropout        = 0.5

Author: FYP Implementation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping: halt training when validation accuracy stops improving.

    Paper: "we stop the training process once the validation accuracy
    begins decreasing."
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",   # "max" for accuracy, "min" for loss
        verbose: bool = True,
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.verbose    = verbose

        self.counter    = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info("  EarlyStopping: stopping training.")

        return self.should_stop

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class ResTranBiLSTMTrainer:
    """
    Full training engine for Res-TranBiLSTM.

    Handles:
      - Dual-branch DataLoader construction
      - Adam optimization (lr=0.0001, paper Table 10)
      - CrossEntropyLoss
      - Early stopping on validation accuracy
      - Checkpoint saving (best model)
      - TensorBoard logging
      - JSON metrics export

    Parameters
    ----------
    model : nn.Module
        Res-TranBiLSTM model instance.
    device : str
        'cuda' or 'cpu'.
    learning_rate : float
        Adam learning rate (0.0001 per paper).
    batch_size : int
        Mini-batch size (256 per paper).
    patience : int
        Early stopping patience (epochs without improvement).
    checkpoint_dir : str | Path
        Directory to save best model checkpoints.
    log_dir : str | Path
        Directory for TensorBoard logs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        patience: int = 10,
        checkpoint_dir: str = "results/checkpoints",
        log_dir: str = "logs",
    ) -> None:
        self.model     = model
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr        = learning_rate
        self.batch_size = batch_size

        self.model.to(self.device)

        # Optimizer: Adam (paper Table 10)
        self.optimizer = Adam(model.parameters(), lr=learning_rate)

        # LR scheduler: reduce on plateau (not in paper, but good practice)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )  # verbose param removed: deprecated in PyTorch 2.x (use get_last_lr())

        # Loss function: CrossEntropyLoss (standard for multi-class)
        self.criterion = nn.CrossEntropyLoss()

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, mode="max")

        # Paths
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        logger.info(f"Trainer initialized — device: {self.device}")
        logger.info(f"  LR={learning_rate}, batch_size={batch_size}, patience={patience}")

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------

    def make_dataloaders(
        self,
        X_img_train: np.ndarray,
        X_seq_train: np.ndarray,
        y_train: np.ndarray,
        X_img_val: np.ndarray,
        X_seq_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Builds train/val DataLoaders from numpy arrays.

        Parameters
        ----------
        X_img_* : np.ndarray, shape (N, 1, 28, 28)
        X_seq_* : np.ndarray, shape (N, 64) or (N, 64, 1)
        y_*     : np.ndarray, shape (N,)

        Returns
        -------
        train_loader, val_loader
        """
        def _make_dataset(X_img, X_seq, y):
            t_img = torch.from_numpy(X_img).float()
            t_seq = torch.from_numpy(X_seq).float()
            if t_seq.dim() == 2:
                t_seq = t_seq.unsqueeze(-1)   # (N, 64) → (N, 64, 1)
            t_y   = torch.from_numpy(y).long()
            return TensorDataset(t_img, t_seq, t_y)

        train_ds = _make_dataset(X_img_train, X_seq_train, y_train)
        val_ds   = _make_dataset(X_img_val,   X_seq_val,   y_val)

        # num_workers=0 on Windows — multiprocessing workers cause
        # "RuntimeError: An attempt has been made to start a new process"
        # when using spawn start method. 0 = load data in main process.
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        logger.info(
            f"DataLoaders: train={len(train_ds):,} | val={len(val_ds):,} "
            f"| batch={self.batch_size}"
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        save_name: str = "best_model",
    ) -> Dict[str, List[float]]:
        """
        Run training loop with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader   : DataLoader
        n_epochs     : int — maximum epochs
        save_name    : str — prefix for checkpoint filename

        Returns
        -------
        history dict with train/val loss and accuracy per epoch.
        """
        best_val_acc = 0.0
        t_start = time.time()

        logger.info(f"\nStarting training for up to {n_epochs} epochs...")
        logger.info("=" * 60)

        for epoch in range(1, n_epochs + 1):
            t_epoch = time.time()

            # Train
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self._eval_epoch(val_loader)

            # LR schedule
            self.scheduler.step(val_acc)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # TensorBoard logging
            self.writer.add_scalar("Loss/train",    train_loss, epoch)
            self.writer.add_scalar("Loss/val",      val_loss,   epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val",  val_acc,   epoch)
            self.writer.add_scalar(
                "LR", self.optimizer.param_groups[0]["lr"], epoch
            )

            epoch_time = time.time() - t_epoch
            logger.info(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(save_name, epoch, val_acc, val_loss)
                logger.info(f"  ✓ New best val_acc: {val_acc:.4f} — checkpoint saved")

            # Early stopping
            if self.early_stopping(val_acc):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - t_start
        logger.info(f"\nTraining complete in {total_time:.1f}s | Best val_acc: {best_val_acc:.4f}")

        self._save_history(save_name)
        self.writer.close()

        return self.history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for x_img, x_seq, y in loader:
            x_img = x_img.to(self.device)
            x_seq = x_seq.to(self.device)
            y     = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_img, x_seq)
            loss   = self.criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss    += loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += len(y)

        avg_loss = total_loss / total_samples
        avg_acc  = total_correct / total_samples
        return avg_loss, avg_acc

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for x_img, x_seq, y in loader:
            x_img = x_img.to(self.device)
            x_seq = x_seq.to(self.device)
            y     = y.to(self.device)

            logits = self.model(x_img, x_seq)
            loss   = self.criterion(logits, y)

            total_loss    += loss.item() * len(y)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += len(y)

        return total_loss / total_samples, total_correct / total_samples

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        val_acc: float,
        val_loss: float,
    ) -> None:
        path = self.checkpoint_dir / f"{name}_best.pth"
        torch.save(
            {
                "epoch":      epoch,
                "val_acc":    val_acc,
                "val_loss":   val_loss,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_best_checkpoint(self, name: str) -> int:
        """Load best checkpoint. Returns epoch number."""
        path = self.checkpoint_dir / f"{name}_best.pth"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(val_acc={ckpt['val_acc']:.4f})"
        )
        return ckpt["epoch"]

    def _save_history(self, name: str) -> None:
        path = self.checkpoint_dir / f"{name}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {path}")