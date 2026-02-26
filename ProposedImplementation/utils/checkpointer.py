"""
checkpointer.py — Save and load model checkpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int = 0,
    val_acc: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model state dict + metadata to path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":            epoch,
        "val_acc":          val_acc,
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info(f"Checkpoint saved: {path} (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint into model. Returns the full checkpoint dict.

    Parameters
    ----------
    strict : bool — if False, allow missing/unexpected keys (for partial loading).
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    epoch   = ckpt.get("epoch", 0)
    val_acc = ckpt.get("val_acc", 0.0)
    logger.info(f"Checkpoint loaded: {path} (epoch={epoch}, val_acc={val_acc:.4f})")
    return ckpt