"""
ctgan_handler.py  —  ProposedImplementation/src/preprocessing/
==============================================================
CTGAN-based data augmentation handler.

WHY CTGAN OVER SMOTE-ENN (Phase 2 justification):
  SMOTE-ENN (used in Phase 1 / base paper) generates synthetic samples by
  linear interpolation between existing minority samples. This creates
  repetitive, near-identical variations — the model overfits to a narrow
  region of the feature space and may fail on mutated or zero-day attacks.

  CTGAN (Conditional Tabular GAN — Xu et al., 2019) trains a Generator and
  Discriminator network to learn the *actual joint probability distribution*
  of the tabular feature space for each class. The Generator then samples
  from this learned distribution, producing:
    - Diverse synthetic samples that span the full distribution
    - Realistic feature co-dependencies (e.g. correlated packet length / rate)
    - Better coverage of rare and boundary-region attack patterns

  This directly addresses the "Rare Attack Diversity Problem" stated in the
  FYP proposal: minority classes (DoS Slowloris: 4,636 samples) benefit most
  from distribution-learning rather than interpolation.

STRATEGY (matching Phase 1 for fair comparison):
  1. Subsample majority classes above target_per_class (same as Phase 1)
  2. For each minority class below target_per_class:
       - Train a CTGAN on that class's real samples
       - Generate (target - real_count) synthetic samples
       - Append to training set
  3. NO ENN cleaning step — GAN samples are already distribution-matched
     and do not introduce the near-boundary noise that ENN removes from SMOTE

OUTPUT STRUCTURE (saved to data/augmented/ctgan_output/):
  X_train_balanced.npy        — full balanced feature matrix
  y_train_balanced.npy        — corresponding labels
  ctgan_balanced_sample.csv   — 50K-row human-readable sample
  class_distribution.txt      — per-class counts and percentages

INSTALL:
  pip install ctgan

Author: FYP ProposedImplementation
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

CLASS_LABEL_MAP = {
    0: "BENIGN",
    1: "DoS_slowloris",
    2: "DoS_Slowhttptest",
    3: "DoS_Hulk",
    4: "DoS_GoldenEye",
}


class CTGANBalancer:
    """
    CTGAN-based class balancer for the CIC-IDS2017 DoS dataset.

    Trains one CTGAN per minority class and generates synthetic tabular
    samples to reach target_per_class for each class.

    Parameters
    ----------
    target_per_class : int
        Target number of samples per class (default 15,000 to match Phase 1).
    epochs : int
        CTGAN training epochs per class. More = better quality, slower.
        50 is a good balance for ~5K samples on a modern GPU.
    batch_size : int
        CTGAN batch size. Must be divisible by pac parameter.
    random_state : int
        Global seed for reproducibility.
    verbose : bool
        Whether to print per-class progress.
    """

    def __init__(
        self,
        target_per_class: int = 15_000,
        epochs: int = 50,
        batch_size: int = 500,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.target_per_class = target_per_class
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.random_state     = random_state
        self.verbose          = verbose
        self.stats: Dict      = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset using CTGAN for minority classes and
        random subsampling for majority classes.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)  — normalised feature matrix
        y : np.ndarray, shape (N,)    — integer class labels

        Returns
        -------
        X_bal : np.ndarray, shape (N_balanced, F)
        y_bal : np.ndarray, shape (N_balanced,)
        """
        try:
            from ctgan import CTGAN
        except ImportError:
            raise ImportError(
                "ctgan is not installed. Run:\n"
                "  pip install ctgan\n"
                "then retry."
            )

        N_orig, F = X.shape
        classes, counts = np.unique(y, return_counts=True)
        t_start = time.time()

        # ── Header ────────────────────────────────────────────────
        print()
        print("=" * 62)
        print("  CTGAN Class Balancing (Phase 2)")
        print("=" * 62)
        print(f"  Input  : {N_orig:,} samples  |  {F} features")
        print(f"  Target : {self.target_per_class:,} samples per class")
        print(f"  CTGAN epochs per class : {self.epochs}")
        print()
        print(f"  {'Class':>6}  {'Name':<22} {'Before':>8}  {'Action':<20}")
        print("  " + "-" * 60)

        for cls, cnt in zip(classes.tolist(), counts.tolist()):
            name   = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
            target = self.target_per_class
            if cnt > target:
                action = f"subsample → {target:,}"
            elif cnt < target:
                action = f"CTGAN    → {target:,}"
            else:
                action = "keep as-is"
            print(f"  {cls:>6}  {name:<22} {cnt:>8,}  {action:<20}")
        print("  " + "-" * 60)
        print()

        self.stats["original_distribution"] = dict(zip(
            classes.tolist(), counts.tolist()
        ))

        # ── Step 1: Cap majority classes ──────────────────────────
        target  = self.target_per_class
        rng     = np.random.default_rng(self.random_state)

        keep_idx = []
        for cls, cnt in zip(classes.tolist(), counts.tolist()):
            idx = np.where(y == cls)[0]
            if cnt > target:
                idx = rng.choice(idx, size=target, replace=False)
            keep_idx.append(idx)
        keep_idx = np.concatenate(keep_idx)

        X_work = X[keep_idx].copy()
        y_work = y[keep_idx].copy()
        classes_w, counts_w = np.unique(y_work, return_counts=True)

        majority_capped = [c for c, cnt in zip(classes.tolist(), counts.tolist()) if cnt > target]
        if majority_capped:
            print(f"  [Step 1/2] Majority classes subsampled to {target:,}")
            for cls in majority_capped:
                name = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
                print(f"             {cls:>2}: {name:<22} → {target:,}")
            print()

        # ── Step 2: CTGAN for minority classes ────────────────────
        minority_classes = [
            (int(cls), int(cnt))
            for cls, cnt in zip(classes_w.tolist(), counts_w.tolist())
            if cnt < target
        ]
        n_minority = len(minority_classes)

        if n_minority == 0:
            print("  All classes already at target. No CTGAN needed.")
        else:
            print(f"  [Step 2/2] CTGAN — generating synthetic samples")
            print(f"             {n_minority} class(es) to augment")
            print()

        ctgan_elapsed = {}
        X_synthetic_all = []
        y_synthetic_all = []

        for i, (cls, real_cnt) in enumerate(minority_classes, 1):
            name        = CLASS_LABEL_MAP.get(cls, f"Class_{cls}")
            n_synthetic = target - real_cnt

            print(f"  [{i}/{n_minority}] Class {cls} ({name})")
            print(f"        Real samples : {real_cnt:,}")
            print(f"        Target       : {target:,}  (+{n_synthetic:,} synthetic)")
            print(f"        Training CTGAN ({self.epochs} epochs)...", flush=True)

            t_cls = time.time()

            # Extract this class's real samples as a DataFrame
            import pandas as pd
            cls_mask   = (y_work == cls)
            X_cls      = X_work[cls_mask]
            feat_cols  = [f"f{i:02d}" for i in range(F)]
            df_cls     = pd.DataFrame(X_cls, columns=feat_cols)

            # Train CTGAN on this class only
            ctgan = CTGAN(
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=False,
            )
            ctgan.fit(df_cls)

            # Sample synthetic rows
            df_synthetic = ctgan.sample(n_synthetic)
            X_syn        = df_synthetic[feat_cols].values.astype(np.float32)

            # Clip to valid normalised range [0, 1] — CTGAN can occasionally
            # generate out-of-range values for bounded tabular features
            X_syn = np.clip(X_syn, 0.0, 1.0)

            X_synthetic_all.append(X_syn)
            y_synthetic_all.append(np.full(n_synthetic, cls, dtype=np.int64))

            elapsed_cls = time.time() - t_cls
            ctgan_elapsed[cls] = elapsed_cls

            after_count = real_cnt + n_synthetic
            print(f"        Done  ({elapsed_cls:.1f}s)")
            print(f"        After : {after_count:,} samples  (+{n_synthetic:,} synthetic)")
            print()

        # ── Assemble final balanced dataset ───────────────────────
        parts_X = [X_work]
        parts_y = [y_work]
        if X_synthetic_all:
            parts_X.extend(X_synthetic_all)
            parts_y.extend(y_synthetic_all)

        X_bal = np.vstack(parts_X).astype(np.float32)
        y_bal = np.concatenate(parts_y).astype(np.int64)

        # Shuffle
        shuffle_idx = rng.permutation(len(y_bal))
        X_bal = X_bal[shuffle_idx]
        y_bal = y_bal[shuffle_idx]

        total_elapsed = time.time() - t_start
        ctgan_total   = sum(ctgan_elapsed.values())

        # ── Final summary ──────────────────────────────────────────
        res_classes, res_counts = np.unique(y_bal, return_counts=True)
        orig_dist = dict(zip(classes.tolist(), counts.tolist()))

        print("=" * 62)
        print("  CTGAN Balancing Complete")
        print("=" * 62)
        print(f"  {'Class':>6}  {'Name':<22} {'Before':>8}  {'After':>8}  {'Change':>8}")
        print("  " + "-" * 58)
        for cls, cnt in zip(res_classes.tolist(), res_counts.tolist()):
            name   = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
            before = orig_dist.get(int(cls), 0)
            change = cnt - before
            sign   = "+" if change >= 0 else ""
            print(f"  {cls:>6}  {name:<22} {before:>8,}  {cnt:>8,}  {sign}{change:>7,}")
        print("  " + "-" * 58)
        print(f"  {'TOTAL':>6}  {'':<22} {N_orig:>8,}  {len(y_bal):>8,}")
        print()

        imbalance_before = counts.max() / max(counts.min(), 1)
        imbalance_after  = res_counts.max() / max(res_counts.min(), 1)
        print(f"  Imbalance ratio : {imbalance_before:.1f}x  →  {imbalance_after:.2f}x")
        print(f"  Total time      : {total_elapsed:.1f}s  "
              f"(CTGAN training: {ctgan_total:.1f}s)")
        print("=" * 62)
        print()

        logger.info(f"CTGAN balancing completed in {total_elapsed:.1f}s")

        self.stats.update({
            "balanced_distribution": dict(zip(res_classes.tolist(), res_counts.tolist())),
            "original_size":   N_orig,
            "balanced_size":   len(y_bal),
            "elapsed_seconds": total_elapsed,
            "ctgan_elapsed":   ctgan_elapsed,
        })

        return X_bal, y_bal

    def get_balance_report(self) -> str:
        """Return a concise text summary of the last fit_resample call."""
        if not self.stats:
            return "No balancing has been run yet."
        lines = ["CTGAN Balance Report", "-" * 40]
        for cls, cnt in self.stats.get("balanced_distribution", {}).items():
            name = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
            lines.append(f"  Class {cls} ({name}): {cnt:,}")
        lines.append(f"Total: {self.stats.get('balanced_size', 0):,}")
        lines.append(f"Time : {self.stats.get('elapsed_seconds', 0):.1f}s")
        return "\n".join(lines)