"""
smote_enn.py
============
Implements the SMOTE-ENN class imbalance handling described in the paper
(Wang et al., 2023, Section 3.2.2):

  "We use the SMOTE algorithm. Specifically, for a minority class sample xi,
   K-nearest neighbor (KNN) method is used to find the k minority samples
   closest to xi. Then, one of the KNN points is randomly selected, and a
   new sample is generated using Eq.(1): x_new = xi + (x̂ + xi) * δ

   However, the disadvantage of SMOTE is that the generated minority class
   samples are apt to overlap with the surrounding majority class samples.
   The ENN algorithm is used to delete data that overlaps."

For our DoS-only study (Wednesday-workingHours.csv), class distribution:
  - BENIGN:           ~439,099  (very dominant)
  - DoS Hulk:         ~229,198  (large)
  - DoS GoldenEye:    ~10,289   (medium)
  - DoS slowloris:    ~5,771    (small)
  - DoS Slowhttptest: ~5,485    (small)
  Heartbleed: 11 samples — excluded due to extreme rarity

SMOTE-ENN Strategy:
  1. SMOTE: Oversample minority classes to reach target counts
  2. ENN: Remove noisy/borderline samples via edited nearest neighbor

Paper's processed result for CIC-IDS2017 (Table 4):
  All 7 classes balanced to ~25,000-26,000 samples each.
  For our 5-class DoS subset: target ~20,000-25,000 per class.

Author: FYP Implementation
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class distribution target (paper-informed, scaled for DoS subset)
# ---------------------------------------------------------------------------

# After SMOTE-ENN, paper achieves ~14-15% per class for 7-class problem.
# For 5-class DoS subset, we target ~20% per class (~25,000 samples each).
# These are soft targets — actual counts depend on SMOTE-ENN dynamics.

DEFAULT_TARGET_PER_CLASS = 25_000   # Adjust if memory is constrained

# For RTX 2050 (4GB VRAM) constraint: keep total dataset manageable
# Wednesday DoS after filter: ~260K samples → target ~25K per class → ~125K total
MEMORY_SAFE_TARGET = 20_000  # Conservative default for 4GB GPU


# ---------------------------------------------------------------------------
# DoS SMOTE-ENN Handler
# ---------------------------------------------------------------------------

class DOSSMOTEENNHandler:
    """
    Applies SMOTE-ENN to balance the DoS-filtered CIC-IDS2017 dataset.

    Pipeline:
      1. Analyze class distribution
      2. Compute SMOTE sampling strategy (oversample minorities)
      3. Apply SMOTE to generate synthetic minority samples
      4. Apply ENN to remove noisy/borderline samples
      5. Report final distribution

    Per paper (Table 4 logic): all classes brought to approximately
    equal representation, with ENN cleaning ensuring clean boundaries.

    Parameters
    ----------
    target_per_class : int
        Desired number of samples per class BEFORE ENN cleaning.
        ENN will reduce this slightly by removing noisy samples.
    k_neighbors_smote : int
        Number of nearest neighbors for SMOTE synthesis (default 5).
    n_neighbors_enn : int
        Number of nearest neighbors for ENN cleaning (default 3).
    sampling_strategy : str | dict
        'auto'   → oversample all minorities to match majority
        'not majority' → oversample all except majority class
        dict     → explicit {class_id: target_count} mapping
    n_jobs : int
        Parallel jobs for KNN computation (-1 = all cores).
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        target_per_class: int = MEMORY_SAFE_TARGET,
        k_neighbors_smote: int = 5,
        n_neighbors_enn: int = 3,
        sampling_strategy: Union[str, Dict[int, int]] = "not majority",
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.target_per_class = target_per_class
        self.k_neighbors_smote = k_neighbors_smote
        self.n_neighbors_enn = n_neighbors_enn
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._smote_enn: Optional[SMOTEENN] = None
        self.stats: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE-ENN to X, y with class-by-class progress tracking.

        Strategy
        --------
        Instead of one opaque SMOTEENN call (which shows 0% for 60+ min),
        we run SMOTE per minority class one at a time, printing live
        progress after each class completes. ENN cleaning runs once
        globally at the end.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
        y : np.ndarray, shape (N,)

        Returns
        -------
        X_res : np.ndarray, shape (N_balanced, F)
        y_res : np.ndarray, shape (N_balanced,)
        """
        from sklearn.neighbors import NearestNeighbors

        N_orig, F = X.shape
        classes, counts = np.unique(y, return_counts=True)
        t_pipeline_start = time.time()

        # ── Header ───────────────────────────────────────────────────
        print()
        print("=" * 62)
        print("  SMOTE-ENN Class Balancing")
        print("=" * 62)
        print(f"  Input  : {N_orig:,} samples  |  {F} features")
        print(f"  Target : {self.target_per_class:,} samples per class")
        print()
        print(f"  {'Class':>6}  {'Name':<22} {'Before':>8}  {'Action':<18}")
        print("  " + "-" * 58)

        CLASS_LABEL_MAP = {
            0: "BENIGN",
            1: "DoS_slowloris",
            2: "DoS_Slowhttptest",
            3: "DoS_Hulk",
            4: "DoS_GoldenEye",
        }

        for cls, cnt in zip(classes.tolist(), counts.tolist()):
            name = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
            target = self.target_per_class
            if self.sampling_strategy == "fixed":
                if cnt > target:
                    action = f"subsample → {target:,}"
                elif cnt < target:
                    action = f"SMOTE    → {target:,}"
                else:
                    action = "keep as-is"
            else:
                action = "SMOTE (auto)"
            print(f"  {cls:>6}  {name:<22} {cnt:>8,}  {action:<18}")
        print("  " + "-" * 58)
        print()

        self.stats["original_distribution"] = dict(zip(
            classes.tolist(), counts.tolist()
        ))

        # ── Step 1: Cap majority classes (fixed mode) ─────────────────
        if self.sampling_strategy == "fixed":
            target = self.target_per_class
            keep_indices = []
            rng = np.random.default_rng(self.random_state)
            for cls, cnt in zip(classes.tolist(), counts.tolist()):
                cls_idx = np.where(y == cls)[0]
                if cnt > target:
                    cls_idx = rng.choice(cls_idx, size=target, replace=False)
                keep_indices.append(cls_idx)
            keep_indices = np.concatenate(keep_indices)
            X = X[keep_indices]
            y = y[keep_indices]
            classes, counts = np.unique(y, return_counts=True)
            print(f"  [Step 1/3] Majority classes capped at {target:,}")
            for cls, cnt in zip(classes.tolist(), counts.tolist()):
                name = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
                print(f"             {cls:>2}: {name:<22} {cnt:>8,}")
            print()

        # ── Step 2: SMOTE — one minority class at a time ──────────────
        strategy = self._build_sampling_strategy(y, classes, counts)

        # Identify which classes need synthetic samples
        minority_classes = []
        if isinstance(strategy, dict):
            minority_classes = [(cls, tgt) for cls, tgt in strategy.items()]
        else:
            # "auto" / "not majority": all non-majority classes
            majority_count = int(counts.max())
            minority_classes = [
                (int(cls), majority_count)
                for cls, cnt in zip(classes.tolist(), counts.tolist())
                if cnt < majority_count
            ]

        n_minority = len(minority_classes)
        print(f"  [Step 2/3] SMOTE — generating synthetic samples")
        print(f"             {n_minority} class(es) to oversample")
        print()

        X_augmented = X.copy()
        y_augmented = y.copy()
        smote_elapsed = {}

        for i, (cls, tgt) in enumerate(minority_classes, 1):
            name = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
            cls_count = int(counts[classes.tolist().index(cls)])
            n_synthetic = tgt - cls_count

            print(f"  [{i}/{n_minority}] Class {cls} ({name})")
            print(f"        Before : {cls_count:,} real samples")
            print(f"        Target : {tgt:,}  (+{n_synthetic:,} synthetic)")
            print(f"        Running SMOTE...", end="", flush=True)

            t_cls = time.time()

            # Binary SMOTE: this class vs everything else
            # Build a binary label array: 1 = this class, 0 = rest
            y_binary = (y_augmented == cls).astype(np.int64)
            current_minority = int(y_binary.sum())
            current_majority = int(len(y_binary) - current_minority)

            # SMOTE sampling strategy for binary case
            binary_strategy = {1: tgt}

            smote = SMOTE(
                sampling_strategy=binary_strategy,
                k_neighbors=NearestNeighbors(
                    n_neighbors=self.k_neighbors_smote,
                    n_jobs=self.n_jobs,
                ),
                random_state=self.random_state + i,
            )

            X_sm, y_sm_binary = smote.fit_resample(X_augmented, y_binary)
            elapsed_cls = time.time() - t_cls
            smote_elapsed[cls] = elapsed_cls

            # Extract only the synthetic samples (new rows appended by SMOTE)
            n_new = len(X_sm) - len(X_augmented)
            X_synthetic = X_sm[len(X_augmented):]
            y_synthetic = np.full(n_new, cls, dtype=np.int64)

            # Append synthetic samples to working arrays
            X_augmented = np.vstack([X_augmented, X_synthetic])
            y_augmented = np.concatenate([y_augmented, y_synthetic])

            after_count = int((y_augmented == cls).sum())
            print(f" done  ({elapsed_cls:.1f}s)")
            print(f"        After  : {after_count:,} samples  "
                  f"(+{n_new:,} synthetic added)")
            print()

        smote_total = sum(smote_elapsed.values())
        print(f"  SMOTE complete — {smote_total:.1f}s total")
        print()

        # ── Step 3: ENN — global noise removal ────────────────────────
        print(f"  [Step 3/3] ENN — removing noisy/borderline samples")
        print(f"             Input: {len(y_augmented):,} samples (real + synthetic)")
        print(f"             Running ENN...", end="", flush=True)

        t_enn = time.time()
        enn = EditedNearestNeighbours(
            n_neighbors=self.n_neighbors_enn,
            n_jobs=self.n_jobs,
        )
        X_res, y_res = enn.fit_resample(X_augmented, y_augmented)
        enn_elapsed = time.time() - t_enn
        n_removed = len(y_augmented) - len(y_res)

        print(f" done  ({enn_elapsed:.1f}s)")
        print(f"             Removed {n_removed:,} noisy samples")
        print(f"             Output : {len(y_res):,} samples")
        print()

        # ── Final summary ─────────────────────────────────────────────
        total_elapsed = time.time() - t_pipeline_start
        res_classes, res_counts = np.unique(y_res, return_counts=True)

        print("=" * 62)
        print("  SMOTE-ENN Complete")
        print("=" * 62)
        print(f"  {'Class':>6}  {'Name':<22} {'Before':>8}  {'After':>8}  {'Change':>8}")
        print("  " + "-" * 58)

        orig_dist = dict(zip(classes.tolist(), counts.tolist()))
        for cls, cnt in zip(res_classes.tolist(), res_counts.tolist()):
            name = CLASS_LABEL_MAP.get(int(cls), f"Class_{cls}")
            before = orig_dist.get(int(cls), 0)
            change = cnt - before
            sign = "+" if change >= 0 else ""
            print(f"  {cls:>6}  {name:<22} {before:>8,}  {cnt:>8,}  "
                  f"{sign}{change:>7,}")
        print("  " + "-" * 58)
        print(f"  {'TOTAL':>6}  {'':<22} {N_orig:>8,}  {len(y_res):>8,}")
        print()

        orig_counts_arr = np.array([orig_dist.get(int(c), 1) for c in res_classes])
        imbalance_before = counts.max() / max(counts.min(), 1)
        imbalance_after  = res_counts.max() / max(res_counts.min(), 1)
        print(f"  Imbalance ratio : {imbalance_before:.1f}x  →  {imbalance_after:.2f}x")
        print(f"  Total time      : {total_elapsed:.1f}s  "
              f"(SMOTE: {smote_total:.1f}s  |  ENN: {enn_elapsed:.1f}s)")
        print("=" * 62)
        print()

        logger.info(f"SMOTE-ENN completed in {total_elapsed:.1f}s")

        self.stats["balanced_distribution"] = dict(zip(
            res_classes.tolist(), res_counts.tolist()
        ))
        self.stats["original_size"]    = N_orig
        self.stats["balanced_size"]    = len(y_res)
        self.stats["elapsed_seconds"]  = total_elapsed
        self.stats["smote_elapsed"]    = smote_elapsed
        self.stats["enn_elapsed"]      = enn_elapsed

        return X_res.astype(np.float32), y_res.astype(np.int64)

    def fit_resample_chunked(
        self,
        X: np.ndarray,
        y: np.ndarray,
        chunk_size: int = 50_000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-efficient variant: applies SMOTE to each minority class
        separately in chunks, then applies ENN globally.

        Useful when RAM is limited and full SMOTE-ENN would OOM.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        chunk_size : int
            Max samples per class for SMOTE fitting.

        Returns
        -------
        X_res, y_res
        """
        logger.info("Running chunked SMOTE-ENN (memory-efficient mode)...")

        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        # Subsample majority class if too large
        majority_cls = classes[np.argmax(counts)]
        if counts.max() > chunk_size * 2:
            logger.info(
                f"  Subsampling majority class {majority_cls}: "
                f"{counts.max():,} → {chunk_size:,}"
            )
            majority_mask = y == majority_cls
            minority_mask = ~majority_mask

            majority_idx = np.where(majority_mask)[0]
            rng = np.random.default_rng(self.random_state)
            sampled_idx = rng.choice(majority_idx, size=chunk_size, replace=False)
            minority_idx = np.where(minority_mask)[0]
            keep_idx = np.concatenate([sampled_idx, minority_idx])

            X = X[keep_idx]
            y = y[keep_idx]

        return self.fit_resample(X, y)

    # ------------------------------------------------------------------
    # Strategy builder
    # ------------------------------------------------------------------

    def _build_sampling_strategy(
        self,
        y: np.ndarray,
        classes: np.ndarray,
        counts: np.ndarray,
    ) -> Union[str, Dict[int, int]]:
        """
        Build the SMOTE sampling_strategy dict.

        "fixed" mode: caps ALL classes (including majority) at
        target_per_class by subsampling majority before SMOTE.
        This prevents the 1.5M sample explosion that occurs when
        "not majority" tries to match the 351K BENIGN class.

        Other modes passed through directly to imbalanced-learn.
        """
        if isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy

        if self.sampling_strategy in ("auto", "not majority", "all"):
            return self.sampling_strategy

        # "fixed" mode: explicit per-class target for ALL classes
        # SMOTE can only oversample, so we handle majority subsampling
        # in fit_resample() before calling SMOTE.
        target = self.target_per_class
        strategy = {}
        for cls, cnt in zip(classes, counts):
            if int(cnt) < target:
                # Minority: oversample up to target
                strategy[int(cls)] = target
            # Majority classes at or above target: left as-is
            # (already subsampled before this call in fit_resample)
        return strategy

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_balance_report(self) -> str:
        """Return a human-readable balance report string."""
        if not self.stats:
            return "No resampling has been performed yet."

        lines = ["SMOTE-ENN Balance Report", "=" * 40]
        orig = self.stats.get("original_distribution", {})
        bal = self.stats.get("balanced_distribution", {})

        lines.append(f"{'Class':<10} {'Before':>12} {'After':>12} {'Change':>10}")
        lines.append("-" * 46)
        all_classes = sorted(set(list(orig.keys()) + list(bal.keys())))
        for cls in all_classes:
            before = orig.get(cls, 0)
            after = bal.get(cls, 0)
            change = after - before
            sign = "+" if change >= 0 else ""
            lines.append(
                f"  {cls:<8} {before:>12,} {after:>12,} "
                f"{sign}{change:>9,}"
            )

        lines.append("-" * 46)
        lines.append(
            f"  {'TOTAL':<8} "
            f"{self.stats['original_size']:>12,} "
            f"{self.stats['balanced_size']:>12,}"
        )
        lines.append(
            f"\n  Time elapsed: {self.stats.get('elapsed_seconds', 0):.1f}s"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience function for direct use in data_pipeline.py
# ---------------------------------------------------------------------------

def apply_smote_enn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_per_class: int = MEMORY_SAFE_TARGET,
    k_neighbors: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-call SMOTE-ENN application for training data.

    IMPORTANT: Apply ONLY to training split — never validation/test.

    Parameters
    ----------
    X_train : np.ndarray, shape (N_train, F) — should be NORMALIZED
    y_train : np.ndarray, shape (N_train,)
    target_per_class : int
    k_neighbors : int
    random_state : int
    verbose : bool

    Returns
    -------
    X_balanced : np.ndarray
    y_balanced : np.ndarray
    """
    handler = DOSSMOTEENNHandler(
        target_per_class=target_per_class,
        k_neighbors_smote=k_neighbors,
        random_state=random_state,
        verbose=verbose,
    )
    X_bal, y_bal = handler.fit_resample(X_train, y_train)

    if verbose:
        print(handler.get_balance_report())

    return X_bal, y_bal


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    rng = np.random.default_rng(42)

    # Simulate imbalanced DoS dataset
    class_sizes = {0: 5000, 1: 3000, 2: 500, 3: 200, 4: 150}
    X_parts, y_parts = [], []
    for cls, n in class_sizes.items():
        X_parts.append(rng.random((n, 64)).astype(np.float32))
        y_parts.append(np.full(n, cls, dtype=np.int64))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    print(f"Before: X={X.shape}, y={y.shape}")
    X_bal, y_bal = apply_smote_enn(X, y, target_per_class=3000, verbose=True)
    print(f"After:  X={X_bal.shape}, y={y_bal.shape}")