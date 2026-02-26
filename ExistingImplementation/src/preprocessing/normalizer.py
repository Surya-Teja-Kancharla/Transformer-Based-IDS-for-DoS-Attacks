"""
normalizer.py
=============
Implements the Min-Max normalization described in the paper (Eq. 2):

    x_norm = (x - x_min) / (x_max - x_min)

Applied PER FEATURE column across the training set. The fitted
normalizer is then applied identically to validation/test sets to
prevent data leakage.

Paper reference (Wang et al., 2023, Section 3.2.3):
  "In order to eliminate data differences caused by different dimensions,
   normalization processing is performed after data digitization and
   encoding. The min-max normalization method is adopted."

Additional details for DoS-only implementation:
  - Normalization applied AFTER feature selection (64 features)
  - Fit ONLY on training split (never on validation/test)
  - NaN-safe: clips output to [0, 1] even if test has values outside
    training range (common in network traffic)
  - Saves min/max arrays for reproducibility and IoT deployment

Author: FYP Implementation
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MinMaxNormalizer
# ---------------------------------------------------------------------------

class MinMaxNormalizer:
    """
    Per-feature Min-Max normalization: scales each feature to [0, 1].

    Formula (paper Eq. 2):
        x_norm = (x - x_min) / (x_max - x_min)

    Handles edge case where x_max == x_min (constant feature) by
    setting output to 0.0 for those features (zero-variance columns
    should have been dropped by FeatureSelector, but this is a safety net).

    Parameters
    ----------
    clip : bool
        If True, clip output values to [0, 1]. Handles test-set values
        outside training-set range. Recommended for network traffic data.
    eps : float
        Small constant added to denominator to prevent division by zero.
    """

    def __init__(
        self,
        clip: bool = True,
        eps: float = 1e-10,
    ) -> None:
        self.clip = clip
        self.eps = eps

        self._x_min: Optional[np.ndarray] = None
        self._x_max: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None   # 1 / (x_max - x_min + eps)
        self._feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MinMaxNormalizer":
        """
        Compute per-feature min and max from training data.

        Parameters
        ----------
        X : np.ndarray, shape (N, F), dtype float32
            Training feature matrix.
        feature_names : List[str], optional
            Column names for logging and diagnostics.

        Returns
        -------
        self
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        N, F = X.shape
        logger.info(
            f"MinMaxNormalizer.fit() | shape=({N:,}, {F}) | "
            f"fitting per-feature min/max..."
        )

        self._x_min = X.min(axis=0).astype(np.float64)
        self._x_max = X.max(axis=0).astype(np.float64)
        self._scale = 1.0 / (self._x_max - self._x_min + self.eps)
        self._feature_names = (
            feature_names if feature_names else [f"f{i}" for i in range(F)]
        )
        self.is_fitted = True

        # Report any near-constant features (potential issue)
        near_const = np.where((self._x_max - self._x_min) < 1e-6)[0]
        if len(near_const) > 0:
            names = [self._feature_names[i] for i in near_const]
            logger.warning(
                f"  {len(near_const)} near-constant features detected "
                f"(range < 1e-6): {names[:5]}..."
            )

        logger.info(
            f"  Min range: [{self._x_min.min():.6f}, {self._x_min.max():.6f}]"
        )
        logger.info(
            f"  Max range: [{self._x_max.min():.6f}, {self._x_max.max():.6f}]"
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Min-Max normalization using fitted parameters.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)

        Returns
        -------
        np.ndarray, shape (N, F), dtype float32, values in [0, 1]
        """
        self._check_fitted()
        if X.shape[1] != len(self._x_min):
            raise ValueError(
                f"Feature dimension mismatch: "
                f"expected {len(self._x_min)}, got {X.shape[1]}"
            )

        X = X.astype(np.float64)
        X_norm = (X - self._x_min) * self._scale

        if self.clip:
            X_norm = np.clip(X_norm, 0.0, 1.0)

        return X_norm.astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Fit and transform in one call (training set only)."""
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Reverse normalization: x = x_norm * (x_max - x_min) + x_min.

        Useful for interpretability or visualization.
        """
        self._check_fitted()
        X_norm = X_norm.astype(np.float64)
        X_orig = X_norm / self._scale + self._x_min
        return X_orig.astype(np.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_normalization_stats(self) -> pd.DataFrame:
        """
        Return a DataFrame summarizing normalization parameters per feature.

        Columns: feature_name, x_min, x_max, range, scale
        """
        self._check_fitted()
        return pd.DataFrame({
            "feature": self._feature_names,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "range": self._x_max - self._x_min,
            "scale": self._scale,
        })

    def check_out_of_range(
        self, X: np.ndarray, threshold: float = 0.05
    ) -> Dict[str, int]:
        """
        Report how many test samples have values outside training range.
        Useful for domain-shift analysis in IDS.

        Parameters
        ----------
        X : np.ndarray — raw (unnormalized) test data
        threshold : float — warn if > threshold fraction out-of-range

        Returns
        -------
        Dict mapping feature names to count of out-of-range samples
        """
        self._check_fitted()
        below = (X < self._x_min).sum(axis=0)
        above = (X > self._x_max).sum(axis=0)
        total_oor = below + above
        N = X.shape[0]

        report = {}
        for i, name in enumerate(self._feature_names):
            if total_oor[i] > 0:
                pct = total_oor[i] / N * 100
                report[name] = int(total_oor[i])
                if pct > threshold * 100:
                    logger.warning(
                        f"  Feature '{name}': {total_oor[i]:,} samples "
                        f"({pct:.1f}%) out of training range"
                    )
        if not report:
            logger.info("  All test features within training range ✓")
        return report

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save normalizer state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "x_min": self._x_min,
            "x_max": self._x_max,
            "scale": self._scale,
            "feature_names": self._feature_names,
            "clip": self.clip,
            "eps": self.eps,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"MinMaxNormalizer saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MinMaxNormalizer":
        """Load normalizer from disk."""
        with open(Path(path), "rb") as f:
            state = pickle.load(f)
        obj = cls(clip=state["clip"], eps=state["eps"])
        obj._x_min = state["x_min"]
        obj._x_max = state["x_max"]
        obj._scale = state["scale"]
        obj._feature_names = state["feature_names"]
        obj.is_fitted = state["is_fitted"]
        logger.info(f"MinMaxNormalizer loaded from: {path}")
        return obj

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "MinMaxNormalizer not fitted. Call fit() first."
            )


# ---------------------------------------------------------------------------
# Utility: normalize DataFrame in-place (preserves Label column)
# ---------------------------------------------------------------------------

def normalize_dataframe(
    df_train: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
    label_col: str = "Label",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, ...]:
    """
    Fit MinMaxNormalizer on df_train features, apply to all splits.

    Parameters
    ----------
    df_train, df_val, df_test : pd.DataFrame
        Splits with the same feature columns + 'Label'.
    label_col : str
        Column to exclude from normalization.
    save_path : optional path to save the fitted normalizer.

    Returns
    -------
    Tuple of normalized DataFrames (only non-None ones returned).
    """
    feature_cols = [c for c in df_train.columns if c != label_col]

    normalizer = MinMaxNormalizer(clip=True)
    X_train = df_train[feature_cols].values
    normalizer.fit(X_train, feature_names=feature_cols)

    results = []
    for df in [df_train, df_val, df_test]:
        if df is not None:
            df_copy = df.copy()
            X = df_copy[feature_cols].values
            df_copy[feature_cols] = normalizer.transform(X)
            results.append(df_copy)

    if save_path:
        normalizer.save(save_path)

    return tuple(results)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    X_train = rng.random((1000, 64)).astype(np.float32) * 100
    X_test = rng.random((200, 64)).astype(np.float32) * 120  # some OOR values

    norm = MinMaxNormalizer(clip=True)
    X_train_norm = norm.fit_transform(X_train)
    X_test_norm = norm.transform(X_test)

    print(f"Train norm range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    print(f"Test norm range:  [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}]")
    norm.check_out_of_range(X_test)

    stats = norm.get_normalization_stats()
    print(stats.head())