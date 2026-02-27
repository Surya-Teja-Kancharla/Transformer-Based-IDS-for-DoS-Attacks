"""
feature_encoder.py
==================
Handles feature selection and the critical 1D → 2D reshaping required
by the Res-TranBiLSTM paper.

Paper specification (Wang et al., 2023, Section 4.1):
  - CIC-IDS2017 starts with 84 features from CICFlowMeter
  - After cleaning and feature selection → 64 features retained
  - 64 features mapped to matrix form of 8 × 8
  - 8 × 8 expanded to 28 × 28 using BICUBIC interpolation
    (consistent with MNIST handwritten digit dataset size)
  - This 28×28 grayscale image feeds into the ResNet spatial branch

For our DoS-only study using Wednesday-workingHours.csv:
  - We select the 64 most informative features using variance + mutual info
  - Map to 8×8 → bicubic to 28×28 for spatial branch
  - Keep raw 64-dim vector → MLP encoding for temporal branch

Author: FYP Implementation
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    f_classif,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from paper)
# ---------------------------------------------------------------------------

TARGET_NUM_FEATURES: int = 64   # Paper Table 4: 64 features for CIC-IDS2017
SPATIAL_MATRIX_SMALL: int = 8   # 8 × 8 initial matrix
SPATIAL_MATRIX_LARGE: int = 28  # 28 × 28 after bicubic interpolation (paper)

# CIC-IDS2017 known-problematic features to exclude before selection
# (high cardinality identifiers, constant in Wednesday subset, known leakage)
EXCLUDE_FEATURES: List[str] = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "External IP",
    # Columns known to be all-zero or constant in CICIDS Wednesday file
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


# ---------------------------------------------------------------------------
# Picklable score function for SelectKBest (replaces unpicklable lambda)
# ---------------------------------------------------------------------------

class _MutualInfoScorer:
    """
    Module-level callable wrapping mutual_info_classif with a fixed
    random_state. Defined outside FeatureSelector so it is fully
    picklable by the standard pickle module (unlike a lambda or closure).
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def __call__(self, X, y):
        return mutual_info_classif(
            X, y,
            discrete_features=False,
            random_state=self.random_state,
        )


# ---------------------------------------------------------------------------
# Feature Selector
# ---------------------------------------------------------------------------

class FeatureSelector:
    """
    Selects the top-k most informative features from the cleaned DataFrame
    using a two-stage pipeline:
      1. VarianceThreshold: Remove near-zero variance features
      2. SelectKBest with mutual_info_classif or f_classif: Keep top 64

    Parameters
    ----------
    n_features : int
        Target number of features to select (default 64, per paper).
    variance_threshold : float
        Features with variance below this are removed first.
    score_func : str
        'mutual_info' or 'f_classif' for SelectKBest scoring.
    random_state : int
        Used for mutual_info reproducibility.
    verbose : bool
    """

    def __init__(
        self,
        n_features: int = TARGET_NUM_FEATURES,
        variance_threshold: float = 0.01,
        score_func: str = "mutual_info",
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.score_func = score_func
        self.random_state = random_state
        self.verbose = verbose

        self._variance_selector: Optional[VarianceThreshold] = None
        self._kbest_selector: Optional[SelectKBest] = None
        self._selected_feature_names: Optional[List[str]] = None
        self._all_feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> "FeatureSelector":
        """
        Fit both selectors on training data.

        Parameters
        ----------
        X : np.ndarray, shape (N, num_raw_features)
        y : np.ndarray, shape (N,)
        feature_names : List[str]
            Names corresponding to columns of X.
        """
        self._all_feature_names = list(feature_names)
        logger.info(f"FeatureSelector.fit() | Input features: {X.shape[1]}")

        # -- Stage 1: Variance Threshold --
        logger.info(
            f"  Stage 1: VarianceThreshold (threshold={self.variance_threshold})"
        )
        self._variance_selector = VarianceThreshold(
            threshold=self.variance_threshold
        )
        X_var = self._variance_selector.fit_transform(X)
        var_mask = self._variance_selector.get_support()
        var_names = [feature_names[i] for i, m in enumerate(var_mask) if m]
        logger.info(
            f"  After VarianceThreshold: {X_var.shape[1]} features "
            f"({X.shape[1] - X_var.shape[1]} removed)"
        )

        # -- Stage 2: SelectKBest --
        k = min(self.n_features, X_var.shape[1])
        logger.info(
            f"  Stage 2: SelectKBest (k={k}, score={self.score_func})"
        )

        if self.score_func == "mutual_info":
            sf = _MutualInfoScorer(random_state=self.random_state)
        else:
            sf = f_classif

        self._kbest_selector = SelectKBest(score_func=sf, k=k)

        with tqdm(
            total=1, desc="Fitting SelectKBest", disable=not self.verbose
        ) as pbar:
            self._kbest_selector.fit(X_var, y)
            pbar.update(1)

        kbest_mask = self._kbest_selector.get_support()
        self._selected_feature_names = [
            var_names[i] for i, m in enumerate(kbest_mask) if m
        ]

        self.is_fitted = True
        logger.info(
            f"  Selected {len(self._selected_feature_names)} features"
        )
        if self.verbose:
            logger.info(
                f"  Top features: {self._selected_feature_names[:10]}..."
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted selectors to X.

        Parameters
        ----------
        X : np.ndarray, shape (N, num_raw_features)

        Returns
        -------
        np.ndarray, shape (N, n_features)
        """
        self._check_fitted()
        X_var = self._variance_selector.transform(X)
        X_sel = self._kbest_selector.transform(X_var)
        return X_sel.astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_selected_feature_names(self) -> List[str]:
        self._check_fitted()
        return list(self._selected_feature_names)

    def save(self, path: Union[str, Path]) -> None:
        """Persist selector to disk (pickle)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"FeatureSelector saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureSelector":
        """Load persisted selector from disk."""
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)
        logger.info(f"FeatureSelector loaded from: {path}")
        return obj

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureSelector has not been fitted. Call fit() first."
            )


# ---------------------------------------------------------------------------
# Image Reshaper: 1D → 8×8 → 28×28 (bicubic)
# ---------------------------------------------------------------------------

class ImageReshaper:
    """
    Converts a 1D feature vector of length 64 into a 28×28 grayscale image
    for input to the ResNet spatial branch, following the paper exactly:

      Step 1: Arrange 64 features into 8×8 matrix
      Step 2: Bicubic interpolation to expand 8×8 → 28×28

    This matches the MNIST handwritten digit size used in the paper.

    For batched inputs, operates sample-by-sample with tqdm progress.

    Parameters
    ----------
    n_features : int
        Number of input features (must equal n_rows * n_cols).
    n_rows, n_cols : int
        Small matrix dimensions (default 8×8 = 64).
    target_size : int
        Final image size after interpolation (default 28).
    """

    def __init__(
        self,
        n_features: int = TARGET_NUM_FEATURES,
        n_rows: int = SPATIAL_MATRIX_SMALL,
        n_cols: int = SPATIAL_MATRIX_SMALL,
        target_size: int = SPATIAL_MATRIX_LARGE,
    ) -> None:
        if n_rows * n_cols != n_features:
            raise ValueError(
                f"n_rows ({n_rows}) × n_cols ({n_cols}) = "
                f"{n_rows * n_cols} ≠ n_features ({n_features})"
            )
        self.n_features = n_features
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.target_size = target_size

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """
        Transform a single 1D feature vector → 28×28 grayscale image.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)

        Returns
        -------
        np.ndarray, shape (1, target_size, target_size)
            Channel-first format for PyTorch Conv2d (1 grayscale channel).
        """
        if x.ndim != 1 or len(x) != self.n_features:
            raise ValueError(
                f"Expected 1D array of length {self.n_features}, "
                f"got shape {x.shape}"
            )

        # -- Step 1: Reshape to small matrix --
        mat = x.reshape(self.n_rows, self.n_cols).astype(np.float32)

        # -- Step 2: Bicubic interpolation to target_size --
        # Use scipy for high-quality bicubic
        from scipy.ndimage import zoom

        scale = self.target_size / self.n_rows
        mat_large = zoom(mat, zoom=scale, order=3)  # order=3 → bicubic

        # Clip to [0,1] range (features are already normalized)
        mat_large = np.clip(mat_large, 0.0, 1.0).astype(np.float32)

        # Add channel dimension: (1, H, W) for PyTorch
        return mat_large[np.newaxis, :, :]  # shape: (1, 28, 28)

    def transform_batch(
        self,
        X: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Transform a batch of 1D feature vectors → batch of 28×28 images.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)
        verbose : bool

        Returns
        -------
        np.ndarray, shape (N, 1, target_size, target_size)
            Suitable for nn.Conv2d input.
        """
        N = X.shape[0]
        out = np.zeros(
            (N, 1, self.target_size, self.target_size), dtype=np.float32
        )

        for i in tqdm(
            range(N),
            desc="Reshaping 1D→28×28",
            unit="sample",
            disable=not verbose,
            miniters=max(1, N // 100),
        ):
            out[i] = self.transform_single(X[i])

        return out

    def transform_batch_fast(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized bicubic interpolation for large batches (much faster).
        Uses cv2 for batch processing.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        np.ndarray, shape (N, 1, target_size, target_size)
        """
        try:
            import cv2
        except ImportError:
            logger.warning(
                "cv2 not available, falling back to scipy (slower)."
            )
            return self.transform_batch(X, verbose=True)

        N = X.shape[0]
        out = np.zeros(
            (N, 1, self.target_size, self.target_size), dtype=np.float32
        )

        # Reshape all at once
        mats = X.reshape(N, self.n_rows, self.n_cols).astype(np.float32)

        for i in range(N):
            resized = cv2.resize(
                mats[i],
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_CUBIC,
            )
            resized = np.clip(resized, 0.0, 1.0)
            out[i, 0] = resized

        return out

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Channel-first output shape (C, H, W)."""
        return (1, self.target_size, self.target_size)


# ---------------------------------------------------------------------------
# MLP Temporal Encoder Input Formatter
# ---------------------------------------------------------------------------

class TemporalInputFormatter:
    """
    Formats the 64-dim feature vector for input to the TranBiLSTM
    temporal branch.

    Paper (Section 3.2.1):
      "similar to the word embedding layer in NLP, an MLP layer is used
       to encode data for each feature. Thereafter, the features are
       amplified to map to different subspaces."

    This class treats each feature as a token in a sequence:
      - Input shape:  (N, 64)              — batch of flat feature vectors
      - Output shape: (N, 64, 1)           — (batch, seq_len, input_dim)
    where seq_len = num_features = 64, and each token has 1 raw value.
    The MLP encoding then expands each token from dim=1 to dim=16/32.

    Parameters
    ----------
    n_features : int
        Length of the 1D feature vector (64 for CIC-IDS2017).
    """

    def __init__(self, n_features: int = TARGET_NUM_FEATURES) -> None:
        self.n_features = n_features

    def format(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape flat features to sequence format.

        Parameters
        ----------
        X : np.ndarray, shape (N, n_features)

        Returns
        -------
        np.ndarray, shape (N, n_features, 1)
            (batch_size, seq_len, input_dim_per_token)
        """
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected (N, {self.n_features}), got {X.shape}"
            )
        return X[:, :, np.newaxis].astype(np.float32)

    @property
    def output_shape(self) -> Tuple[int, int]:
        """(seq_len, input_dim_per_token)"""
        return (self.n_features, 1)


# ---------------------------------------------------------------------------
# Utility: get 64 most important features for DoS classification
# ---------------------------------------------------------------------------

def get_dos_feature_importance_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 64,
) -> pd.DataFrame:
    """
    Compute mutual information scores for all features and return
    a sorted DataFrame. Used for documentation / reporting in paper.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    y : np.ndarray, shape (N,)
    feature_names : List[str]
    top_k : int

    Returns
    -------
    pd.DataFrame with columns ['feature', 'mi_score', 'rank']
    """
    from sklearn.feature_selection import mutual_info_classif

    logger.info("Computing mutual information for feature importance...")
    scores = mutual_info_classif(
        X, y, discrete_features=False, random_state=42
    )
    report = pd.DataFrame(
        {"feature": feature_names, "mi_score": scores}
    ).sort_values("mi_score", ascending=False).head(top_k)
    report["rank"] = range(1, len(report) + 1)
    report.reset_index(drop=True, inplace=True)
    return report


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate 64-feature input
    rng = np.random.default_rng(42)
    N = 100
    X_fake = rng.random((N, 64)).astype(np.float32)
    y_fake = rng.integers(0, 5, size=N)

    reshaper = ImageReshaper()
    images = reshaper.transform_batch(X_fake, verbose=True)
    print(f"Image batch shape: {images.shape}")  # (100, 1, 28, 28)

    formatter = TemporalInputFormatter()
    seq = formatter.format(X_fake)
    print(f"Temporal seq shape: {seq.shape}")  # (100, 64, 1)
