"""
dataset_loader.py
=================
Handles loading and initial cleaning of the CIC-IDS2017
Wednesday-workingHours.csv file, filtering ONLY DoS-related traffic
classes plus BENIGN for binary/multi-class DoS detection.

Wednesday file contains:
  - BENIGN
  - DoS slowloris
  - DoS Slowhttptest
  - DoS Hulk
  - DoS GoldenEye
  - Heartbleed  (very rare, included as a DoS-adjacent attack)

Paper reference (Wang et al., 2023):
  - CIC-IDS2017 originally has 84 features extracted by CICFlowMeter
  - After cleaning and feature selection → 64 features retained
  - Labels are merged and filtered per Table 4 of the paper
  - For our DoS-focused study: Normal vs {DoS slowloris, DoS Slowhttptest,
    DoS Hulk, DoS GoldenEye} → 5-class multi-classification task

Author: FYP Implementation
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants specific to Wednesday-workingHours.csv
# ---------------------------------------------------------------------------

# Exact label strings as they appear in the CSV (column: ' Label')
RAW_LABEL_COL = " Label"          # Note: CIC-IDS2017 has a leading space in column name

# DoS label groups as defined in the paper (Table 4) adapted for Wednesday file
DOS_LABEL_MAP: Dict[str, int] = {
    "BENIGN":            0,   # Normal traffic
    "DoS slowloris":     1,   # Slowloris HTTP keep-alive DoS
    "DoS Slowhttptest":  2,   # Slow HTTP test
    "DoS Hulk":          3,   # Hulk HTTP flood
    "DoS GoldenEye":     4,   # GoldenEye HTTP attack
}

# Heartbleed is extremely rare (11 samples) — optionally include
HEARTBLEED_LABEL = "Heartbleed"
HEARTBLEED_CLASS_ID = 5

# Human-readable class names for logging / plotting
CLASS_NAMES: Dict[int, str] = {
    0: "BENIGN",
    1: "DoS_slowloris",
    2: "DoS_Slowhttptest",
    3: "DoS_Hulk",
    4: "DoS_GoldenEye",
    5: "Heartbleed",
}

# Features to ALWAYS drop (metadata, identifiers, constant-zero cols)
# These are standard CIC-IDS2017 problematic columns identified in literature
DROP_COLUMNS: List[str] = [
    "Flow ID",
    " Source IP",
    " Source Port",
    " Destination IP",
    " Destination Port",
    " Protocol",
    " Timestamp",
    "External IP",
]

# Features with known NaN / Infinity issues in CIC-IDS2017
# Will be handled in cleaning step
INFINITY_PRONE_COLS: List[str] = [
    " Flow Bytes/s",
    " Flow Packets/s",
]

# ---------------------------------------------------------------------------
# Core DatasetLoader class
# ---------------------------------------------------------------------------

class CICIDSDatasetLoader:
    """
    Loads and performs initial cleaning on the CIC-IDS2017
    Wednesday-workingHours.csv file, retaining only DoS-related records.

    Pipeline:
        1. load_raw()         → raw DataFrame with all columns
        2. clean()            → remove NaN/Inf, drop metadata cols
        3. filter_dos_only()  → keep only BENIGN + DoS classes
        4. encode_labels()    → map string labels → integer class IDs
        5. get_class_stats()  → print/log class distribution

    Parameters
    ----------
    data_path : str | Path
        Full path to Wednesday-workingHours.csv
    include_heartbleed : bool
        If True, Heartbleed samples (very rare) are retained as class 5.
        Default False — paper merges it into DoS group or drops it.
    max_benign_samples : int | None
        If set, randomly subsample BENIGN class to this count to reduce
        class imbalance before SMOTE-ENN. Set None to keep all.
    random_state : int
        Seed for reproducibility of any subsampling.
    verbose : bool
        If True, emit detailed progress logs.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        include_heartbleed: bool = False,
        max_benign_samples: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.include_heartbleed = include_heartbleed
        self.max_benign_samples = max_benign_samples
        self.random_state = random_state
        self.verbose = verbose

        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._dos_df: Optional[pd.DataFrame] = None

        # Statistics tracking
        self.stats: Dict[str, object] = {}

        self._validate_path()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """
        Load the raw CSV. Returns a DataFrame with all original columns.

        The CIC-IDS2017 Wednesday file has:
          - ~691,000 rows
          - 79 feature columns + 1 label column = 80 columns
          - Many column names have leading spaces (CICFlowMeter artifact)
        """
        logger.info(f"Loading raw CSV from: {self.data_path}")

        # Read with low_memory=False to avoid mixed-type dtype warnings
        # typical file size ~150MB
        with tqdm(total=1, desc="Reading CSV", unit="file",
                  disable=not self.verbose) as pbar:
            df = pd.read_csv(
                self.data_path,
                low_memory=False,
                encoding="utf-8",
            )
            pbar.update(1)

        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # After strip, label column becomes 'Label'
        # Standardize label column name
        if "Label" in df.columns:
            self._label_col = "Label"
        elif " Label" in df.columns:
            df.rename(columns={" Label": "Label"}, inplace=True)
            self._label_col = "Label"
        else:
            raise ValueError(
                f"Label column not found. Available columns: {df.columns.tolist()}"
            )

        self._raw_df = df
        self.stats["raw_shape"] = df.shape
        self.stats["raw_label_counts"] = df["Label"].value_counts().to_dict()

        logger.info(
            f"Raw data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols"
        )
        self._log_label_distribution(df, "Raw")
        return df

    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the DataFrame:
          1. Drop metadata/identifier columns (Flow ID, IPs, ports, timestamp)
          2. Strip remaining whitespace from column names
          3. Replace ±Inf values with NaN
          4. Drop rows with any NaN
          5. Remove duplicate rows
          6. Convert all feature columns to float32

        Parameters
        ----------
        df : DataFrame, optional
            If None, uses self._raw_df (must have called load_raw() first).

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        if df is None:
            if self._raw_df is None:
                raise RuntimeError("Call load_raw() before clean()")
            df = self._raw_df.copy()

        logger.info("Starting data cleaning...")
        initial_rows = len(df)

        # -- Step 1: Drop metadata columns (silently ignore missing ones) --
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        # Also strip-based match
        cols_to_drop += [
            c for c in df.columns
            if c.strip() in [d.strip() for d in DROP_COLUMNS]
            and c not in cols_to_drop
        ]
        cols_to_drop = list(set(cols_to_drop))
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        logger.info(f"  Dropped {len(cols_to_drop)} metadata columns")

        # -- Step 2: Strip remaining column name whitespace --
        df.columns = df.columns.str.strip()

        # -- Step 3: Identify feature columns (all except Label) --
        label_col = "Label"
        feature_cols = [c for c in df.columns if c != label_col]

        # -- Step 4: Replace Inf/-Inf with NaN across all feature cols --
        before_inf = len(df)
        df[feature_cols] = df[feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        inf_rows = df[feature_cols].isnull().any(axis=1).sum()
        logger.info(f"  Rows with Inf/NaN values: {inf_rows:,}")

        # -- Step 5: Drop rows with any NaN --
        df.dropna(inplace=True)
        after_nan = len(df)
        logger.info(f"  Dropped {before_inf - after_nan:,} rows (NaN/Inf)")

        # -- Step 6: Remove exact duplicate rows --
        before_dup = len(df)
        df.drop_duplicates(inplace=True)
        after_dup = len(df)
        logger.info(f"  Dropped {before_dup - after_dup:,} duplicate rows")

        # -- Step 7: Convert feature columns to float32 (memory efficient) --
        with tqdm(total=len(feature_cols),
                  desc="Converting dtypes",
                  unit="col",
                  disable=not self.verbose) as pbar:
            for col in feature_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                        np.float32
                    )
                except Exception as e:
                    logger.warning(f"  Could not convert column '{col}': {e}")
                pbar.update(1)

        # Drop any newly-created NaN from coerce
        df.dropna(inplace=True)

        # -- Step 8: Remove zero-variance columns (all-zero or all-constant) --
        feature_cols = [c for c in df.columns if c != label_col]
        zero_var_cols = [
            c for c in feature_cols if df[c].std() == 0.0
        ]
        if zero_var_cols:
            df.drop(columns=zero_var_cols, inplace=True)
            logger.info(
                f"  Dropped {len(zero_var_cols)} zero-variance columns: "
                f"{zero_var_cols}"
            )

        df.reset_index(drop=True, inplace=True)
        self._clean_df = df

        self.stats["clean_shape"] = df.shape
        self.stats["rows_removed_cleaning"] = initial_rows - len(df)

        logger.info(
            f"Cleaning complete: {len(df):,} rows retained "
            f"({initial_rows - len(df):,} removed)"
        )
        logger.info(f"  Feature columns remaining: {len(df.columns) - 1}")

        return df

    def filter_dos_only(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Filter the DataFrame to retain only:
          - BENIGN (class 0)
          - DoS slowloris (class 1)
          - DoS Slowhttptest (class 2)
          - DoS Hulk (class 3)
          - DoS GoldenEye (class 4)
          - Heartbleed (class 5) — only if include_heartbleed=True

        Optionally subsample BENIGN to max_benign_samples to reduce
        extreme class imbalance (BENIGN ~2.26M vs DoS ~251K in full dataset;
        in Wednesday file BENIGN ~439K vs all DoS ~254K).

        Parameters
        ----------
        df : DataFrame, optional
            If None, uses self._clean_df.

        Returns
        -------
        pd.DataFrame
            DoS-filtered DataFrame with integer 'Label' column.
        """
        if df is None:
            if self._clean_df is None:
                raise RuntimeError("Call clean() before filter_dos_only()")
            df = self._clean_df.copy()

        logger.info("Filtering for DoS classes only...")

        # Build target label set
        target_labels = list(DOS_LABEL_MAP.keys())
        if self.include_heartbleed:
            target_labels.append(HEARTBLEED_LABEL)

        # Filter rows
        mask = df["Label"].isin(target_labels)
        df = df[mask].copy()
        logger.info(
            f"  After DoS filter: {len(df):,} rows "
            f"({mask.sum() / len(mask) * 100:.1f}% of cleaned data)"
        )

        # Subsample BENIGN if requested
        if self.max_benign_samples is not None:
            benign_mask = df["Label"] == "BENIGN"
            benign_count = benign_mask.sum()
            if benign_count > self.max_benign_samples:
                benign_idx = (
                    df[benign_mask]
                    .sample(
                        n=self.max_benign_samples,
                        random_state=self.random_state,
                    )
                    .index
                )
                non_benign_idx = df[~benign_mask].index
                df = df.loc[benign_idx.union(non_benign_idx)].copy()
                logger.info(
                    f"  BENIGN subsampled: {benign_count:,} → "
                    f"{self.max_benign_samples:,}"
                )

        df.reset_index(drop=True, inplace=True)
        self._dos_df = df
        self.stats["dos_filter_shape"] = df.shape

        self._log_label_distribution(df, "DoS-Filtered")
        return df

    def encode_labels(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Map string labels to integer class IDs defined in DOS_LABEL_MAP.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with 'Label' column replaced by integer class IDs.
        class_map : Dict[int, str]
            Mapping from integer ID → human-readable class name.
        """
        if df is None:
            if self._dos_df is None:
                raise RuntimeError(
                    "Call filter_dos_only() before encode_labels()"
                )
            df = self._dos_df.copy()

        logger.info("Encoding string labels → integer class IDs...")

        label_to_int = dict(DOS_LABEL_MAP)
        if self.include_heartbleed:
            label_to_int[HEARTBLEED_LABEL] = HEARTBLEED_CLASS_ID

        # Validate all labels are known
        unknown = set(df["Label"].unique()) - set(label_to_int.keys())
        if unknown:
            raise ValueError(
                f"Unknown labels found after filtering: {unknown}. "
                "Check filter_dos_only() step."
            )

        df["Label"] = df["Label"].map(label_to_int).astype(np.int64)

        # Build reverse map for reference
        int_to_name = {v: k for k, v in label_to_int.items()}

        self.stats["final_class_distribution"] = (
            df["Label"].value_counts().sort_index().to_dict()
        )
        self.stats["num_classes"] = df["Label"].nunique()
        self.stats["num_features"] = df.shape[1] - 1  # exclude Label

        logger.info(f"  Classes: {int_to_name}")
        logger.info(
            f"  Distribution: {self.stats['final_class_distribution']}"
        )
        return df, int_to_name

    def run_full_pipeline(
        self,
    ) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Convenience method: runs load_raw → clean → filter_dos_only →
        encode_labels in sequence.

        Returns
        -------
        df : pd.DataFrame
            Fully processed DataFrame ready for normalization/SMOTE-ENN.
        class_map : Dict[int, str]
            Integer → class name mapping.
        """
        logger.info("=" * 60)
        logger.info("CIC-IDS2017 DoS Dataset Loading Pipeline")
        logger.info("=" * 60)
        self.load_raw()
        self.clean()
        self.filter_dos_only()
        df, class_map = self.encode_labels()
        self._print_summary()
        return df, class_map

    def get_feature_names(self) -> List[str]:
        """Return feature column names (all columns except 'Label')."""
        if self._dos_df is None:
            raise RuntimeError("Run pipeline first.")
        return [c for c in self._dos_df.columns if c != "Label"]

    def get_num_classes(self) -> int:
        """Return number of unique classes after filtering."""
        n = self.stats.get("num_classes", None)
        if n is None:
            raise RuntimeError("Run pipeline first.")
        return int(n)

    def save_intermediate(
        self,
        df: pd.DataFrame,
        save_path: Union[str, Path],
        stage: str = "processed",
    ) -> None:
        """
        Save a pipeline-stage DataFrame to CSV for debugging/reuse.

        Parameters
        ----------
        df : pd.DataFrame
        save_path : str | Path
            Directory path where CSV will be saved.
        stage : str
            Stage name for filename (e.g., 'processed', 'augmented').
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fname = save_path / f"cicids2017_wednesday_{stage}.csv"
        logger.info(f"Saving {stage} data to: {fname}")
        df.to_csv(fname, index=False)
        logger.info(f"  Saved {len(df):,} rows × {df.shape[1]} cols")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_path(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_path}\n"
                "Please place Wednesday-workingHours.csv in "
                "ExistingImplementation/data/raw/"
            )
        if not self.data_path.suffix.lower() == ".csv":
            raise ValueError(
                f"Expected a .csv file, got: {self.data_path.suffix}"
            )

    def _log_label_distribution(
        self, df: pd.DataFrame, stage: str
    ) -> None:
        logger.info(f"  [{stage}] Label distribution:")
        counts = df["Label"].value_counts()
        total = len(df)
        for label, count in counts.items():
            pct = count / total * 100
            logger.info(f"    {str(label):<30} {count:>10,}  ({pct:5.2f}%)")

    def _print_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"  Raw shape          : {self.stats.get('raw_shape')}")
        logger.info(f"  Clean shape        : {self.stats.get('clean_shape')}")
        logger.info(f"  DoS-filtered shape : {self.stats.get('dos_filter_shape')}")
        logger.info(f"  Num classes        : {self.stats.get('num_classes')}")
        logger.info(f"  Num features       : {self.stats.get('num_features')}")
        dist = self.stats.get("final_class_distribution", {})
        logger.info("  Final class distribution:")
        for cls_id, count in sorted(dist.items()):
            name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
            logger.info(f"    [{cls_id}] {name:<25} {count:>10,}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Standalone helper: load from processed CSV (skip re-cleaning)
# ---------------------------------------------------------------------------

def load_processed_csv(
    path: Union[str, Path],
    label_col: str = "Label",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load an already-processed CSV (output of save_intermediate) directly
    into numpy arrays. Useful when pipeline has already been run once.

    Returns
    -------
    X : np.ndarray, shape (N, num_features), dtype float32
    y : np.ndarray, shape (N,), dtype int64
    feature_names : List[str]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")

    logger.info(f"Loading processed CSV: {path}")
    df = pd.read_csv(path, low_memory=False)

    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    logger.info(f"  Loaded X: {X.shape}, y: {y.shape}")
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Quick sanity-check (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print(
            "Usage: python dataset_loader.py "
            "<path/to/Wednesday-workingHours.csv>"
        )
        sys.exit(1)

    loader = CICIDSDatasetLoader(
        data_path=sys.argv[1],
        include_heartbleed=False,
        max_benign_samples=None,
        verbose=True,
    )
    df, class_map = loader.run_full_pipeline()
    print(f"\nFinal DataFrame shape: {df.shape}")
    print(f"Class map: {class_map}")
    print(df.head(3))