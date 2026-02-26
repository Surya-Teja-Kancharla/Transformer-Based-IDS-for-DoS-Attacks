"""
data_pipeline.py
================
Master preprocessing pipeline for the Res-TranBiLSTM DoS detection study.
Orchestrates all preprocessing steps in the correct order per the paper:

  Paper (Wang et al., 2023) Pipeline:
    1. Load Wednesday-workingHours.csv
    2. Clean (drop NaN/Inf, remove duplicates, drop metadata cols)
    3. Filter DoS classes only (BENIGN + 4 DoS attack types)
    4. Encode string labels → integer IDs
    5. Feature selection (64 features per paper)
    6. Train/Val/Test split (80/10/10 or 80/20 per paper)
    7. Min-Max Normalization (fit on train only, apply to all)
    8. SMOTE-ENN on training set only
    9. Prepare dual-branch inputs:
       - Spatial branch: 1D (64,) → 2D 8×8 → bicubic 28×28 images
       - Temporal branch: 1D (64,) → sequence (64, 1) for MLP encoding
   10. Save processed arrays as .npy files for fast reloading

Usage:
    # Full pipeline run (first time):
    python data_pipeline.py --csv path/to/Wednesday-workingHours.csv

    # Subsequent runs (load from cache):
    python data_pipeline.py --load-cache

Author: FYP Implementation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocessing.dataset_loader import CICIDSDatasetLoader, CLASS_NAMES, DOS_LABEL_MAP
from preprocessing.feature_encoder import FeatureSelector, ImageReshaper, TemporalInputFormatter
from preprocessing.normalizer import MinMaxNormalizer
from preprocessing.smote_enn import DOSSMOTEENNHandler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to ExistingImplementation/)
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR     = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_AUGMENTED_DIR = Path("data/augmented")
CACHE_MANIFEST_FILE   = "pipeline_manifest.json"

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

PIPELINE_CONFIG = {
    # Data loading
    "csv_filename": "Wednesday-workingHours.csv",
    "include_heartbleed": False,

    # Feature selection
    "n_features": 64,             # Paper: 64 features for CIC-IDS2017
    "variance_threshold": 0.01,
    "score_func": "mutual_info",

    # Spatial image dimensions (paper)
    "spatial_small": 8,           # 8×8 initial matrix
    "spatial_large": 28,          # 28×28 after bicubic upsampling

    # Train/Val/Test split
    "train_ratio": 0.80,          # Paper uses 80/20 train/test split
    "val_ratio": 0.10,            # We further split 10% of train for val
    "test_ratio": 0.10,
    "stratify": True,

    # SMOTE-ENN
    "smote_target_per_class": 20_000,  # Memory-safe for 4GB GPU
    "smote_k_neighbors": 5,

    # Reproducibility
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Processed data container
# ---------------------------------------------------------------------------

class ProcessedDataset:
    """
    Container for all preprocessed arrays ready for model training.

    Attributes
    ----------
    Spatial branch (for ResNet):
      X_train_img : (N_train, 1, 28, 28)  float32  — after SMOTE-ENN
      X_val_img   : (N_val,   1, 28, 28)  float32
      X_test_img  : (N_test,  1, 28, 28)  float32

    Temporal branch (for TranBiLSTM):
      X_train_seq : (N_train, 64, 1)  float32 — after SMOTE-ENN
      X_val_seq   : (N_val,   64, 1)  float32
      X_test_seq  : (N_test,  64, 1)  float32

    Labels:
      y_train : (N_train,)  int64
      y_val   : (N_val,)    int64
      y_test  : (N_test,)   int64

    Metadata:
      class_names : {0: 'BENIGN', 1: 'DoS_slowloris', ...}
      num_classes : int
      feature_names : List[str]
      config : dict
    """

    def __init__(self) -> None:
        # Spatial
        self.X_train_img: Optional[np.ndarray] = None
        self.X_val_img: Optional[np.ndarray] = None
        self.X_test_img: Optional[np.ndarray] = None

        # Temporal
        self.X_train_seq: Optional[np.ndarray] = None
        self.X_val_seq: Optional[np.ndarray] = None
        self.X_test_seq: Optional[np.ndarray] = None

        # Raw normalized (for diagnostics)
        self.X_train_flat: Optional[np.ndarray] = None
        self.X_val_flat: Optional[np.ndarray] = None
        self.X_test_flat: Optional[np.ndarray] = None

        # Labels
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Metadata
        self.class_names: Dict[int, str] = {}
        self.num_classes: int = 0
        self.num_features: int = 0
        self.feature_names: List[str] = []
        self.config: Dict = {}

    def summary(self) -> str:
        lines = ["ProcessedDataset Summary", "=" * 50]
        for split, y in [("Train", self.y_train),
                          ("Val",   self.y_val),
                          ("Test",  self.y_test)]:
            if y is not None:
                lines.append(f"  {split}: {len(y):>10,} samples")
                for cls_id, name in sorted(self.class_names.items()):
                    cnt = (y == cls_id).sum()
                    pct = cnt / len(y) * 100
                    lines.append(f"    [{cls_id}] {name:<25} {cnt:>8,}  ({pct:5.1f}%)")
        lines.append(f"\n  Num classes  : {self.num_classes}")
        lines.append(f"  Num features : {self.num_features}")
        if self.X_train_img is not None:
            lines.append(f"  Spatial shape: {self.X_train_img.shape}")
        if self.X_train_seq is not None:
            lines.append(f"  Temporal shape: {self.X_train_seq.shape}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

class DOSPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for Res-TranBiLSTM DoS detection.

    Parameters
    ----------
    base_dir : str | Path
        Root directory of ExistingImplementation (contains data/, src/).
    config : dict, optional
        Override any PIPELINE_CONFIG values.
    verbose : bool
    """

    def __init__(
        self,
        base_dir: Union[str, Path] = ".",
        config: Optional[Dict] = None,
        verbose: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.config = {**PIPELINE_CONFIG, **(config or {})}
        self.verbose = verbose

        self.raw_dir = self.base_dir / DEFAULT_RAW_DIR
        self.processed_dir = self.base_dir / DEFAULT_PROCESSED_DIR
        self.augmented_dir = self.base_dir / DEFAULT_AUGMENTED_DIR

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_dir.mkdir(parents=True, exist_ok=True)

        # Artifacts saved during pipeline
        self._normalizer: Optional[MinMaxNormalizer] = None
        self._selector: Optional[FeatureSelector] = None
        self._reshaper = ImageReshaper(
            n_features=self.config["n_features"],
            n_rows=self.config["spatial_small"],
            n_cols=self.config["spatial_small"],
            target_size=self.config["spatial_large"],
        )
        self._formatter = TemporalInputFormatter(
            n_features=self.config["n_features"]
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        force_rerun: bool = False,
    ) -> ProcessedDataset:
        """
        Run the full preprocessing pipeline.

        Steps:
          1. Load + clean + filter DoS classes
          2. Encode labels
          3. Split train/val/test (stratified)
          4. Feature selection (fit on train)
          5. Normalization (fit on train)
          6. SMOTE-ENN (on train only)
          7. Reshape to image + sequence inputs
          8. Save to disk

        Parameters
        ----------
        csv_path : path to Wednesday-workingHours.csv (optional if cached)
        force_rerun : bool — if True, ignore cache and re-run from scratch

        Returns
        -------
        ProcessedDataset
        """
        t_pipeline_start = time.time()
        logger.info("\n" + "=" * 65)
        logger.info("  Res-TranBiLSTM DoS Preprocessing Pipeline")
        logger.info("=" * 65)

        # Check cache
        if not force_rerun and self._cache_exists():
            logger.info("Cache found — loading from disk...")
            return self._load_cache()

        # -- Step 1-4: Load, clean, filter, encode --
        if csv_path is None:
            csv_path = self.raw_dir / self.config["csv_filename"]

        loader = CICIDSDatasetLoader(
            data_path=csv_path,
            include_heartbleed=self.config["include_heartbleed"],
            verbose=self.verbose,
        )
        df, class_map = loader.run_full_pipeline()

        # Save intermediate processed CSV
        loader.save_intermediate(df, self.processed_dir, stage="filtered")

        # Extract features and labels
        feature_cols = [c for c in df.columns if c != "Label"]
        X_all = df[feature_cols].values.astype(np.float32)
        y_all = df["Label"].values.astype(np.int64)
        num_classes = len(np.unique(y_all))

        logger.info(f"\nStep 2: Stratified Train/Val/Test Split")
        logger.info(f"  Ratios: {self.config['train_ratio']}/{self.config['val_ratio']}/{self.config['test_ratio']}")

        # -- Step 2: Train/Val/Test split --
        # First split: (train+val) vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X_all, y_all,
            test_size=self.config["test_ratio"],
            stratify=y_all if self.config["stratify"] else None,
            random_state=self.config["random_state"],
        )

        # Second split: train vs val
        val_ratio_adjusted = self.config["val_ratio"] / (
            self.config["train_ratio"] + self.config["val_ratio"]
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_adjusted,
            stratify=y_trainval if self.config["stratify"] else None,
            random_state=self.config["random_state"],
        )

        logger.info(
            f"  Train: {len(y_train):,} | Val: {len(y_val):,} | "
            f"Test: {len(y_test):,}"
        )

        # -- Step 3: Feature Selection (fit on train only) --
        logger.info(f"\nStep 3: Feature Selection (k={self.config['n_features']})")
        self._selector = FeatureSelector(
            n_features=self.config["n_features"],
            variance_threshold=self.config["variance_threshold"],
            score_func=self.config["score_func"],
            random_state=self.config["random_state"],
            verbose=self.verbose,
        )
        X_train_sel = self._selector.fit_transform(X_train, y_train, feature_cols)
        X_val_sel   = self._selector.transform(X_val)
        X_test_sel  = self._selector.transform(X_test)

        selected_feature_names = self._selector.get_selected_feature_names()
        logger.info(f"  Selected {len(selected_feature_names)} features")

        # Save selector
        self._selector.save(self.processed_dir / "feature_selector.pkl")

        # -- Step 4: Min-Max Normalization (fit on train only) --
        logger.info("\nStep 4: Min-Max Normalization (paper Eq. 2)")
        self._normalizer = MinMaxNormalizer(clip=True)
        X_train_norm = self._normalizer.fit_transform(
            X_train_sel, feature_names=selected_feature_names
        )
        X_val_norm  = self._normalizer.transform(X_val_sel)
        X_test_norm = self._normalizer.transform(X_test_sel)

        # Save normalizer
        self._normalizer.save(self.processed_dir / "normalizer.pkl")

        # Save normalized splits as CSV for reference
        self._save_split_csv(X_train_norm, y_train, selected_feature_names,
                             self.processed_dir / "train_normalized.csv")
        self._save_split_csv(X_val_norm, y_val, selected_feature_names,
                             self.processed_dir / "val_normalized.csv")
        self._save_split_csv(X_test_norm, y_test, selected_feature_names,
                             self.processed_dir / "test_normalized.csv")

        # -- Step 5: SMOTE-ENN on training set only --
        logger.info("\nStep 5: SMOTE-ENN Class Balancing (train only)")
        logger.info("  [Paper Section 3.2.2: SMOTE + ENN cleaning]")

        smote_handler = DOSSMOTEENNHandler(
            target_per_class=self.config["smote_target_per_class"],
            k_neighbors_smote=self.config["smote_k_neighbors"],
            random_state=self.config["random_state"],
            verbose=self.verbose,
        )
        X_train_bal, y_train_bal = smote_handler.fit_resample(
            X_train_norm, y_train
        )
        logger.info(f"\n{smote_handler.get_balance_report()}")

        # Save augmented arrays
        np.save(self.augmented_dir / "X_train_balanced.npy", X_train_bal)
        np.save(self.augmented_dir / "y_train_balanced.npy", y_train_bal)

        # -- Step 6: Prepare dual-branch inputs --
        logger.info("\nStep 6: Preparing Dual-Branch Inputs")
        logger.info("  Spatial: 1D(64) → 8×8 → bicubic → 28×28 images")
        logger.info("  Temporal: 1D(64) → sequence(64, 1)")

        # Spatial: images (N, 1, 28, 28)
        logger.info("  Converting training set to images...")
        X_train_img = self._reshaper.transform_batch(X_train_bal, verbose=self.verbose)
        logger.info("  Converting validation set to images...")
        X_val_img   = self._reshaper.transform_batch(X_val_norm,  verbose=self.verbose)
        logger.info("  Converting test set to images...")
        X_test_img  = self._reshaper.transform_batch(X_test_norm, verbose=self.verbose)

        # Temporal: sequences (N, 64, 1)
        X_train_seq = self._formatter.format(X_train_bal)
        X_val_seq   = self._formatter.format(X_val_norm)
        X_test_seq  = self._formatter.format(X_test_norm)

        # -- Step 7: Save all arrays to disk --
        logger.info("\nStep 7: Saving processed arrays to disk")
        self._save_arrays(
            X_train_img, X_val_img, X_test_img,
            X_train_seq, X_val_seq, X_test_seq,
            X_train_bal, X_val_norm, X_test_norm,
            y_train_bal, y_val, y_test,
        )

        # Save pipeline manifest
        self._save_manifest(
            selected_feature_names, class_map, num_classes,
            X_train_bal.shape, X_val_norm.shape, X_test_norm.shape
        )

        # -- Assemble ProcessedDataset --
        dataset = ProcessedDataset()
        dataset.X_train_img = X_train_img
        dataset.X_val_img   = X_val_img
        dataset.X_test_img  = X_test_img
        dataset.X_train_seq = X_train_seq
        dataset.X_val_seq   = X_val_seq
        dataset.X_test_seq  = X_test_seq
        dataset.X_train_flat = X_train_bal
        dataset.X_val_flat   = X_val_norm
        dataset.X_test_flat  = X_test_norm
        dataset.y_train = y_train_bal
        dataset.y_val   = y_val
        dataset.y_test  = y_test
        dataset.class_names  = CLASS_NAMES
        dataset.num_classes  = num_classes
        dataset.num_features = self.config["n_features"]
        dataset.feature_names = selected_feature_names
        dataset.config = self.config

        elapsed = time.time() - t_pipeline_start
        logger.info(f"\nPipeline completed in {elapsed:.1f}s")
        logger.info("\n" + dataset.summary())

        return dataset

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_exists(self) -> bool:
        manifest = self.augmented_dir / CACHE_MANIFEST_FILE
        arrays_dir = self.augmented_dir / "arrays"
        required_files = [
            "X_train_img.npy", "X_val_img.npy", "X_test_img.npy",
            "X_train_seq.npy", "X_val_seq.npy", "X_test_seq.npy",
            "y_train.npy", "y_val.npy", "y_test.npy",
        ]
        if not manifest.exists():
            return False
        return all((arrays_dir / f).exists() for f in required_files)

    def _load_cache(self) -> ProcessedDataset:
        logger.info("Loading cached arrays from disk...")
        d = self.augmented_dir / "arrays"   # arrays saved in arrays/ subdir

        dataset = ProcessedDataset()
        dataset.X_train_img  = np.load(d / "X_train_img.npy")
        dataset.X_val_img    = np.load(d / "X_val_img.npy")
        dataset.X_test_img   = np.load(d / "X_test_img.npy")
        dataset.X_train_seq  = np.load(d / "X_train_seq.npy")
        dataset.X_val_seq    = np.load(d / "X_val_seq.npy")
        dataset.X_test_seq   = np.load(d / "X_test_seq.npy")
        dataset.X_train_flat = np.load(d / "X_train_flat.npy")
        dataset.X_val_flat   = np.load(d / "X_val_flat.npy")
        dataset.X_test_flat  = np.load(d / "X_test_flat.npy")
        dataset.y_train = np.load(d / "y_train.npy")
        dataset.y_val   = np.load(d / "y_val.npy")
        dataset.y_test  = np.load(d / "y_test.npy")

        # Load manifest
        with open(d / CACHE_MANIFEST_FILE) as f:
            manifest = json.load(f)
        dataset.class_names   = {int(k): v for k, v in manifest["class_names"].items()}
        dataset.num_classes   = manifest["num_classes"]
        dataset.num_features  = manifest["num_features"]
        dataset.feature_names = manifest["feature_names"]
        dataset.config        = manifest["config"]

        logger.info("Cache loaded successfully")
        logger.info("\n" + dataset.summary())
        return dataset

    def _save_arrays(self, *arrays_with_names) -> None:
        """
        Save all processed arrays into organised subdirectories:

          data/augmented/
            arrays/        <- .npy files for fast reload during training
            smote_output/  <- SMOTE-balanced flat features, CSV sample,
                              class distribution summary
        """
        import pandas as pd

        names = [
            "X_train_img", "X_val_img", "X_test_img",
            "X_train_seq", "X_val_seq", "X_test_seq",
            "X_train_flat", "X_val_flat", "X_test_flat",
            "y_train", "y_val", "y_test",
        ]

        # ── arrays/ subdirectory ──────────────────────────────────────
        arrays_dir = self.augmented_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        arr_dict = {}
        for name, arr in zip(names, arrays_with_names):
            save_path = arrays_dir / f"{name}.npy"
            np.save(save_path, arr)
            mb = arr.nbytes / 1e6
            logger.info(f"  Saved {name}: {arr.shape} ({mb:.1f} MB) -> arrays/")
            arr_dict[name] = arr

        # Update _cache_exists lookup path (arrays now in arrays/ subdir)
        # _load_cache is updated correspondingly below.

        # ── smote_output/ subdirectory ────────────────────────────────
        smote_dir = self.augmented_dir / "smote_output"
        smote_dir.mkdir(parents=True, exist_ok=True)

        X_flat = arr_dict["X_train_flat"]   # (N, 64) balanced normalised features
        y_bal  = arr_dict["y_train"]         # (N,)   balanced integer labels

        # Save full balanced arrays as .npy
        np.save(smote_dir / "X_train_balanced.npy", X_flat)
        np.save(smote_dir / "y_train_balanced.npy", y_bal)
        logger.info(
            f"  SMOTE arrays saved -> smote_output/ "
            f"[X:{X_flat.shape}, y:{y_bal.shape}]"
        )

        # Class distribution summary text file
        classes, counts = np.unique(y_bal, return_counts=True)
        total = int(len(y_bal))
        lines = [
            "SMOTE-ENN Output — Class Distribution Summary",
            "=" * 55,
            f"Total training samples after SMOTE-ENN: {total:,}",
            "",
            f"  {'Class ID':<10} {'Class Name':<22} {'Count':>10} {'Percent':>9}",
            "  " + "-" * 53,
        ]
        for cls, cnt in zip(classes.tolist(), counts.tolist()):
            name_str = CLASS_NAMES.get(cls, str(cls)) if isinstance(CLASS_NAMES, dict) else str(cls)
            pct = cnt / total * 100
            lines.append(f"  {cls:<10} {name_str:<22} {cnt:>10,} {pct:>8.1f}%")
        lines += [
            "  " + "-" * 53,
            f"  {'TOTAL':<32} {total:>10,} {'100.0%':>9}",
            "",
            "Array shapes:",
            f"  X_train_flat : {X_flat.shape}  (64 normalised features per sample)",
            f"  y_train      : {y_bal.shape}",
            "",
            "Files in smote_output/:",
            "  X_train_balanced.npy  <- full balanced feature matrix",
            "  y_train_balanced.npy  <- corresponding labels",
            "  smote_balanced_sample.csv  <- 50K-row readable sample",
            "  class_distribution.txt     <- this file",
        ]
        summary_text = "\n".join(lines)
        (smote_dir / "class_distribution.txt").write_text(summary_text)
        logger.info("  Class distribution -> smote_output/class_distribution.txt")
        logger.info("\n" + summary_text)

        # Balanced sample CSV (up to 50K rows — full file can be 100+ MB)
        max_rows = min(50_000, total)
        rng = np.random.default_rng(42)
        idx = rng.choice(total, size=max_rows, replace=False)
        feat_cols = [f"feature_{i:02d}" for i in range(X_flat.shape[1])]
        df = pd.DataFrame(X_flat[idx], columns=feat_cols)
        df["label"] = y_bal[idx]
        df.to_csv(smote_dir / "smote_balanced_sample.csv", index=False)
        logger.info(
            f"  CSV sample ({max_rows:,} rows) -> smote_output/smote_balanced_sample.csv"
        )


    def _save_manifest(
        self,
        feature_names: List[str],
        class_map: Dict,
        num_classes: int,
        train_shape: tuple,
        val_shape: tuple,
        test_shape: tuple,
    ) -> None:
        manifest = {
            "feature_names": feature_names,
            "class_names": CLASS_NAMES,
            "num_classes": num_classes,
            "num_features": self.config["n_features"],
            "train_shape": list(train_shape),
            "val_shape": list(val_shape),
            "test_shape": list(test_shape),
            "config": self.config,
        }
        path = self.augmented_dir / CACHE_MANIFEST_FILE
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"  Pipeline manifest saved: {path}")

    def _save_split_csv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        path: Path,
    ) -> None:
        import pandas as pd
        df = pd.DataFrame(X, columns=feature_names)
        df["Label"] = y
        df.to_csv(path, index=False)
        logger.info(f"  Saved {path.name}: {df.shape}")

    def get_normalizer(self) -> Optional[MinMaxNormalizer]:
        return self._normalizer

    def get_selector(self) -> Optional[FeatureSelector]:
        return self._selector


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Res-TranBiLSTM DoS Preprocessing Pipeline"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to Wednesday-workingHours.csv"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Path to ExistingImplementation/ root"
    )
    parser.add_argument(
        "--load-cache",
        action="store_true",
        help="Load from cached .npy files if available"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cache and re-run full pipeline"
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=PIPELINE_CONFIG["smote_target_per_class"],
        help="SMOTE target samples per class"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config_override = {
        "smote_target_per_class": args.target_per_class,
    }

    pipeline = DOSPreprocessingPipeline(
        base_dir=args.base_dir,
        config=config_override,
        verbose=True,
    )

    force = args.force_rerun or not args.load_cache
    dataset = pipeline.run(
        csv_path=args.csv,
        force_rerun=force,
    )

    print("\n" + "=" * 65)
    print(dataset.summary())
    print("=" * 65)
    print("\nPreprocessing pipeline complete. Arrays ready for training.")


if __name__ == "__main__":
    main()