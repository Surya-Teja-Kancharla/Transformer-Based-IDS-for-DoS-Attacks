# Phase 1 — Existing Implementation: Res-TranBiLSTM

Replication of **Wang et al. (2023)** — _"Res-TranBiLSTM: A Hybrid Deep Learning Model for Network Intrusion Detection"_ — applied to the DoS-only subset of the **CIC-IDS2017** dataset (Wednesday traffic capture).

This is Phase 1 of the FYP. Its purpose is to establish a **baseline** of model parameters, FLOPs, and classification metrics that Phase 2 (ProposedImplementation) will improve upon.

---

## Results (Phase 1 Baseline)

| Metric           | Value                                      |
| ---------------- | ------------------------------------------ |
| Accuracy         | **98.90%**                                 |
| Precision        | **97.03%**                                 |
| Recall           | **99.15%**                                 |
| F1-Score         | **98.06%**                                 |
| Total Parameters | **11,334,117**                             |
| Model Size       | **45.34 MB** (float32)                     |
| Total MACs       | **126,596,741**                            |
| Total GMACs      | **0.1266**                                 |
| Total FLOPs      | **253,193,482** (= 2 × MACs)               |
| Total GFLOPs     | **0.2532**                                 |
| Training Time    | ~22 min (RTX 2050, 30 epochs, 74K samples) |

> Paper target accuracy: 99.15%. The 0.25% gap is expected — the paper used 100 epochs and full SMOTE. This implementation uses 30 epochs and capped SMOTE (15K/class) to keep runtime under 2 hours on a 4GB GPU while preserving the parameter/FLOPs measurement goal.

---

## Architecture

The model follows the dual-branch design from Fig. 2 of Wang et al. (2023):

```
Raw network features (78 cols)
        │
        ▼
Feature Selection (SelectKBest, k=64)
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
[Spatial Branch]                  [Temporal Branch]
1D(64) → 8×8 grid                 1D(64) → sequence(64,1)
      → bicubic → 28×28                   │
        image                      MLP encoder (FC-2)
        │                                 │
   ResNet (11.2M params)         Transformer encoder
        │                           + BiLSTM (FC-3)
        ▼                                 │
  spatial_feat(128)             temporal_feat(128)
        │                                 │
        └─────────────┬───────────────────┘
                      ▼
              concat → (256,)
                      │
           Classification Head
          FC-4(128) → FC-5(64)
             → FC-6(5) → softmax
                      │
                      ▼
          5-class prediction
```

### Parameter breakdown

| Module                       | Parameters     | Share |
| ---------------------------- | -------------- | ----- |
| Spatial Branch (ResNet)      | 11,233,344     | 99.1% |
| Temporal Branch (TranBiLSTM) | 59,296         | 0.5%  |
| Classification Head          | 41,477         | 0.4%  |
| **Total**                    | **11,334,117** | —     |

### MACs breakdown

| Module                        | MACs        | Share |
| ----------------------------- | ----------- | ----- |
| Spatial (Conv2d layers)       | 122,555,520 | 96.8% |
| Temporal (LSTM + Transformer) | 3,999,744   | 3.2%  |
| Classifier (Linear layers)    | 41,477      | ~0%   |

The ResNet spatial branch accounts for 99.1% of parameters and 96.8% of computation. This is the primary target for reduction in Phase 2.

---

## Dataset

**CIC-IDS2017** — Wednesday working hours capture.

| Class            | Label | Original Count | After SMOTE-ENN |
| ---------------- | ----- | -------------- | --------------- |
| BENIGN           | 0     | 351,279        | ~14,900         |
| DoS Slowloris    | 1     | 4,636          | ~14,900         |
| DoS Slowhttptest | 2     | 4,399          | ~14,900         |
| DoS Hulk         | 3     | 183,358        | ~14,900         |
| DoS GoldenEye    | 4     | 8,228          | ~14,900         |

**Split:** 80% train / 10% validation / 10% test (stratified).

**SMOTE-ENN** is applied to the training set only. All minority classes are oversampled and majority classes subsampled to a cap of 15,000 per class (75,000 total), keeping RAM and training time manageable on a 4GB GPU.

---

## Project Structure

```
ExistingImplementation/
├── configs/
│   └── dos_config.yaml          ← All hyperparameters (edit here, not in code)
│
├── data/
│   ├── raw/
│   │   └── Wednesday-workingHours.csv   ← CIC-IDS2017 source file
│   ├── processed/               ← Normalised train/val/test CSVs (auto-generated)
│   └── augmented/
│       ├── arrays/              ← .npy cache for fast reload (auto-generated)
│       │   ├── X_train_img.npy  (74K, 1, 28, 28) — spatial branch input
│       │   ├── X_train_seq.npy  (74K, 64, 1)     — temporal branch input
│       │   ├── X_train_flat.npy (74K, 64)         — balanced flat features
│       │   ├── y_train.npy      (74K,)
│       │   ├── X_val_*.npy / X_test_*.npy / y_val.npy / y_test.npy
│       │   └── pipeline_manifest.json
│       └── smote_output/        ← Human-readable SMOTE artefacts
│           ├── X_train_balanced.npy
│           ├── y_train_balanced.npy
│           ├── class_distribution.txt   ← Per-class sample counts
│           └── smote_balanced_sample.csv ← 50K-row sample (Excel-friendly)
│
├── logs/                        ← Per-run log files (auto-generated)
│
├── results/
│   ├── checkpoints/             ← Best model weights (auto-generated)
│   ├── metrics/                 ← JSON metrics + FLOPs reports (auto-generated)
│   └── plots/                   ← Confusion matrix, training curves (auto-generated)
│
└── src/
    ├── train.py                 ← Main entry point
    ├── evaluation/
    │   ├── metrics.py           ← Accuracy, precision, recall, F1, confusion matrix
    │   └── flops_counter.py     ← MAC/FLOPs counter (hook-based, no external libs)
    ├── models/
    │   ├── res_tranbilstm.py    ← Full model assembly
    │   ├── spatial/
    │   │   ├── resnet_block.py  ← ResidualBlock definition
    │   │   └── spatial_extractor.py  ← ResNet spatial branch
    │   ├── temporal/
    │   │   └── temporal_extractor.py ← MLP + Transformer + BiLSTM temporal branch
    │   └── classification/
    │       └── classifier.py    ← FC-4 → FC-5 → FC-6 head
    ├── preprocessing/
    │   ├── data_pipeline.py     ← Orchestrates all preprocessing steps
    │   ├── dataset_loader.py    ← CSV loading and label encoding
    │   ├── feature_encoder.py   ← SelectKBest feature selection
    │   ├── normalizer.py        ← MinMax normalisation
    │   └── smote_enn.py         ← SMOTE-ENN with class-by-class progress
    ├── training/
    │   └── trainer.py           ← Training loop, early stopping, checkpointing
    └── utils/
        ├── checkpointer.py      ← Save/load best model weights
        ├── logger.py            ← Timestamped file + console logging
        └── seed.py              ← Global seed for reproducibility
```

---

## Setup

### Prerequisites

- Python 3.11.8
- NVIDIA GPU with CUDA 11.8 (tested on RTX 2050, 4GB VRAM)
- Windows 10/11 (Linux also supported)

### 1. Create virtual environment

```bash
python -m venv FYP_env
FYP_env\Scripts\activate        # Windows
# source FYP_env/bin/activate   # Linux/macOS
```

### 2. Install PyTorch with CUDA 11.8

Must be done first — plain `pip install torch` fetches a CPU-only build.

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU is detected

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 2050
```

### 5. Place the dataset

Download `Wednesday-workingHours.csv` from the CIC-IDS2017 dataset and rename/place it at:

```
ExistingImplementation/data/raw/Wednesday-workingHours.csv
```

---

## Running

All commands are run from inside `ExistingImplementation/`.

### Full pipeline (first run)

Runs preprocessing → SMOTE-ENN → training → evaluation → FLOPs profiling. Takes ~1 hour total on RTX 2050.

```bash
python src/train.py --config configs/dos_config.yaml
```

### Load cached preprocessed data (subsequent runs)

Skips the ~10 minute preprocessing and SMOTE step. Uses `.npy` files saved in `data/augmented/arrays/`.

```bash
python src/train.py --config configs/dos_config.yaml --load-cache
```

### Force full rerun (discard cache)

Use this after changing SMOTE settings or feature selection parameters.

```bash
python src/train.py --config configs/dos_config.yaml --force-rerun
```

### Override settings from command line

```bash
# Use a different CSV path
python src/train.py --config configs/dos_config.yaml --csv path/to/data.csv

# Override number of epochs
python src/train.py --config configs/dos_config.yaml --epochs 50

# Force CPU (useful for debugging)
python src/train.py --config configs/dos_config.yaml --device cpu
```

---

## Configuration

All hyperparameters live in `configs/dos_config.yaml`. Key settings:

| Parameter                     | Value     | Paper value          | Notes                           |
| ----------------------------- | --------- | -------------------- | ------------------------------- |
| `n_features`                  | 64        | 64                   | Features after SelectKBest      |
| `spatial_large`               | 28        | 28                   | Image size (bicubic upsampled)  |
| `smote_enn.target_per_class`  | 15,000    | ~351K (not majority) | Capped to keep RAM safe         |
| `smote_enn.sampling_strategy` | `"fixed"` | `"not majority"`     | Prevents 1.56M sample explosion |
| `training.batch_size`         | 256       | 256                  | Per paper Table 10              |
| `training.learning_rate`      | 0.0001    | 0.0001               | Adam optimiser                  |
| `training.max_epochs`         | 30        | 100                  | Reduced for FYP runtime         |
| `training.patience`           | 10        | —                    | Early stopping                  |
| `training.dropout`            | 0.5       | 0.5                  | Per paper Table 10              |

---

## Output Files

After a successful run the following are created automatically:

| Path                                                 | Contents                                                |
| ---------------------------------------------------- | ------------------------------------------------------- |
| `logs/train_YYYYMMDD_HHMMSS.log`                     | Full timestamped training log                           |
| `results/checkpoints/<name>_best.pt`                 | Best model weights (lowest val loss)                    |
| `results/metrics/<name>_test_metrics.json`           | Accuracy, precision, recall, F1, per-class breakdown    |
| `results/metrics/<name>_flops_report.json`           | MACs, GMACs, FLOPs, GFLOPs, params, by-module breakdown |
| `results/metrics/<name>_flops_layers.json`           | Full per-layer FLOPs breakdown                          |
| `results/plots/<name>_confusion_matrix.png`          | Normalised confusion matrix                             |
| `results/plots/<name>_per_class.png`                 | Per-class precision/recall/F1 bar chart                 |
| `results/plots/<name>_training_history.png`          | Train/val loss and accuracy curves                      |
| `data/augmented/arrays/*.npy`                        | Preprocessed arrays for fast cache reload               |
| `data/augmented/smote_output/class_distribution.txt` | SMOTE-ENN per-class sample counts                       |

---

## Dependency Notes

### Critical version pins

These three packages form an interdependent chain — changing any one breaks the others:

| Package            | Pinned version | Reason                                                                        |
| ------------------ | -------------- | ----------------------------------------------------------------------------- |
| `numpy`            | 1.26.4         | PyTorch 2.1.x compiled against NumPy 1.x ABI — NumPy 2.x causes import crash  |
| `opencv-python`    | 4.8.1.78       | OpenCV 4.9+ requires NumPy ≥ 2.0, incompatible with above pin                 |
| `scikit-learn`     | 1.4.2          | imbalanced-learn 0.12.3 requires sklearn < 1.6 (1.6+ removed `parse_version`) |
| `imbalanced-learn` | 0.12.3         | Must match scikit-learn pin above                                             |

### Windows-specific

`num_workers=0` is set in all DataLoader instances. Windows uses `spawn` for multiprocessing which causes `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` with any value > 0.

---

## Reference

Wang, Y. et al. (2023). _Res-TranBiLSTM: An Efficient Intrusion Detection Method for the Internet of Things_. Computers & Security / IEEE Access.

Dataset: Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). _Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization_. ICISSP. [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).
