# Phase 2 — Proposed Implementation: Ensemble-TranBiLSTM

Proposed improvement over **Wang et al. (2023)** — _"Res-TranBiLSTM: A Hybrid Deep Learning Model for Network Intrusion Detection"_ — applied to the DoS-only subset of the **CIC-IDS2017** dataset (Wednesday traffic capture).

This is Phase 2 of the FYP. It replaces the single heavy ResNet-18 spatial branch with a **MobileNetV2 + EfficientNet-B0 soft ensemble**, and replaces SMOTE-ENN data augmentation with **CTGAN** (Conditional Tabular GAN), achieving comparable accuracy at a fraction of the computational cost — making the model viable for real-time deployment on IoT fog nodes.

---

## Results (Phase 2 vs Phase 1 Baseline)

| Metric           | Phase 1 (Res-TranBiLSTM) | Phase 2 (Ensemble-TranBiLSTM)              | Change     |
| ---------------- | ------------------------ | ------------------------------------------ | ---------- |
| Accuracy         | 98.90%                   | **99.15%**                                 | +0.25%     |
| Precision        | 97.03%                   | **95.97%**                                 | −1.06%     |
| Recall           | 99.15%                   | **99.20%**                                 | +0.05%     |
| F1-Score         | 98.06%                   | **97.54%**                                 | −0.52%     |
| Total Parameters | 11,334,117               | **3,243,238**                              | **−71.4%** |
| Model Size       | 45.34 MB                 | **12.97 MB**                               | **−71.4%** |
| Total MACs       | 126,596,741              | **113,793,110**                            | **−10.1%** |
| Total GMACs      | 0.1266                   | **0.1138**                                 | **−10.1%** |
| Total FLOPs      | 253,193,482              | **227,586,220**                            | **−10.1%** |
| Training Time    | ~22 min                  | ~75 min (RTX 2050, 20 epochs, 75K samples) | —          |

> **Research contribution:** Phase 2 achieves equal or better accuracy than the Phase 1 baseline while using **71% fewer parameters** (3.2M vs 11.3M) and **10% fewer FLOPs**, directly validating the fog-node deployment argument: two lightweight models combined cost less than one ResNet-18.

---

## Architecture

The model follows the same dual-branch design as Phase 1 (Fig. 2, Wang et al. 2023), with the spatial branch replaced by a two-model ensemble:

```
Raw network features (78 cols)
        │
        ▼
Feature Selection (SelectKBest, k=64)
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
[Spatial Branch — Ensemble]       [Temporal Branch]
 1D(64) → 8×8 grid                1D(64) → sequence(64,1)
       → bicubic → 28×28                  │
         image                     MLP encoder (FC-2)
         │                                │
   ┌─────┴─────┐                 Transformer encoder
   ▼           ▼                    + BiLSTM (FC-3)
MobileNetV2  EfficientNet-B0              │
 (544K)       (2.6M)             temporal_feat(128)
   │           │                          │
   └─────┬─────┘                          │
   avg(feat_a, feat_b)                    │
   spatial_feat(128)                      │
         │                                │
         └──────────────┬─────────────────┘
                        ▼
               concat → (256,)
                        │
             Classification Head
            FC-4(128) → FC-5(64)
               → FC-6(6) → softmax
                        │
                        ▼
            6-class prediction
```

### Ensemble Design (soft voting)

Both branches process the **same 28×28 input independently in parallel**. Their `(B, 128)` feature vectors are averaged (soft feature ensemble), producing one `(B, 128)` spatial feature vector. This is equivalent to soft voting and prevents any single model from being overconfident — critical for Heartbleed with only 11 training samples.

| Branch                | Inductive bias                                       | Activation | SE blocks | Params        |
| --------------------- | ---------------------------------------------------- | ---------- | --------- | ------------- |
| MobileNetV2Branch     | Local spatial patterns via depthwise separable convs | ReLU6      | No        | 544,768       |
| EfficientNetB0Branch  | Channel-wise relationships via SE attention          | SiLU       | Yes       | 2,610,304     |
| **Ensemble combined** | Both views averaged                                  | —          | —         | **3,155,072** |

**28×28 adaptation:** Both branches have their stem stride changed from 2 → 1 and early downsampling stages adjusted so that spatial dimensions are not collapsed before meaningful features are learned from the small 28×28 traffic images.

### Parameter breakdown

| Module                           | Parameters    | Share |
| -------------------------------- | ------------- | ----- |
| Spatial Branch (MobileNetV2)     | 544,768       | 16.8% |
| Spatial Branch (EfficientNet-B0) | 2,610,304     | 80.5% |
| Temporal Branch (TranBiLSTM)     | 46,624        | 1.4%  |
| Classification Head              | 41,542        | 1.3%  |
| **Total**                        | **3,243,238** | —     |

### MACs breakdown

| Module                            | MACs  | Share |
| --------------------------------- | ----- | ----- |
| Spatial (ensemble, Conv2d layers) | ~108M | ~95%  |
| Temporal (GRU + Transformer)      | ~5M   | ~4%   |
| Classifier (Linear layers)        | ~0.7M | ~1%   |

---

## Dataset

**CIC-IDS2017** — Wednesday working hours capture (6-class including Heartbleed).

| Class            | Label | Original Count | After CTGAN |
| ---------------- | ----- | -------------- | ----------- |
| BENIGN           | 0     | 351,279        | 15,000      |
| DoS slowloris    | 1     | 4,636          | 15,000      |
| DoS Slowhttptest | 2     | 4,399          | 15,000      |
| DoS Hulk         | 3     | 183,358        | 15,000      |
| DoS GoldenEye    | 4     | 8,228          | 15,000      |
| Heartbleed       | 5     | 11             | 15,000      |

**Split:** 80% train / 10% validation / 10% test (stratified).

**CTGAN** is applied to the training set only. It learns the real joint feature distribution per class through Generator/Discriminator competition, producing diverse and realistic synthetic minority samples — unlike SMOTE's linear interpolation between existing samples, which generates repetitive variations and is prone to overfitting on rare classes like Heartbleed (11 real samples).

---

## Project Structure

```
ProposedImplementation/
├── configs/
│   ├── dos_config.yaml           ← Main config (edit here, not in code)
│   └── proposed_config.yaml      ← Alternative config (legacy)
│
├── data/
│   ├── raw/
│   │   └── Wednesday-workingHours.csv   ← CIC-IDS2017 source file
│   ├── processed/                ← Normalised train/val/test arrays (auto-generated)
│   └── augmented/
│       ├── arrays/               ← .npy cache for fast reload (auto-generated)
│       │   ├── X_train_img.npy   (75K, 1, 28, 28) — spatial branch input
│       │   ├── X_train_seq.npy   (75K, 64, 1)     — temporal branch input
│       │   ├── X_train_flat.npy  (75K, 64)         — CTGAN-balanced flat features
│       │   ├── y_train.npy       (75K,)
│       │   ├── X_val_*.npy / X_test_*.npy / y_val.npy / y_test.npy
│       │   └── pipeline_manifest.json
│       └── ctgan_output/         ← CTGAN artefacts per class (auto-generated)
│
├── logs/                         ← Per-run log files (auto-generated)
│
├── results/
│   ├── checkpoints/              ← Best model weights (auto-generated)
│   ├── metrics/                  ← JSON metrics + FLOPs reports (auto-generated)
│   └── plots/                    ← Confusion matrix, training curves (auto-generated)
│
└── src/
    ├── train.py                  ← Main entry point
    ├── evaluation/
    │   ├── metrics.py            ← Accuracy, precision, recall, F1, confusion matrix
    │   ├── flops_counter.py      ← MAC/FLOPs counter (hook-based, no external libs)
    │   └── comparison_report.py  ← Phase 1 vs Phase 2 side-by-side report generator
    ├── models/
    │   ├── proposed_model.py     ← Full model assembly (LightweightIDSModel)
    │   ├── spatial/
    │   │   ├── dsconv_block.py   ← MobileNetV2Branch + EfficientNetB0Branch definitions
    │   │   └── spatial_extractor.py  ← EnsembleSpatialExtractor (soft feature avg)
    │   ├── temporal/
    │   │   └── temporal_extractor.py ← MLP + LinearAttn + BiGRU temporal branch
    │   └── classification/
    │       └── classifier.py     ← FC-4 → FC-5 → FC-6 head (identical to Phase 1)
    ├── preprocessing/
    │   ├── data_pipeline.py      ← Orchestrates all preprocessing steps
    │   ├── dataset_loader.py     ← CSV loading and label encoding
    │   ├── ctgan_handler.py      ← CTGAN training and synthesis per minority class
    │   ├── feature_encoder.py    ← SelectKBest feature selection
    │   ├── normalizer.py         ← MinMax normalisation
    │   └── smote_enn.py          ← SMOTE-ENN (retained for ablation comparison)
    ├── training/
    │   └── trainer.py            ← Training loop, early stopping, checkpointing
    └── utils/
        ├── checkpointer.py       ← Save/load best model weights
        ├── logger.py             ← Timestamped file + console logging
        └── seed.py               ← Global seed for reproducibility
```

---

## Setup

### Prerequisites

- Python 3.11.8
- NVIDIA GPU with CUDA 11.8 (tested on RTX 2050, 4GB VRAM)
- Windows 10/11 (Linux also supported)
- Phase 1 (`ExistingImplementation/`) must be present at the same directory level for the comparison report

### 1. Create virtual environment (shared with Phase 1)

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

Download `Wednesday-workingHours.pcap_ISCX.csv` from the CIC-IDS2017 dataset and rename/place it at:

```
ProposedImplementation/data/raw/Wednesday-workingHours.csv
```

---

## Running

All commands are run from inside `ProposedImplementation/`.

### Full pipeline (first run)

Runs preprocessing → CTGAN augmentation → training → evaluation → FLOPs profiling. CTGAN adds ~30–40 minutes on top of the ~35 minute training time on RTX 2050.

```bash
python src/train.py --config configs/dos_config.yaml
```

### Load cached preprocessed data (subsequent runs)

Skips preprocessing and CTGAN (which is the expensive step). Uses `.npy` files saved in `data/augmented/arrays/`.

```bash
python src/train.py --config configs/dos_config.yaml --load-cache
```

### Force full rerun (discard cache)

Use this after changing CTGAN settings or feature selection parameters.

```bash
python src/train.py --config configs/dos_config.yaml --force-rerun
```

### Override settings from command line

```bash
# Use a different CSV path
python src/train.py --config configs/dos_config.yaml --csv path/to/data.csv

# Override number of epochs
python src/train.py --config configs/dos_config.yaml --epochs 30

# Force CPU (useful for debugging)
python src/train.py --config configs/dos_config.yaml --device cpu
```

### Generate Phase 1 vs Phase 2 comparison report

```bash
python src/evaluation/comparison_report.py \
    --existing-ckpt  ../ExistingImplementation/results/checkpoints/res_tranbilstm_dos_best.pth \
    --proposed-ckpt  results/checkpoints/ensemble_res_tranbilstm_dos_best.pth \
    --test-img       data/processed/X_test_img.npy \
    --test-seq       data/processed/X_test_seq.npy \
    --test-labels    data/processed/y_test.npy \
    --output         results/metrics/comparison_report.json
```

---

## Configuration

All hyperparameters live in `configs/dos_config.yaml`. Key settings:

| Parameter                | Value  | Phase 1 value  | Notes                                                      |
| ------------------------ | ------ | -------------- | ---------------------------------------------------------- |
| `data.n_features`        | 64     | 64             | Features after SelectKBest — identical for fair comparison |
| `data.spatial_large`     | 28     | 28             | Image size — identical                                     |
| `data.num_classes`       | 6      | 5              | Phase 2 includes Heartbleed                                |
| `ctgan.target_per_class` | 15,000 | 15,000 (SMOTE) | Same total training size for fair comparison               |
| `ctgan.epochs`           | 50     | —              | CTGAN training iterations per minority class               |
| `training.batch_size`    | 256    | 256            | Per paper Table 10                                         |
| `training.learning_rate` | 0.0001 | 0.0001         | Adam optimiser                                             |
| `training.max_epochs`    | 30     | 30             | Matched to Phase 1 for fair runtime comparison             |
| `training.patience`      | 10     | 10             | Early stopping                                             |
| `training.dropout`       | 0.5    | 0.5            | Per paper Table 10                                         |

---

## Output Files

After a successful run the following are created automatically:

| Path                                     | Contents                                                              |
| ---------------------------------------- | --------------------------------------------------------------------- |
| `logs/train_YYYYMMDD_HHMMSS.log`         | Full timestamped training log                                         |
| `results/checkpoints/<n>_best.pt`        | Best model weights (lowest val loss)                                  |
| `results/metrics/<n>_test_metrics.json`  | Accuracy, precision, recall, F1, per-class breakdown                  |
| `results/metrics/<n>_flops_report.json`  | MACs, GMACs, FLOPs, GFLOPs, params, by-module and by-branch breakdown |
| `results/metrics/<n>_flops_layers.json`  | Full per-layer FLOPs breakdown                                        |
| `results/plots/<n>_confusion_matrix.png` | Normalised confusion matrix                                           |
| `results/plots/<n>_per_class.png`        | Per-class precision/recall/F1 bar chart                               |
| `results/plots/<n>_training_history.png` | Train/val loss and accuracy curves                                    |
| `data/augmented/arrays/*.npy`            | Preprocessed arrays for fast cache reload                             |
| `data/augmented/ctgan_output/`           | CTGAN model checkpoints and synthesis logs                            |

---

## Dependency Notes

### Critical version pins

| Package            | Pinned version | Reason                                                                       |
| ------------------ | -------------- | ---------------------------------------------------------------------------- |
| `numpy`            | 1.26.4         | PyTorch 2.1.x compiled against NumPy 1.x ABI — NumPy 2.x causes import crash |
| `opencv-python`    | 4.8.1.78       | OpenCV 4.9+ requires NumPy ≥ 2.0, incompatible with above pin                |
| `scikit-learn`     | 1.4.2          | imbalanced-learn 0.12.3 requires sklearn < 1.6                               |
| `imbalanced-learn` | 0.12.3         | Must match scikit-learn pin above                                            |

### CTGAN memory requirements

CTGAN training runs on CPU and requires approximately 2–4 GB RAM per minority class. Total CTGAN augmentation time on a modern CPU is 30–50 minutes for 6 classes at 50 epochs each. The trained generator is cached — subsequent runs with `--load-cache` skip this entirely.

### Windows-specific

`num_workers=0` is set in all DataLoader instances. Windows uses `spawn` for multiprocessing which causes a bootstrapping error with any value > 0.

---

## Reference

Wang, Y. et al. (2023). _Res-TranBiLSTM: An Efficient Intrusion Detection Method for the Internet of Things_. Computers & Security / IEEE Access.

Dataset: Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). _Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization_. ICISSP. [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

Sandler, M. et al. (2018). _MobileNetV2: Inverted Residuals and Linear Bottlenecks_. CVPR.

Tan, M. & Le, Q. V. (2019). _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_. ICML.

Xu, L. et al. (2019). _Modeling Tabular Data using Conditional GAN_. NeurIPS.
