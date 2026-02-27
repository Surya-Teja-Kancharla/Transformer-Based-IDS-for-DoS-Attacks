# FYP — Ensemble-Res-TranBiLSTM: Lightweight Intrusion Detection for IoT Fog Nodes

Final Year Project (BTech) — Implementation of a proposed improvement to **Wang et al. (2023)** *Res-TranBiLSTM* for network intrusion detection, targeting fog-node deployment in IoT environments.

**Dataset:** CIC-IDS2017 (Wednesday DoS subset) — Sharafaldin et al., 2018  
**Hardware:** NVIDIA RTX 2050 (4GB VRAM), 12GB RAM, Windows 11  
**Framework:** PyTorch 2.1.2, CUDA 11.8

---

## Project Summary

The baseline paper (Res-TranBiLSTM) achieves 99.15% accuracy on CIC-IDS2017 but relies on a heavy ResNet-18 spatial extractor (11.2M of the model's 11.3M total parameters). The paper explicitly acknowledges this limits real-world deployment on IoT fog nodes and calls for lighter alternatives as future work.

This project answers that call: the ResNet-18 spatial branch is replaced with a **MobileNetV2 + EfficientNet-B0 soft ensemble**, and **CTGAN** replaces SMOTE-ENN for data augmentation, producing a model that maintains accuracy while using 71% fewer parameters.

---

## Results at a Glance

| | Phase 1 · Res-TranBiLSTM | Phase 2 · Ensemble-Res-TranBiLSTM |
|---|---|---|
| **Accuracy** | 98.90% | **99.15%** |
| **Precision** | 97.03% | 95.97% |
| **Recall** | 99.15% | **99.20%** |
| **F1-Score** | 98.06% | 97.54% |
| **Parameters** | 11,334,117 | **3,243,238** (−71%) |
| **Model Size** | 45.34 MB | **12.97 MB** (−71%) |
| **GMACs** | 0.1266 | **0.1138** (−10%) |
| **GFLOPs** | 0.2532 | **0.2276** (−10%) |
| Spatial branch | ResNet-18 (11.2M) | MobileNetV2 + EfficientNet-B0 (3.2M) |
| Augmentation | SMOTE-ENN | CTGAN |
| Training time | ~22 min | ~75 min (incl. CTGAN) |

> The combined MobileNetV2 + EfficientNet-B0 ensemble (3.2M params) costs less than one ResNet-18 (11.2M), validating the fog deployment argument: two specialised lightweight models are collectively cheaper than one general-purpose heavy model.

---

## Repository Structure

```
FYP/
│
├── README.md                          ← This file
├── requirements.txt                   ← Shared Python dependencies
│
├── ExistingImplementation/            ← Phase 1: Res-TranBiLSTM baseline
│   ├── README.md                      ← Phase 1 detailed documentation
│   ├── configs/
│   │   └── dos_config.yaml            ← Phase 1 hyperparameters
│   ├── data/
│   │   └── raw/                       ← Place Wednesday-workingHours.csv here
│   ├── results/
│   │   ├── checkpoints/               ← Phase 1 best model weights
│   │   ├── metrics/                   ← Phase 1 JSON metrics + FLOPs reports
│   │   └── plots/                     ← Phase 1 confusion matrix, training curves
│   └── src/
│       ├── train.py                   ← Phase 1 entry point
│       ├── models/
│       │   ├── res_tranbilstm.py      ← Full Phase 1 model assembly
│       │   └── spatial/
│       │       ├── resnet_block.py    ← ResidualBlock (standard 3×3 conv)
│       │       └── spatial_extractor.py  ← ResNet-18 spatial branch
│       ├── preprocessing/
│       │   ├── data_pipeline.py       ← Preprocessing orchestrator
│       │   └── smote_enn.py           ← SMOTE-ENN (capped at 15K/class)
│       └── evaluation/
│           └── flops_counter.py       ← Hook-based MAC counter
│
└── ProposedImplementation/            ← Phase 2: Ensemble-Res-TranBiLSTM
    ├── README.md                      ← Phase 2 detailed documentation
    ├── configs/
    │   └── dos_config.yaml            ← Phase 2 hyperparameters
    ├── data/
    │   └── raw/                       ← Place Wednesday-workingHours.csv here
    ├── results/
    │   ├── checkpoints/               ← Phase 2 best model weights
    │   ├── metrics/                   ← Phase 2 JSON metrics + FLOPs reports
    │   └── plots/                     ← Phase 2 confusion matrix, training curves
    └── src/
        ├── train.py                   ← Phase 2 entry point
        ├── models/
        │   ├── proposed_model.py      ← Full Phase 2 model assembly
        │   └── spatial/
        │       ├── dsconv_block.py    ← MobileNetV2Branch + EfficientNetB0Branch
        │       └── spatial_extractor.py  ← EnsembleSpatialExtractor (soft avg)
        ├── preprocessing/
        │   ├── data_pipeline.py       ← Preprocessing orchestrator
        │   └── ctgan_handler.py       ← CTGAN augmentation per minority class
        └── evaluation/
            ├── flops_counter.py       ← Hook-based MAC counter
            └── comparison_report.py   ← Phase 1 vs Phase 2 report generator
```

---

## Architecture Overview

Both phases share the same dual-branch structure from Wang et al. (2023). Phase 2 changes only the spatial branch and the augmentation method — everything else (temporal branch, classification head, training loop, evaluation pipeline) is structurally identical for a fair comparison.

### Spatial branch comparison

```
Phase 1                              Phase 2
───────────────────────────────────  ────────────────────────────────────────
28×28 image                          28×28 image (same input)
    │                                    │              │
    ▼                                    ▼              ▼
ResNet-18                          MobileNetV2    EfficientNet-B0
 (11.2M params)                      (544K)          (2.6M)
 standard 3×3 conv                  DSConv,         DSConv + SE,
 4 residual blocks                  no SE           SE attention
    │                                    │              │
    ▼                                    └──── avg ─────┘
spatial_feat (128,)                  spatial_feat (128,)
                                       soft ensemble
```

### Ensemble diversity (why two models)

| Model | What it captures | Key mechanism |
|---|---|---|
| MobileNetV2 | Local spatial patterns in the 28×28 traffic image | Depthwise separable convolutions, ReLU6 |
| EfficientNet-B0 | Channel-wise feature relationships | Squeeze-and-Excitation attention, SiLU |

The two models examine the same traffic image through different mathematical lenses. Averaging their feature vectors (soft ensemble) reduces overconfidence.

---

## Quick Start

### Prerequisites

- Python 3.11.8
- NVIDIA GPU with CUDA 11.8 (tested on RTX 2050, 4GB VRAM)
- Windows 10/11 or Linux 

### 1. Install dependencies

```bash
# Create and activate virtual environment
python -m venv FYP_env
FYP_env\Scripts\activate          # Windows
# source FYP_env/bin/activate     # Linux/macOS

# Install PyTorch with CUDA 11.8 (must come first)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

### 2. Place the dataset

Download `Wednesday-workingHours.pcap_ISCX.csv` from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and place it (renamed) at both locations:

```
ExistingImplementation/data/raw/Wednesday-workingHours.csv
ProposedImplementation/data/raw/Wednesday-workingHours.csv
```

### 3. Run Phase 1 (baseline)

```bash
cd ExistingImplementation
python src/train.py --config configs/dos_config.yaml
```

Expected output: ~98.90% accuracy, 11.3M params, 0.1266 GMACs, ~22 min on RTX 2050.

### 4. Run Phase 2 (proposed)

```bash
cd ../ProposedImplementation
python src/train.py --config configs/dos_config.yaml
```

Expected output: ~99.15% accuracy, 3.2M params, 0.1138 GMACs, ~75 min on RTX 2050 (CTGAN adds ~40 min first run; use `--load-cache` on subsequent runs).

### 5. Generate comparison report

```bash
# From ProposedImplementation/
python src/evaluation/comparison_report.py \
    --existing-ckpt  ../ExistingImplementation/results/checkpoints/res_tranbilstm_dos_best.pth \
    --proposed-ckpt  results/checkpoints/ensemble_res_tranbilstm_dos_best.pth \
    --test-img       data/processed/X_test_img.npy \
    --test-seq       data/processed/X_test_seq.npy \
    --test-labels    data/processed/y_test.npy \
    --output         results/metrics/comparison_report.json
```

---

## Research Motivation

### Why the baseline already has a problem

Wang et al. (2023) designed Res-TranBiLSTM for **fog-node deployment** on IoT edge devices (smart routers, gateways). However, 99.1% of the model's parameters live in the ResNet-18 spatial branch alone — a standard-convolution backbone originally designed for 224×224 ImageNet images running on datacenter GPUs.

The authors acknowledged this in their conclusion: *"due to the instability of the data captured in the real network environment, the proposed model is only conducted in simulation experiments"* and called for *"more suitable methods for converting network traffic data to 2D images"* as future work.

### Why an ensemble of two lightweight models

Rather than simply swapping ResNet-18 for a single lighter model, using two architecturally distinct lightweight models gives three benefits:

1. **Parameter efficiency** — MobileNetV2 (544K) + EfficientNet-B0 (2.6M) combined = 3.2M, which is less than one ResNet-18 at 11.2M. Two models for the cost of one.

2. **Prediction reliability** — Neither model can be overconfident alone; the soft average balances them. This matters most for rare classes (Heartbleed: 11 real samples).

3. **Architectural diversity** — MobileNetV2 uses depthwise separable convolutions (local spatial patterns); EfficientNet-B0 adds Squeeze-and-Excitation channel attention (global channel relationships). They look at the same 28×28 traffic image through fundamentally different mathematical lenses.

### Why CTGAN replaces SMOTE-ENN

SMOTE generates synthetic samples by linear interpolation between real samples. For extremely rare classes (Heartbleed: 11 samples), SMOTE creates 15,000 nearly identical synthetic samples — essentially adding noise, not diversity. CTGAN trains a generator to learn the actual joint probability distribution of the traffic features, producing synthetic samples that span the real data manifold rather than a convex hull of 11 points.

---

## Implementation Notes

### 28×28 adaptation of MobileNetV2 and EfficientNet-B0

Standard versions of both models target 224×224 ImageNet input. A naive port to 28×28 would use a stride-2 stem convolution that immediately halves the spatial dimensions to 14×14 before any spatial features are learned. Both branches instead use **stride-1 stems** and defer downsampling to later stages, mirroring exactly what Wang et al. did for ResNet-18 (Table 6: `conv1: stride=1`, `conv2_x: maxpool stride=1`).

### No pretrained weights

Both spatial branches are trained from scratch. ImageNet pretrained weights are not used — traffic image pixels have no semantic relationship to natural image pixels, and the input is single-channel grayscale vs the 3-channel RGB that pretrained models expect.

### Fair comparison protocol

Phase 1 and Phase 2 use identical settings for all non-spatial, non-augmentation components:

- Same dataset (Wednesday CIC-IDS2017)
- Same 80/10/10 stratified split
- Same feature selection (SelectKBest, k=64)
- Same 28×28 bicubic upsampling
- Same training target (15,000 samples/class)
- Same batch size, learning rate, dropout, early stopping patience
- Same classification head (FC-4 → FC-5 → FC-6)

---

## Dependency Notes

### Critical version pins

| Package | Version | Reason |
|---|---|---|
| `numpy` | 1.26.4 | PyTorch 2.1.x compiled against NumPy 1.x ABI |
| `opencv-python` | 4.8.1.78 | OpenCV 4.9+ requires NumPy ≥ 2.0 |
| `scikit-learn` | 1.4.2 | imbalanced-learn 0.12.3 requires sklearn < 1.6 |
| `imbalanced-learn` | 0.12.3 | Must match scikit-learn pin |

### Windows DataLoader

`num_workers=0` is set throughout. Windows uses `spawn` multiprocessing which causes a bootstrapping error with any value > 0.

---

## References

Wang, Y. et al. (2023). *Res-TranBiLSTM: An Efficient Intrusion Detection Method for the Internet of Things*. Computers & Security / IEEE Access.

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP. [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.

Tan, M. & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.

Xu, L. et al. (2019). *Modeling Tabular Data using Conditional GAN*. NeurIPS.

Hu, J. et al. (2018). *Squeeze-and-Excitation Networks*. CVPR.