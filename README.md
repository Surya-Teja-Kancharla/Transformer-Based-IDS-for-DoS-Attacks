# FYP — Transformer-Based Intrusion Detection System

## Project Structure

```
FYP/
├── venv/                          # Shared Python virtual environment
├── requirements.txt               # Shared dependencies
├── README.md                      # This file
│
├── ExistingImplementation/        # Phase 1: Replicate Res-TranBiLSTM (Wang et al., 2023)
│   ├── data/
│   │   ├── raw/                   # Original CIC-IDS2017 CSVs
│   │   ├── processed/             # After cleaning, encoding, normalization
│   │   └── augmented/             # After SMOTE-ENN balancing
│   ├── src/
│   │   ├── preprocessing/         # Data pipeline modules
│   │   ├── models/
│   │   │   ├── spatial/           # ResNet spatial extractor
│   │   │   ├── temporal/          # TranBiLSTM temporal extractor
│   │   │   ├── classification/    # FC classifier head
│   │   │   └── res_tranbilstm.py  # Full assembled model
│   │   ├── training/              # Trainer, optimizer, scheduler, early stopping
│   │   ├── evaluation/            # Metrics, confusion matrix, FLOPs counter
│   │   └── utils/                 # Logger, seed, checkpointer, profiler
│   ├── configs/                   # YAML config files
│   ├── notebooks/                 # EDA and result analysis notebooks
│   ├── logs/                      # Training logs (JSON + TensorBoard)
│   └── results/
│       ├── metrics/               # Saved metric JSON/CSV files
│       ├── plots/                 # Generated figures
│       └── checkpoints/           # Model .pth files
│
└── ProposedImplementation/        # Phase 2: Lightweight proposed model
    ├── data/                      # (same structure as above)
    ├── src/
    │   ├── preprocessing/         # Same pipeline (separate copy)
    │   ├── models/
    │   │   ├── spatial/           # Lightweight spatial module
    │   │   ├── temporal/          # Efficient attention + lightweight temporal
    │   │   ├── classification/    # Shared classifier head
    │   │   └── proposed_model.py  # Full assembled proposed model
    │   ├── training/
    │   ├── evaluation/
    │   │   └── comparison_report.py  # FLOPs/param comparison vs Existing
    │   └── utils/
    ├── configs/
    ├── notebooks/
    ├── logs/
    └── results/
```

## Setup

```bash
cd FYP
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install --upgrade pip
# PyTorch with CUDA 11.8 (RTX 2050):
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118
# Remaining dependencies:
pip install -r requirements.txt
```

## Dataset

Download CIC-IDS2017 from:
https://www.unb.ca/cic/datasets/ids-2017.html

Place the CSV files inside:
- `ExistingImplementation/data/raw/`
- `ProposedImplementation/data/raw/`  (or symlink)

## Running

```bash
# Phase 1 — Existing model
cd ExistingImplementation
python run_pipeline.py --config configs/cicids2017_config.yaml

# Phase 2 — Proposed model + comparison
cd ProposedImplementation
python run_pipeline.py --config configs/cicids2017_config.yaml
python compare_models.py
```

## Comparison Metrics (FLOPs & Parameters)

Both implementations log FLOPs (measured at batch=1 inference) and
parameter counts to `logs/model_profile.json` via `thop` and `torchinfo`.
The ProposedImplementation's `compare_models.py` produces a side-by-side
report saved to `results/metrics/comparison_report.json`.