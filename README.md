# IDL_CAD_HGCN

**Hierarchical GCN/GraphSAGE for Machining Feature Classification on MFCAD++**

CMU 24-788 — Final Project

---

## Overview

This repository contains training, evaluation, and inference code for a hierarchical graph neural network that classifies machining features on CAD B-rep models. We train two architectures (GCN baseline and GraphSAGE variant) across three feature configurations on the MFCAD++ dataset (59,665 labeled parts, 25 feature classes).

| Experiment | Acc | Macro F1 | wandb |
|---|---|---|---|
| GCN V1 flat | 0.4432 | 0.3375 | fw8zrpy4 |
| GCN V1+V2 hierarchical | 0.3628 | 0.3055 | 89g2g8my |
| GCN UV-net | 0.5383 | 0.4692 | pkgqwb7b |
| GraphSAGE V1 flat | 0.6380 | 0.5453 | 024839ck |
| GraphSAGE V1+V2 hierarchical | 0.7937 | 0.7273 | mfl4y8ti |
| **GraphSAGE UV-net (best)** | **0.8501** | **0.7852** | **yfm13ito** |

All six trained model checkpoints are included in `checkpoints/`.

---

## Repository Structure

```
IDL_CAD_HGCN/
├── train_mfcad.py          Main training script
├── extract_uv_features.py  UV-net feature pre-extraction
├── eval_models.py          Quick per-model evaluation utility
├── reproduce_results.py    Loads checkpoints, regenerates paper figures
├── backfill_wandb.py       Utility to log historical runs to wandb
├── model_config.yaml       Model hyperparameters
├── train_config.yaml       Training hyperparameters
├── requirements.txt        Python dependencies
└── checkpoints/
    ├── label_map.json
    ├── model_config.json
    ├── gcn_v1_flat/
    ├── gcn_v1v2_hierarchical/
    ├── gcn_uvnet/
    ├── graphsage_v1_flat/
    ├── graphsage_v1v2_hierarchical/
    └── graphsage_uvnet/
```

---

## Environment Setup

**Python 3.10+ recommended.**

```bash
pip install -r requirements.txt
```

PyTorch Geometric has extra dependencies that must be installed separately:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

Replace `cu121` with your CUDA version (or `cpu` for CPU-only).

---

## Dataset

Download the **MFCAD++** dataset from the official source:

> Colligan et al., *"Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition"*, CAD 2022.
> Dataset: [https://github.com/NCIS-SGLAB/MFCAD](https://github.com/NCIS-SGLAB/MFCAD)

The dataset is a zip file containing three HDF5 files (`train.h5`, `val.h5`, `test.h5`). Either keep it as a zip or extract it — both are supported.

---

## Reproducing Key Results

### Option A — V1 and V1+V2 runs only (no UV cache needed)

```bash
python reproduce_results.py \
    --data_dir /path/to/MFCAD++_dataset \
    --no_uvnet \
    --out_dir paper_figures
```

This evaluates the four non-UV-net checkpoints and prints Table 1 rows.

### Option B — All six runs including UV-net (recommended)

**Step 1**: Pre-extract UV-net features from the dataset. This takes ~30–60 minutes and requires ~2 GB disk space.

```bash
python extract_uv_features.py \
    --data_dir /path/to/MFCAD++_dataset \
    --out_dir ./uv_cache
```

**Step 2**: Run the full reproduce script.

```bash
python reproduce_results.py \
    --data_dir /path/to/MFCAD++_dataset \
    --uv_cache_dir ./uv_cache \
    --out_dir paper_figures
```

**Outputs** in `paper_figures/`:
- `table1.csv` — all six runs: acc, macro-F1, precision, recall
- `confusion_matrix.png` — normalized 25×25 confusion matrix (GraphSAGE UV-net)
- `per_class_f1.png` — per-class F1 bar chart sorted ascending (GraphSAGE UV-net)
- `classification_report.txt` — full sklearn classification report (GraphSAGE UV-net)

---

## Training from Scratch

### V1 flat or V1+V2 hierarchical

```bash
python train_mfcad.py \
    --data_dir /path/to/MFCAD++_dataset \
    --model both \
    --epochs 100
```

### UV-net (requires UV cache)

```bash
# Step 1: extract UV features (one-time, ~30-60 min)
python extract_uv_features.py \
    --data_dir /path/to/MFCAD++_dataset \
    --out_dir ./uv_cache

# Step 2: train with UV features
python train_mfcad.py \
    --data_dir /path/to/MFCAD++_dataset \
    --uv_cache_dir ./uv_cache \
    --model both \
    --epochs 100
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `both` | `gcn`, `graphsage`, or `both` |
| `--epochs` | `100` | Max training epochs |
| `--patience` | `15` | Early stopping patience |
| `--no_wandb` | off | Disable wandb logging |
| `--uv_cache_dir` | None | Enable UV-net features |

---

## wandb Project

All runs are logged to wandb project `mfcad-feature-classification`. View the full experiment comparison at the project page. Each run is tagged with `arch:<x>`, `feature_set:<x>`, `weighted_loss:<x>`, and `live` or `backfilled`.

---

## Model Architecture

Both models use a **hierarchical B-AAG graph** with two node types:
- **V₁** (face-level): surface type, area, normal, centroid — 5D flat or 12D with UV-net
- **V₂** (facet-level): 4D geometric features
- **A₃** (cross-level edges): V₁↔V₂ adjacency

**GCN baseline**: `GCNConv × 3`, transductive, degree-normalised aggregation.  
**GraphSAGE variant**: `SAGEConv × 3`, inductive, sampled mean aggregation. Self-features are concatenated before aggregation, preserving per-face identity.

**UV-net features**: Each face is sampled on a 5×5 B-Rep UV grid. Mean normal (3D), normal std (3D), and UV coverage (1D) are computed per face and concatenated to V₁, yielding 12D face features.
