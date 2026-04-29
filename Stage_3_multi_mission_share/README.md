# Multi-Task MIL Model for Classification and Survival Analysis

## Overview

This project implements a **Multi-Task Multiple Instance Learning (MIL)** model that simultaneously performs:
- **Classification**: Binary classification (Neg/Pos) for histopathology slides
- **Survival Analysis**: Disease-free survival (DFS) prediction using Cox Proportional Hazards loss

## Project Structure

```
multi_mission_share/
├── code/                          # Source code
│   ├── train_test_multi.py        # Main training and evaluation script
│   ├── CoxPHLoss.py               # Cox Proportional Hazards loss implementation
│   └── utils.py                   # Utility functions (C-index, risk set, etc.)
├── datalists/                     # Data configuration
│   ├── train.csv                  # Training data list with survival info
│   └── val.csv                    # Validation data list
├── sample_data/                   # Sample feature files (.pkl format)
├── models/                        # Saved model checkpoints
├── logs/                          # Training logs
└── results/                       # Evaluation results
```

## Data Format

### CSV Columns
| Column | Description |
|--------|-------------|
| `name` | Sample identifier (matches .pkl filename) |
| `GT` | Ground truth label (Neg/Pos) |
| `dfs_status` | Disease-free survival status (0/1) |
| `dfs_time` | Disease-free survival time (months) |

### Feature Files
- Format: `.pkl` (pickle) files
- Each file contains the feature maps inferred from patches cropped at 40x magnification according to patch size. For each class, the top N features are sorted by confidence in descending order, concatenated, and aggregated into a Tensor. For example, if an image can be cropped into 50 × 60 patches, each patch is input into the model to produce patch-level classification results and feature maps. The top 300 features per class (sorted by confidence in descending order) are concatenated into a [6×300, featuremap_length] array and saved as a .pkl file. During usage, the first 200 feature maps are extracted from each group of 300.

```bash
cls_bag_size = 300
top_cls_bag_size = 300
use_cls_bag_size = 200
```

## Training

```bash
cd code
python train_test_multi.py
```

### Key Parameters
- `fea_size`: Feature dimension
- `batch_size`: Training batch size
- `max_epoch`: Maximum training epochs
- `max_patches`: Maximum patches per class
- `alpha`: Classification loss weight
- `beta`: Survival loss weight

## Evaluation Metrics

- **Classification**: AUC, Sensitivity, Specificity
- **Survival**: C-Index, Time-dependent AUC (12, 24, 36, 48, 60 months)

## Dependencies

Refer to requirements.txt

## Sample Data

The `sample_data/` directory contains 24 sample feature files (1.pkl - 24.pkl) for demonstration purposes.

## Logs

Training logs are saved to `logs`

---

*Note: README and comments were polished with AI assistance.*