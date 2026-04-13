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

## Model Architecture

- **Backbone**: Attention-based MIL with shared dense layers
- **Classification Head**: Multi-layer perceptron for binary classification
- **Survival Head**: Regression head for risk score prediction
- **Loss Function**: Combined Cross-Entropy + Cox PH Loss

## Data Format

### CSV Columns
| Column | Description |
|--------|-------------|
| `name` | Sample identifier (matches .pkl filename) |
| `GT` | Ground truth label (Neg/Pos) |
| `dfs_status` | Disease-free survival status (0/1) |
| `dfs_time` | Disease-free survival time (months) |

### Feature Files
- Format: `.pkl` (pickle) files containing 768-dimensional features
- Each file contains features for multiple patches organized by class type

## Training

```bash
cd code
python train_test_multi.py
```

### Key Parameters
- `fea_size`: Feature dimension (768)
- `batch_size`: Training batch size (4)
- `max_epoch`: Maximum training epochs (100)
- `max_patches`: Maximum patches per class (200)
- `alpha`: Classification loss weight (1.0)
- `beta`: Survival loss weight (1.0)

## Evaluation Metrics

- **Classification**: AUC, Sensitivity, Specificity
- **Survival**: C-Index, Time-dependent AUC (12, 24, 36, 60 months)

## Dependencies

- PyTorch
- scikit-learn
- lifelines
- pandas
- numpy
- tensorboard

## Sample Data

The `sample_data/` directory contains 24 sample feature files (1.pkl - 24.pkl) for demonstration purposes.

## Logs

Training logs are saved to `logs/lymph_survival_train.log` with:
- Per-batch loss values
- Per-epoch C-Index and AUC metrics
- Validation results

---

*Note: README and comments were polished with AI assistance.*