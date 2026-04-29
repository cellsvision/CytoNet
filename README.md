# CytoNet: Multi-Class Attention-based MIL Diagnostic Classification System

## Project Overview

CytoNet is a deep learning system for automated diagnostic classification of liquid-based cytology samples. The system employs a two-stage multi-class feature aggregation framework with attention-based Multiple Instance Learning (MIL), enabling patient-level (slide-level) classification and survival analysis prediction.

### Key Features

- **Self-Supervised Pre-training**: Label-free feature learning based on DINO
- **Supervised Fine-tuning**: Cell classification with RegNet-Y and CBAM attention
- **Multi-task MIL**: Supports both diagnostic classification and survival analysis
- **Multi-class Feature Aggregation**: Extract discriminative features grouped by cell type

---

## Project Structure

```
CytoNet/
├── dino/                          # Stage 1: Self-supervised Pre-training
│   ├── code/
│   │   ├── main_dino.py           # DINO training main program
│   │   ├── vision_transformer.py  # Vision Transformer implementation
│   │   ├── csv_dataloader.py      # Data loader
│   │   ├── utils.py               # Utility functions
│   │   └── run_train.sh           # Training launch script
│   ├── datalists/                 # Data lists
│   ├── models/                    # Model checkpoints
│   └── sample_data/               # Sample data
│
├── supervised_finetune/           # Stage 2: Supervised Fine-tuning
│   ├── code/
│   │   ├── config.py              # Training configuration
│   │   ├── train_test.py          # Training and testing script
│   │   ├── convert_to_torchvision.py  # Weight format conversion
│   │   └── extract_weights.py     # Weight extraction
│   ├── checkpoint/                # Pre-trained weights
│   ├── datalists/                 # Data lists
│   ├── models/                    # Fine-tuned models
│   └── sample_data/               # Cell sample data
│       ├── AUC/                   # AUC class samples
│       ├── HGUC/                  # HGUC class samples
│       ├── yin/                   # Normal samples (NILM/NHGUC)
│       ├── HISTIOCYTE/            # Histiocyte samples
│       ├── IMPURITY/              # Impurity samples
│       └── blank/                 # Blank region samples
│
├── multi_mission_share/           # Stage 3: Multi-task MIL
│   ├── code/
│   │   ├── train_test_multi.py    # Multi-task training script
│   │   ├── CoxPHLoss.py           # Cox proportional hazards loss
│   │   └── utils.py               # Survival analysis utilities
│   ├── datalists/                 # Data with survival information
│   ├── models/                    # MIL models
│   └── sample_data/               # Feature files
│
└── Method.docx                    # Detailed method description
```

---

## Technical Methodology

### 1. Input Data Organization

#### Stage 1: DINO Self-Supervised Pre-training

Uses over 100,000 unlabeled WSIs, cropped by size.

#### Stage 2: Supervised Fine-tuning (Urine Cytology)

- Each WSI is divided into multiple patches
- Based on annotations, corresponding labeled patches are cropped. For urine cytology-specific background images (blood/glial/urinary protein, histiocytes) and blank backgrounds, 6 categories of patches are prepared:
  - **NHGUC**: Normal cells (NILM/NHGUC)
  - **AUC**: Atypical Urothelial Cells
  - **HGUC**: High-grade Urothelial Carcinoma
  - **HISTIOCYTE**: Histiocytes
  - **IMPURITY**: Impurities
  - **BLANK**: Blank regions

#### Stage 3:

- The raw input originates from cell classification results on whole slide urine cytology images (WSI):
  - Under 40x magnification, patches matching the patch size are inferred for their class and feature map. For each class, the top-N features by confidence are sorted in descending order, concatenated, and aggregated into a Tensor. For example, if an image can be cropped into 50 × 60 patches, each patch is fed into the model to output a patch-level classification result and extract a feature map. The top 300 features per class (by confidence, descending) are concatenated into a [6×300, featuremap_length] array and saved as a pkl file.
- During data input, the top-200 most representative patches per class are selected.

### 2. Stage 1: DINO Self-Supervised Pre-training

#### Model Architecture
- Supports Vision Transformer (ViT-Tiny/Small/Base) and RegNet-Y-800MF
- Teacher-student network structure with EMA weight updates
- DINO projection head: 3-layer MLP + L2 normalization

#### Data Augmentation Strategy
- **Global Crops** (512×512): Scale range 0.4 ~ 1.0
- **Local Crops** (224×224): Scale range 0.05 ~ 0.4, 8 crops total
- Includes: random flip, color jitter, Gaussian blur, solarization

#### Training Parameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0005 |
| Weight Decay | 0.04 → 0.4 (cosine schedule) |
| Batch Size | 4/GPU |
| Optimizer | AdamW |
| Output Dimension | 65536 |

### 3. Stage 2: Supervised Fine-tuning

#### Model Structure
- **Backbone**: RegNet-Y-800MF (loaded with DINO pre-trained weights)
- **Attention Module**: CBAM (Convolutional Block Attention Module)
- **Classification Head**: 6-class cell classification

#### Classification Categories
```python
train_patch_class_id = {
    'nilm': 0,     # Normal (NILM/NHGUC)
    'auc': 1,      # Atypical Urothelial Cells
    'hguc': 2,     # High-grade Urothelial Carcinoma
    'impurity': 3, # Impurity
    'histiocyte': 4, # Histiocyte
    'blank': 5     # Blank
}
```

#### Data Augmentation (Albumentations)
- JPEG compression
- Random Gamma / CLAHE / Brightness-Contrast adjustment
- Flip
- Hue-Saturation-Value / RGB shift
- ISO noise / Gaussian noise

#### Training Strategy
- Label smoothing (0.95 confidence)
- Per-slide sampling balance
- CosineAnnealingWarmRestarts learning rate schedule

### 4. Stage 3: Multi-task MIL Model

#### Model Architecture

**Instance-level Feature Encoder**:
```
h_i^c = MLP_c(x_i^c)  # Class-specific fully connected layers
```

**Attention Weight Module**:
```
A_c = w_c · (tanh(V_c · h) ⊙ sigmoid(U_c · h))
```
- V_c: Tanh branch, captures feature saliency
- U_c: Sigmoid branch, learns gating mechanism
- Intermediate dimension D = 256

**Attention-weighted Aggregation**:
```
z_c = Σ_i softmax(A_c,i) · h_i^c
```

**Multi-task Heads**:
- **Classification Head**: Outputs 2-class diagnosis (Negative/Positive)
- **Survival Analysis Head**: Outputs single-dimensional risk score

#### Loss Functions

**Multi-task Joint Loss**:
```
L_total = α · L_cls + β · L_surv
```

**Classification Loss**: Cross-Entropy Loss
```
L_cls = -Σ y_k log(p_k)
```

**Survival Analysis Loss**: Cox Proportional Hazards Loss
```
L_surv = -Σ_{i:E_i=1} (log_h_i - log(Σ_{j∈R_i} exp(log_h_j)))
```

#### Evaluation Metrics
- **Classification**: AUC, Sensitivity, Specificity, Kappa coefficient
- **Survival Analysis**: C-Index, Time-dependent AUC (12/24/36/48/60 months)

---

## Quick Start

### Environment Setup

```bash
# Python 3.7+
# PyTorch 1.9+
pip3 install torch torchvision
pip3 install timm albumentations pandas numpy scikit-learn
pip3 install lifelines tensorboard torch-optimizer
pip3 install easydict Pillow opencv-python
```

### Stage 1: DINO Pre-training

```bash
cd dino/code

# Train with 4 GPUs (example)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.run --nproc_per_node=4 main_dino.py \
    --arch regnet_y_800mf \
    --valid_data_pkl ../datalists/sample_cells_valid_img_path.pkl \
    --output_dir ../models \
    --batch_size_per_gpu 4 \
    --saveckp_freq 1
```

### Stage 2: Weight Conversion and Fine-tuning

```bash
cd supervised_finetune/code

# Step 1: Convert DINO weights to torchvision format
python3 convert_to_torchvision.py

# Step 2: Train classification model
python3 train_test.py
```

### Stage 3: Multi-task MIL Training

```bash
cd multi_mission_share/code

# Train multi-task model (classification + survival analysis)
python3 train_test_multi.py
```

---

## Core Code Description

### DINO Training ([`dino/code/main_dino.py`](dino/code/main_dino.py))

- `DINOLoss`: Knowledge distillation loss, cross-entropy between teacher-student networks
- `DataAugmentationDINO`: Multi-crop data augmentation strategy
- `train_dino()`: Main training loop, supports distributed training

### Supervised Fine-tuning ([`supervised_finetune/code/train_test.py`](supervised_finetune/code/train_test.py))

- `UrineDataset`: Cell image dataset, supports per-epoch resampling
- `UrineModel`: RegNet-Y + CBAM classification model
- `evaluate_model()`: Computes multi-class and binary accuracy

### Multi-task MIL ([`multi_mission_share/code/train_test_multi.py`](multi_mission_share/code/train_test_multi.py))

- `MILAttModule`: Attention module with Tanh-Sigmoid gating mechanism
- `MultiTaskMILModel`: Multi-task MIL model with classification + survival analysis heads
- `MultiTaskMILDataset`: Loads feature files and survival information

### Cox Loss ([`multi_mission_share/code/CoxPHLoss.py`](multi_mission_share/code/CoxPHLoss.py))

- `CoxPHLoss`: Cox proportional hazards loss function
- `coxPHLoss()`: Batched Cox loss computation
- `make_riskset()`: Constructs risk set matrix

---

## Data Formats

### DINO Training Data
- **Input**: Pickle file containing image path list
- **Format**: `['/path/to/image1.png', '/path/to/image2.png', ...]`

### Supervised Fine-tuning Data
- **CSV format**:
```
filename
WSI1
WSI2
...
```
- **Directory structure**:
```
sample_data/
├── AUC/WSI4/AUC/*.png
├── HGUC/WSI6/HGUC/*.png
├── yin/WSI7/NILM/*.png
...
```

### MIL Training Data
- **CSV format** (with survival information):
```
name,GT,dfs_status,dfs_time
patient_001,Pos,1,24
patient_002,Neg,0,60
...
```
- **Feature files**: `{name}.pkl`, shape `(1200, 768)`

---

## Model Checkpoints

### DINO Checkpoint Contents
- `student`: Student network weights
- `teacher`: Teacher network weights (recommended for downstream tasks)
- `optimizer`: Optimizer state
- `epoch`: Training epoch number

### Fine-tuned Models
- `ckpt_{epoch}_{acc}_{acc_2}.pth`: Contains classification accuracy metrics

### MIL Models
- Multi-task model weights supporting both classification and survival analysis

---

## Applications

1. **Urine Cytology Lymph Node Metastasis Diagnosis**
   - Automated distinction between negative and positive cases
   - Assists pathologists in diagnosis

2. **Patient Survival Prediction**
   - Predict disease-free survival (DFS)
   - Risk stratification

3. **Cell Image Feature Extraction**
   - General features learned through self-supervised learning
   - Transferable to other medical image tasks

---

## References

- Caron M, et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
- Ilse M, et al. "Attention-based Deep Multiple Instance Learning." ICML 2018.
- Kvamme H, et al. "Time-to-Event Prediction with Neural Networks and Cox Regression."

---

## License

This project is based on Apache License 2.0, with partial code derived from Facebook Research DINO project.

---

## Contact

For questions or suggestions, please contact via the project Issue page.

---

*Note: README and comments were polished with AI assistance.*