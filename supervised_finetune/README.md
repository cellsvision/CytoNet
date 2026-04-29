# Supervised Fine-tuning for Urine Cytology Classification

## Project Overview

This project is a supervised fine-tuning system for urine cytology image classification based on deep learning. The system uses a pre-trained RegNet-Y-800MF model as the backbone network, combined with CBAM attention mechanism, to perform multi-class classification of urine cytology images.

## Classification Categories

The project supports classification of the following 6 categories:

| Class Code | Class Name | Description |
|------------|------------|-------------|
| NHGUC | Negative | Normal urine cells (NILM/NHGUC) |
| AUC | Atypical Urothelial Cells | Atypical urothelial cells |
| HGUC | High-Grade Urothelial Carcinoma | High-grade urothelial carcinoma |
| IMPURITY | Impurity | Impurity regions in images |
| HISTIOCYTE | Histiocyte | Histiocyte cells |
| BLANK | Blank | Blank background regions |

## Directory Structure

```
supervised_finetune/
├── checkpoint/          # Pre-trained weights storage
│   └── teacher_backbone_torchvision.pth  # Teacher backbone weights
├── code/                # Source code directory
│   ├── config.py        # Training configuration
│   ├── train_test.py    # Main training and testing script
│   ├── extract_weights.py    # Extract weights from DINO checkpoint
│   └── convert_to_torchvision.py  # Weight format conversion
├── datalists/           # Data lists directory
│   ├── train.csv        # Training set file list
│   └── val.csv          # Validation set file list
├── logs/                # Training logs directory
├── models/              # Model save directory
├── results/             # Results output directory
└── sample_data/         # Sample data directory
    ├── AUC/             # AUC class samples
    ├── blank/           # Blank region samples
    ├── HGUC/            # HGUC class samples
    ├── histiocyte/      # Histiocyte samples
    ├── impurity/        # Impurity samples
    └── yin/             # NILM/NHGUC class samples
```

*Note: Annotators may habitually use NILM or NHGUC to label negative cases. This is merely a difference in notation; both truly represent normal urothelial cells.*

## Model Architecture

- **Backbone Network**: RegNet-Y-800MF
- **Attention Mechanism**: CBAM (Convolutional Block Attention Module)
- **Classification Head**: 6-class classification output

## Dependencies

Refer to requirements.txt

## Usage

### 1. Data Preparation

Place urine cytology images in the `sample_data/` directory by category. Each WSI (Whole Slide Image) patch should be stored in corresponding subdirectories.

Data directory structure example:
```
sample_data/
├── AUC/
│   └── WSI5/
│       └── AUC/
│           └── *.png
├── HGUC/
│   └── WSI6/
│       └── HGUC/
│           └── *.png
├── impurity/
│   └── WSI1/
│       └── IMPURITY/
│           └── *.png
└── ...
```

### 2. Configuration

Modify training configuration in [`code/config.py`](code/config.py):

```python
train_cfg.GPU_ID = "0, 1, 2, 3"  # GPU device IDs
train_cfg.batch_size = 4          # Batch size
train_cfg.max_epoch = 30           # Training epochs
train_cfg.target_size = 1024     # Input image size
train_cfg.checkpoint_path = '../checkpoint/teacher_backbone_torchvision.pth'  # Pre-trained weights path
```

### 3. Start Training

```bash
cd code
python train_test.py
```

### 4. Weight Extraction and Conversion

To extract weights from DINO checkpoint:

```bash
# Extract weights
python extract_weights.py

# Convert to torchvision format
python convert_to_torchvision.py
```

## Training Strategy

### Optimizer Settings
- **Optimizer**: Adam
- **Learning Rate**: 5e-6
- **Learning Rate Scheduler**: CosineAnnealingWarmRestarts
  - T_0 = 3
  - T_mult = 2
  - eta_min = 1e-7

### Loss Function
- CrossEntropyLoss

### Data Augmentation Strategy
The training process uses the albumentations library for data augmentation:

- JPEG compression (quality 70-95)
- Random Gamma correction (gamma_limit: 50-150)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Random brightness and contrast adjustment
- Random flip
- Hue, saturation, value adjustment
- RGB shift
- Gaussian noise
- ISO noise
- Multiplicative noise

## Evaluation Metrics

- **Accuracy**: 6-class classification accuracy
- **Binary Accuracy (acc_2)**: Accuracy after merging classes into negative/positive
  - Negative classes: NILM/NHGUC, IMPURITY, HISTIOCYTE, BLANK
  - Positive classes: AUC, HGUC

## Code Files Description

### [`config.py`](code/config.py)
Training configuration file, including:
- GPU device configuration
- Data path configuration
- Class mapping configuration
- Training hyperparameters

### [`train_test.py`](code/train_test.py)
Main training script, including:
- `UrineDataset`: Urine image dataset class
- `UrineModel`: Classification model class
- `load_model()`: Load pre-trained model
- `load_trained_model()`: Load trained model
- Training and validation loops

### [`extract_weights.py`](code/extract_weights.py)
Extract weights from DINO checkpoint, generating:
- student_backbone.pth: Student model backbone weights
- teacher_backbone.pth: Teacher model backbone weights
- student_full.pth: Student model full weights
- teacher_full.pth: Teacher model full weights

### [`convert_to_torchvision.py`](code/convert_to_torchvision.py)
Convert DINO weight format to torchvision's RegNet-Y-800MF format.

## Notes

1. **Class Balancing**: Training data uses class balancing strategy with limited sampling per class from each WSI
2. **Multi-GPU Training**: Model supports multi-GPU parallel training (DataParallel)
3. **Logging**: Training process uses TensorBoard to record loss and accuracy curves
4. **Label Smoothing**: Uses label smoothing strategy, positive class weight 0.95, other classes weight 0.05/(n-1)

## Training Log Example

```
INFO:root:Start epoch 0
INFO:root:epoch 0 Batch: 0, loss = 1.617, avg_batch_time = 19.350
INFO:root:epoch 0 Batch: 1, loss = 1.705, avg_batch_time = 9.806
...
INFO:root:=========================== acc 0.34782608695652173 acc_2 0.5072463768115942 =================
INFO:root:Start epoch 1
...
INFO:root:=========================== acc 0.7391304347826086 acc_2 0.8260869565217391 =================
```

## License

This project is for research purposes only.

## Contact

For questions or issues, please contact the project maintainers.

---

*Note: README and comments were polished with AI assistance.*