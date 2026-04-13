# DINO Self-Supervised Learning Project

## Project Overview

This project implements self-supervised feature learning for cell images based on Facebook Research's DINO (Self-Distillation with No Labels) method. DINO is a label-free self-supervised learning approach that trains vision models through knowledge distillation, particularly suitable for medical image analysis.

## Directory Structure

```
dino/
├── code/                    # Core code directory
│   ├── run_train.sh         # Training launch script
│   ├── main_dino.py         # DINO training main program
│   ├── vision_transformer.py # Vision Transformer model implementation
│   ├── csv_dataloader.py    # Data loader
│   └── utils.py             # Utility functions
├── datalists/               # Data lists directory
│   └── sample_cells_valid_img_path.pkl  # Sample image path list
├── logs/                    # Logs directory
│   └── log_DINO.txt         # Training logs
├── models/                  # Model checkpoint directory
│   ├── checkpoint.pth       # Latest checkpoint
│   ├── checkpoint0000.pth   # Epoch 0 checkpoint
│   ├── checkpoint0001.pth   # Epoch 1 checkpoint
│   └── log.txt              # Training records
├── results/                 # Results output directory
├── sample_data/             # Sample data directory (128 cell images)
└── README_EN.md             # English documentation
```

## Core Module Description

### 1. main_dino.py - DINO Training Main Program

Main features:
- **Architecture Support**: Vision Transformer (ViT-Tiny/Small/Base) and RegNet architectures
- **Multi-crop Training Strategy**: Generates 2 global crops and 8 local crops for enhanced feature learning
- **Teacher-Student Network**: Teacher network updated via EMA (Exponential Moving Average)
- **DINO Loss Function**: Cross-entropy based knowledge distillation loss
- **Mixed Precision Training**: FP16 training support for improved efficiency
- **Distributed Training**: Multi-GPU distributed training support

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--arch` | vit_small | Model architecture |
| `--epochs` | 2 | Training epochs |
| `--batch_size_per_gpu` | 96 | Batch size per GPU |
| `--lr` | 0.0005 | Learning rate |
| `--out_dim` | 65536 | DINO head output dimension |
| `--local_crops_number` | 8 | Number of local crops |
| `--global_crops_scale` | (0.4, 1.) | Global crop scale range |
| `--local_crops_scale` | (0.05, 0.4) | Local crop scale range |

### 2. vision_transformer.py - Vision Transformer Implementation

Components:
- **VisionTransformer**: Complete ViT model with position encoding interpolation support
- **PatchEmbed**: Image to patch embedding conversion
- **Attention**: Multi-head self-attention mechanism
- **Block**: Transformer block with attention layer and MLP
- **DINOHead**: DINO projection head, 3-layer MLP + L2 normalization + weight normalized linear layer

Supported model variants:
- `vit_tiny`: embed_dim=192, depth=12, heads=3
- `vit_small`: embed_dim=384, depth=12, heads=6
- `vit_base`: embed_dim=768, depth=12, heads=12

### 3. csv_dataloader.py - Data Loader

Three dataset classes provided:
- **CSVDataset**: Load images from directory list with random cropping support
- **CSVDataset_v2**: Multi-source sampling with ratio support
- **CSVDataset_v3**: Load image path list from pickle file (used in this project)

Image preprocessing:
- Supports PNG, JPEG, JPG, TIF, TIFF formats
- Automatic BGR to RGB conversion (using cv2)

### 4. utils.py - Utility Functions

Main features:
- **Distributed Training**: `init_distributed_mode()`, `get_world_size()`, `get_rank()`
- **Data Augmentation**: `GaussianBlur`, `Solarization`
- **Optimizer**: `LARS` optimizer implementation
- **Learning Rate Schedule**: `cosine_scheduler()` cosine annealing schedule
- **Model Wrapper**: `MultiCropWrapper` handles multi-resolution inputs
- **Logging**: `MetricLogger`, `SmoothedValue`
- **Checkpoint Management**: `restart_from_checkpoint()`, `save_on_master()`

## Data Augmentation Strategy

DINO employs a multi-crop data augmentation strategy:

1. **Global Crops** (512×512):
   - Scale range: 0.4 ~ 1.0
   - Includes: random flip, color jitter, Gaussian blur, solarization

2. **Local Crops** (224×224):
   - Scale range: 0.05 ~ 0.4
   - Quantity: 8 crops
   - Includes: random flip, color jitter, Gaussian blur

Normalization parameters (optimized for cell images):
- Mean: (0.7896, 0.8113, 0.8456)
- Std: (0.1599, 0.1378, 0.1046)

## Quick Start

### Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- CUDA 11.0+ (recommended)
- OpenCV (cv2)
- PIL (Pillow)
- NumPy

### Training Command

```bash
cd code

# Train with 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.run --nproc_per_node=4 main_dino.py \
    --arch regnet_y_800mf \
    --valid_data_pkl ../datalists/sample_cells_valid_img_path.pkl \
    --output_dir ../models \
    --batch_size_per_gpu 4 \
    --saveckp_freq 1
```

### Training with ViT Model

```bash
python3 -m torch.distributed.run --nproc_per_node=4 main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --valid_data_pkl ../datalists/sample_cells_valid_img_path.pkl \
    --output_dir ../models \
    --batch_size_per_gpu 4 \
    --epochs 100 \
    --saveckp_freq 10
```

## Training Log Example

```
| distributed init (rank 0): env://
arch: regnet_y_800mf
batch_size_per_gpu: 4
epochs: 2
Data loaded: there are 128 images.
Student and Teacher are built: they are both regnet_y_800mf network.
Starting DINO training !
Epoch: [0/2]  loss: 10.404975  lr: 0.000000  wd: 0.040000
Epoch: [1/2]  loss: 10.856192  lr: 0.000031  wd: 0.220000
Training time 0:00:17
```

## Model Checkpoints

Checkpoints saved during training contain:
- `student`: Student network state
- `teacher`: Teacher network state
- `optimizer`: Optimizer state
- `epoch`: Current epoch number
- `args`: Training arguments
- `dino_loss`: Loss function state
- `fp16_scaler`: Mixed precision scaler state (if enabled)

## Applications

This project is particularly suitable for:
- Cell image feature extraction
- Medical image unsupervised pre-training
- Downstream tasks: classification, detection, segmentation

## References

- Caron M, Touvron H, Misra I, et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
- DINO GitHub: https://github.com/facebookresearch/dino

## License

This project is based on Apache License 2.0, with code derived from Facebook Research DINO project.

---

*Note: README and comments were polished with AI assistance.*