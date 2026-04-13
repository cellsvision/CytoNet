# DINO 自监督学习项目

## 项目简介

本项目基于 Facebook Research 的 DINO (Self-Distillation with No Labels) 方法，实现用于细胞图像的自监督特征学习。DINO 是一种无需标签的自监督学习方法，通过知识蒸馏的方式训练视觉模型，特别适用于医学图像分析领域。

## 目录结构

```
dino/
├── code/                    # 核心代码目录
│   ├── run_train.sh         # 训练启动脚本
│   ├── main_dino.py         # DINO训练主程序
│   ├── vision_transformer.py # Vision Transformer模型实现
│   ├── csv_dataloader.py    # 数据加载器
│   └── utils.py             # 工具函数
├── datalists/               # 数据列表目录
│   └── sample_cells_valid_img_path.pkl  # 样本图像路径列表
├── logs/                    # 日志目录
│   └── log_DINO.txt         # 训练日志
├── models/                  # 模型保存目录
│   ├── checkpoint.pth       # 最新检查点
│   ├── checkpoint0000.pth   # Epoch 0 检查点
│   ├── checkpoint0001.pth   # Epoch 1 检查点
│   └── log.txt              # 训练记录
├── results/                 # 结果输出目录
├── sample_data/             # 样本数据目录（128张细胞图像）
└── README_CN.md             # 中文说明文档
```

## 核心模块说明

### 1. main_dino.py - DINO训练主程序

主要功能：
- **模型架构支持**：支持 Vision Transformer (ViT-Tiny/Small/Base) 和 RegNet 等多种架构
- **多裁剪训练策略**：生成2个全局裁剪和8个局部裁剪，增强特征学习
- **教师-学生网络**：通过 EMA (Exponential Moving Average) 更新教师网络
- **DINO损失函数**：基于交叉熵的知识蒸馏损失
- **混合精度训练**：支持 FP16 训练，提升效率
- **分布式训练**：支持多GPU分布式训练

关键参数：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | vit_small | 模型架构 |
| `--epochs` | 2 | 训练轮数 |
| `--batch_size_per_gpu` | 96 | 每GPU批次大小 |
| `--lr` | 0.0005 | 学习率 |
| `--out_dim` | 65536 | DINO头输出维度 |
| `--local_crops_number` | 8 | 局部裁剪数量 |
| `--global_crops_scale` | (0.4, 1.) | 全局裁剪尺度范围 |
| `--local_crops_scale` | (0.05, 0.4) | 局部裁剪尺度范围 |

### 2. vision_transformer.py - Vision Transformer实现

包含组件：
- **VisionTransformer**：完整的ViT模型，支持位置编码插值
- **PatchEmbed**：图像到patch嵌入的转换
- **Attention**：多头自注意力机制
- **Block**：Transformer块，包含注意力层和MLP
- **DINOHead**：DINO投影头，3层MLP + L2归一化 + 权重归一化线性层

支持的模型变体：
- `vit_tiny`: embed_dim=192, depth=12, heads=3
- `vit_small`: embed_dim=384, depth=12, heads=6
- `vit_base`: embed_dim=768, depth=12, heads=12

### 3. csv_dataloader.py - 数据加载器

提供三个数据集类：
- **CSVDataset**：从目录列表加载图像，支持随机裁剪
- **CSVDataset_v2**：支持多数据源按比例采样
- **CSVDataset_v3**：从pickle文件加载图像路径列表（本项目使用）

图像预处理：
- 支持 PNG、JPEG、JPG、TIF、TIFF 格式
- 自动 BGR 到 RGB 转换（使用 cv2 读取）

### 4. utils.py - 工具函数

主要功能：
- **分布式训练**：`init_distributed_mode()`, `get_world_size()`, `get_rank()`
- **数据增强**：`GaussianBlur`, `Solarization`
- **优化器**：`LARS` 优化器实现
- **学习率调度**：`cosine_scheduler()` 余弦退火调度
- **模型包装**：`MultiCropWrapper` 处理多分辨率输入
- **日志记录**：`MetricLogger`, `SmoothedValue`
- **检查点管理**：`restart_from_checkpoint()`, `save_on_master()`

## 数据增强策略

DINO采用多裁剪数据增强策略：

1. **全局裁剪** (512×512)：
   - 尺度范围：0.4 ~ 1.0
   - 包含：随机翻转、颜色抖动、高斯模糊、日光化

2. **局部裁剪** (224×224)：
   - 尺度范围：0.05 ~ 0.4
   - 数量：8个
   - 包含：随机翻转、颜色抖动、高斯模糊

归一化参数（针对细胞图像优化）：
- Mean: (0.7896, 0.8113, 0.8456)
- Std: (0.1599, 0.1378, 0.1046)

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- torchvision
- CUDA 11.0+ (推荐)
- OpenCV (cv2)
- PIL (Pillow)
- NumPy

### 训练命令

```bash
cd code

# 使用4个GPU训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.run --nproc_per_node=4 main_dino.py \
    --arch regnet_y_800mf \
    --valid_data_pkl ../datalists/sample_cells_valid_img_path.pkl \
    --output_dir ../models \
    --batch_size_per_gpu 4 \
    --saveckp_freq 1
```

### 使用ViT模型训练

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

## 训练日志示例

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

## 模型检查点

训练过程中保存的检查点包含：
- `student`: 学生网络状态
- `teacher`: 教师网络状态
- `optimizer`: 优化器状态
- `epoch`: 当前轮数
- `args`: 训练参数
- `dino_loss`: 损失函数状态
- `fp16_scaler`: 混合精度缩放器状态（如果启用）

## 应用场景

本项目特别适用于：
- 细胞图像特征提取
- 医学图像无监督预训练
- 下游任务：分类、检测、分割

## 参考文献

- Caron M, Touvron H, Misra I, et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
- DINO GitHub: https://github.com/facebookresearch/dino

## 许可证

本项目基于 Apache License 2.0 许可证，代码来源于 Facebook Research DINO 项目。

---

*注：README和注释使用了AI辅助润色。*