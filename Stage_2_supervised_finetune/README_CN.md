# 液基细胞学图像分类监督微调项目（尿液）

## 项目简介

本项目是一个基于深度学习的液基细胞学图像分类监督微调（Supervised Fine-tuning）系统。该系统使用预训练的 RegNet-Y-800MF 模型作为骨干网络，结合 CBAM 注意力机制，以尿液细胞学为示例，对尿液细胞学图像进行多类别分类。

## 分类类别

本项目支持以下 6 个类别的分类：

| 类别代码 | 类别名称 | 描述 |
|---------|---------|------|
| NHGUC | 阴性 | 正常尿液细胞（NILM/NHGUC） |
| AUC | 非典型尿路上皮细胞 | Atypical Urothelial Cells |
| HGUC | 高级别尿路上皮癌 | High-Grade Urothelial Carcinoma |
| IMPURITY | 杂质 | 图像中的杂质区域 |
| HISTIOCYTE | 组织细胞 | Histiocyte 细胞 |
| BLANK | 空白 | 空白背景区域 |

## 目录结构

```
supervised_finetune/
├── checkpoint/          # 预训练权重存储目录
│   └── teacher_backbone_torchvision.pth  # 教师模型骨干权重
├── code/                # 源代码目录
│   ├── config.py        # 训练配置文件
│   ├── train_test.py    # 训练与测试主程序
│   ├── extract_weights.py    # 从DINO检查点提取权重
│   └── convert_to_torchvision.py  # 权重格式转换
├── datalists/           # 数据列表目录
│   ├── train.csv        # 训练集文件列表
│   └── val.csv          # 验证集文件列表
├── logs/                # 训练日志目录
├── models/              # 模型保存目录
├── results/             # 结果输出目录
└── sample_data/         # 示例数据目录
    ├── AUC/             # AUC 类别样本
    ├── blank/           # 空白区域样本
    ├── HGUC/            # HGUC 类别样本
    ├── histiocyte/      # 组织细胞样本
    ├── impurity/        # 杂质样本
    └── yin/             # NILM/NHGUC 类别样本
```

*标注人员可能会习惯使用NILM或NHGUC的写法来标注阴性，仅写法差异，真实含义均代表正常的尿路上皮细胞。

## 模型架构

- **骨干网络**: RegNet-Y-800MF
- **注意力机制**: CBAM (Convolutional Block Attention Module)
- **分类头**: 6 类分类输出

## 环境依赖

参考requirements.txt

## 使用方法

### 1. 数据准备

将尿液细胞学图像按类别放置在 `sample_data/` 目录下，每个 WSI（全切片图像）的 patch 图像存放在对应的子目录中。

数据目录结构示例：
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

### 2. 配置修改

在 [`code/config.py`](code/config.py) 中修改训练配置：

```python
train_cfg.GPU_ID = "0, 1, 2, 3"  # GPU 设备ID
train_cfg.batch_size = 4          # 批次大小
train_cfg.max_epoch = 30           # 训练轮数
train_cfg.target_size = 1024     # 输入图像尺寸
train_cfg.checkpoint_path = '../checkpoint/teacher_backbone_torchvision.pth'  # 预训练权重路径
```

### 3. 开始训练

```bash
cd code
python train_test.py
```

### 4. 权重提取与转换

如果需要从 DINO 检查点提取权重：

```bash
# 提取权重
python extract_weights.py

# 转换为 torchvision 格式
python convert_to_torchvision.py
```

## 训练策略

### 优化器设置
- **优化器**: Adam
- **学习率**: 5e-6
- **学习率调度**: CosineAnnealingWarmRestarts
  - T_0 = 3
  - T_mult = 2
  - eta_min = 1e-7

### 损失函数
- CrossEntropyLoss

### 数据增强策略
训练过程中使用 albumentations 库进行数据增强：

- JPEG 压缩（质量 70-95）
- 随机 Gamma 校正（gamma_limit: 50-150）
- CLAHE（对比度受限自适应直方图均衡化）
- 随机亮度对比度调整
- 随机翻转
- 色调、饱和度、值调整
- RGB 偏移
- 高斯噪声
- ISO 噪声
- 乘性噪声

## 评估指标

- **准确率 (Accuracy)**: 6 类分类准确率
- **二分类准确率 (acc_2)**: 将类别合并为阴性/阳性后的准确率
  - 阴性类别: NILM/NHGUC, IMPURITY, HISTIOCYTE, BLANK
  - 阳性类别: AUC, HGUC

## 代码文件说明

### [`config.py`](code/config.py)
训练配置文件，包含：
- GPU 设备配置
- 数据路径配置
- 类别映射配置
- 训练超参数配置

### [`train_test.py`](code/train_test.py)
主训练脚本，包含：
- `UrineDataset`: 尿液图像数据集类
- `UrineModel`: 分类模型类
- `load_model()`: 加载预训练模型
- `load_trained_model()`: 加载已训练模型
- 训练和验证循环

### [`extract_weights.py`](code/extract_weights.py)
从 DINO 检查点提取权重，生成：
- student_backbone.pth: 学生模型骨干权重
- teacher_backbone.pth: 教师模型骨干权重
- student_full.pth: 学生模型完整权重
- teacher_full.pth: 教师模型完整权重

### [`convert_to_torchvision.py`](code/convert_to_torchvision.py)
将 DINO 权重格式转换为 torchvision 的 RegNet-Y-800MF 格式。

## 注意事项

1. **类别平衡**: 训练数据采用类别平衡策略，每个 WSI 中各类别采样数量有限制
2. **多 GPU 训练**: 模型支持多 GPU 并行训练 (DataParallel)
3. **日志记录**: 训练过程使用 TensorBoard 记录损失和准确率曲线
4. **标签平滑**: 使用标签平滑策略，正类权重 0.95，其他类权重 0.05/(n-1)

## 训练日志示例

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

## 许可证

本项目仅供研究使用。

## 联系方式

如有问题或建议，请联系项目维护人员。

---

*注：README和注释使用了AI辅助润色。*