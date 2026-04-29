# CytoNet: 基于多类别注意力机制的MIL诊断分类系统

## 项目概述

CytoNet是一个用于液基细胞学样本自动诊断分类的深度学习系统。该系统采用二阶段多类别特征聚合的注意力机制多实例学习（MIL）框架，实现患者级（slide-level）的分类任务与生存分析预测。

### 核心特点

- **自监督预训练**：基于DINO方法的无标签特征学习
- **监督微调**：RegNet-Y + CBAM注意力机制的细胞分类
- **多任务MIL**：同时支持诊断分类与生存分析
- **多类别特征聚合**：按细胞类型分组提取判别性特征

---

## 项目结构

```
CytoNet/
├── dino/                          # 阶段一：自监督预训练
│   ├── code/
│   │   ├── main_dino.py           # DINO训练主程序
│   │   ├── vision_transformer.py  # Vision Transformer实现
│   │   ├── csv_dataloader.py      # 数据加载器
│   │   ├── utils.py               # 工具函数
│   │   └── run_train.sh           # 训练启动脚本
│   ├── datalists/                 # 数据列表
│   ├── models/                    # 模型检查点
│   └── sample_data/               # 样本数据
│
├── supervised_finetune/           # 阶段二：监督微调
│   ├── code/
│   │   ├── config.py              # 训练配置
│   │   ├── train_test.py          # 训练测试脚本
│   │   ├── convert_to_torchvision.py  # 权重格式转换
│   │   └── extract_weights.py     # 权重提取工具
│   ├── checkpoint/                # 预训练权重
│   ├── datalists/                 # 数据列表
│   ├── models/                    # 微调模型
│   └── sample_data/               # 分类样本数据
│       ├── AUC/                   # AUC类别样本
│       ├── HGUC/                  # HGUC类别样本
│       ├── yin/                   # 正常类别样本（NILM/NHGUC）
│       ├── HISTIOCYTE/            # 组织细胞样本
│       ├── IMPURITY/              # 杂质样本
│       └── blank/                 # 空白区域样本
│
├── multi_mission_share/           # 阶段三：多任务MIL
│   ├── code/
│   │   ├── train_test_multi.py    # 多任务训练脚本
│   │   ├── CoxPHLoss.py           # Cox比例风险损失
│   │   └── utils.py               # 生存分析工具
│   ├── datalists/                 # 包含生存信息的数据
│   ├── models/                    # MIL模型
│   └── sample_data/               # 特征文件
│
└── Method.docx                    # 方法详细描述文档
```

---

## 技术方法

### 1. 输入数据组织

#### 阶段一：DINO自监督预训练

使用超过10万张未标记WSI，按大小尺寸进行裁剪。

#### 阶段二：监督微调（尿液细胞学）

- 每张WSI被划分为多个图像块（patches）
- 根据标注裁切对应标注的图像块，并针对尿液细胞学特殊的背景图（血液/胶质/尿蛋白、组织细胞）与空白背景，准备6个类别的图像块：
  - **NHGUC**：正常细胞（NILM/NHGUC）
  - **AUC**：非典型尿细胞
  - **HGUC**：高级别尿细胞癌
  - **HISTIOCYTE**：组织细胞
  - **IMPURITY**：杂质
  - **BLANK**：空白区域

#### 阶段三：

- 原始输入来源于全片尿液细胞学图像（WSI）的细胞分类结果：
  - 按40倍图像下、符合patch size大小的patch推理的类别与featuremap按照每一类的置信度前N个feature倒序排列并拼接聚合形成Tensor。例如，某张图可以裁切出50 * 60张patch，每个patch输入模型输出patch级别分类结果以及提取出特征图后，按照每一类置信度倒序的前300拼接成 [6*300, featuremap_length] 数组，保存成pkl。
- 数据输入时，选取每类别top-200个最具代表性的图像块

### 2. 阶段一：DINO自监督预训练

#### 模型架构
- 支持Vision Transformer (ViT-Tiny/Small/Base) 和 各种CNN架构
- 教师-学生网络结构，通过EMA更新教师权重
- DINO投影头：3层MLP + L2归一化

#### 数据增强策略
- **全局裁剪** (512×512)：尺度范围 0.4~1.0
- **局部裁剪** (224×224)：尺度范围 0.05~0.4，共8个
- 包含：随机翻转、颜色抖动、高斯模糊、日光化

#### 训练参数
| 参数 | 值 |
|------|-----|
| 学习率 | 0.0005 |
| 权重衰减 | 0.04 → 0.4 (余弦调度) |
| 批次大小 | 4/GPU |
| 优化器 | AdamW |
| 输出维度 | 65536 |

### 3. 阶段二：监督微调

#### 模型结构
- **骨干网络**：RegNet-Y-800MF（加载DINO预训练权重）
- **注意力模块**：CBAM (Convolutional Block Attention Module)
- **分类头**：6类细胞分类

#### 分类类别
```python
train_patch_class_id = {
    'nilm': 0,     # 正常（NILM/NHGUC）
    'auc': 1,      # 非典型尿细胞
    'hguc': 2,     # 高级别尿细胞癌
    'impurity': 3, # 杂质
    'histiocyte': 4, # 组织细胞
    'blank': 5     # 空白
}
```

#### 数据增强（Albumentations）
- JPEG压缩
- Gamma校正 / CLAHE / 亮度对比度调整
- 翻转
- 色调饱和度 / RGB偏移
- ISO噪声 / 高斯噪声

#### 训练策略
- 标签平滑（0.95置信度）
- 每slide采样平衡
- CosineAnnealingWarmRestarts学习率调度

### 4. 阶段三：多任务MIL模型

#### 模型架构

**实例级特征编码器**：
```
h_i^c = MLP_c(x_i^c)  # 每类别独立的全连接层
```

**注意力权重模块**：
```
A_c = W_c · (tanh(V_c · h) ⊙ sigmoid(U_c · h))
```
- V_c: Tanh分支，捕捉特征显著性
- U_c: Sigmoid分支，学习门控机制
- 中间维度 D = 256

**注意力加权聚合**：
```
z_c = Σ_i softmax(A_c,i) · h_i^c
```

**多任务头**：
- **分类头**：输出2类诊断结果（阴性/阳性）
- **生存分析头**：输出单维风险分数

#### 损失函数

**多任务联合损失**：
```
L_total = α · L_cls + β · L_surv
```

**分类损失**：交叉熵损失
```
L_cls = -Σ y_k log(p_k)
```

**生存分析损失**：Cox比例风险损失
```
L_surv = -Σ_e=1 (log_h_i - log(Σ_j∈R_i exp(log_h_j)))
```

#### 评估指标
- **分类**：AUC、敏感度、特异度、Kappa系数
- **生存分析**：C-Index、时间依赖AUC (12/24/36/48/60月)

---

## 快速开始

### 环境要求

```bash
# Python 3.7+
# PyTorch 1.9+
pip3 install torch torchvision
pip3 install timm albumentations pandas numpy scikit-learn
pip3 install lifelines tensorboard torch-optimizer
pip3 install easydict PIL cv2
```

### 阶段一：DINO预训练

```bash
cd dino/code

# 使用4个GPU训练(示例)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.run --nproc_per_node=4 main_dino.py \
    --arch regnet_y_800mf \
    --valid_data_pkl ../datalists/sample_cells_valid_img_path.pkl \
    --output_dir ../models \
    --batch_size_per_gpu 4 \
    --saveckp_freq 1
```

### 阶段二：权重转换与微调

```bash
cd supervised_finetune/code

# 步骤1：转换DINO权重为torchvision格式
python3 convert_to_torchvision.py

# 步骤2：训练分类模型
python3 train_test.py
```

### 阶段三：多任务MIL训练

```bash
cd multi_mission_share/code

# 训练多任务模型（分类+生存分析）
python3 train_test_multi.py
```

---

## 核心代码说明

### DINO训练 ([`dino/code/main_dino.py`](dino/code/main_dino.py))

- `DINOLoss`: 知识蒸馏损失，教师-学生网络交叉熵
- `DataAugmentationDINO`: 多裁剪数据增强策略
- `train_dino()`: 主训练流程，支持分布式训练

### 监督微调 ([`supervised_finetune/code/train_test.py`](supervised_finetune/code/train_test.py))

- `UrineDataset`: 细胞图像数据集，支持每epoch重新采样
- `UrineModel`: RegNet-Y + CBAM分类模型
- `evaluate_model()`: 计算多类和二类准确率

### 多任务MIL ([`multi_mission_share/code/train_test_multi.py`](multi_mission_share/code/train_test_multi.py))

- `MILAttModule`: 注意力模块，Tanh-Sigmoid门控机制
- `MultiTaskMILModel`: 多任务MIL模型，分类+生存分析双头
- `MultiTaskMILDataset`: 加载特征文件与生存信息

### Cox损失 ([`multi_mission_share/code/CoxPHLoss.py`](multi_mission_share/code/CoxPHLoss.py))

- `CoxPHLoss`: Cox比例风险损失函数
- `coxPHLoss()`: 批处理Cox损失计算
- `make_riskset()`: 构建风险集矩阵

---

## 数据格式

### DINO训练数据
- **输入**：pickle文件包含图像路径列表
- **格式**：`['/path/to/image1.png', '/path/to/image2.png', ...]`

### 监督微调数据
- **CSV格式**：
```
filename
WSI1
WSI2
...
```
- **目录结构**：
```
sample_data/
├── AUC/WSI4/AUC/*.png
├── HGUC/WSI6/HGUC/*.png
├── yin/WSI7/NILM/*.png
...
```

### MIL训练数据
- **CSV格式**（包含生存信息）：
```
name,GT,dfs_status,dfs_time
patient_001,Pos,1,24
patient_002,Neg,0,60
...
```
- **特征文件**：`{name}.pkl`，形状 `(1200, 768)`

---

## 模型检查点

### DINO检查点包含
- `student`: 学生网络权重
- `teacher`: 教师网络权重（推荐用于下游任务）
- `optimizer`: 优化器状态
- `epoch`: 训练轮数

### 微调模型
- `ckpt_{epoch}_{acc}_{acc_2}.pth`: 包含分类准确率

### MIL模型
- 多任务模型权重，支持分类和生存分析推理

---

## 应用场景

1. **尿液细胞学淋巴结转移诊断**
   - 自动区分阴性与阳性病例
   - 辅助病理医生诊断

2. **患者生存预测**
   - 预测无病生存期（DFS）
   - 风险分层

3. **细胞图像特征提取**
   - 自监督学习的通用特征
   - 可迁移至其他医学图像任务

---

## 参考文献

- Caron M, et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
- Ilse M, et al. "Attention-based Deep Multiple Instance Learning." ICML 2018.
- Kvamme H, et al. "Time-to-Event Prediction with Neural Networks and Cox Regression."

---

## 许可证

本项目代码基于 Apache License 2.0，部分代码来源于 Facebook Research DINO 项目。

---

## 联系方式

如有问题或建议，请通过项目Issue页面联系。

---

*注：README和注释使用了AI辅助润色。*