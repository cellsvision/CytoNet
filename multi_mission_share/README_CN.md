# 多任务MIL模型：分类与生存分析

## 概述

本项目实现了一个**多任务多示例学习(MIL)**模型，同时完成以下任务：
- **分类任务**：病理切片的二分类（阴性/阳性）
- **生存分析**：基于Cox比例风险损失的无病生存期(DFS)预测

## 项目结构

```
multi_mission_share/
├── code/                          # 源代码
│   ├── train_test_multi.py        # 主训练与评估脚本
│   ├── CoxPHLoss.py              # Cox比例风险损失函数实现
│   └── utils.py                   # 工具函数（C-index、风险集计算等）
├── datalists/                     # 数据配置
│   ├── train.csv                  # 训练数据列表（含生存信息）
│   └── val.csv                    # 验证数据列表
├── sample_data/                   # 示例特征文件（.pkl格式）
├── models/                        # 保存的模型检查点
├── logs/                          # 训练日志
└── results/                       # 评估结果
```

## 模型架构

- **骨干网络**：基于注意力机制的MIL，含共享全连接层
- **分类头**：多层感知机，用于二分类
- **生存分析头**：回归头，用于风险评分预测
- **损失函数**：交叉熵损失 + Cox PH损失的联合损失

## 数据格式

### CSV列说明
| 列名 | 描述 |
|------|------|
| `name` | 样本标识符（对应.pkl文件名） |
| `GT` | 真实标签（Neg/Pos） |
| `dfs_status` | 无病生存状态（0/1） |
| `dfs_time` | 无病生存时间（月） |

### 特征文件
- 格式：`.pkl`（pickle）文件，包含768维特征
- 每个文件包含按类别组织的多个patch特征

## 训练方法

```bash
cd code
python train_test_multi.py
```

### 关键参数
- `fea_size`：特征维度（768）
- `batch_size`：训练批次大小（4）
- `max_epoch`：最大训练轮数（100）
- `max_patches`：每类最大patch数（200）
- `alpha`：分类损失权重（1.0）
- `beta`：生存损失权重（1.0）

## 评估指标

- **分类任务**：AUC、敏感性、特异性
- **生存分析**：C-Index、时间依赖性AUC（12、24、36、60个月）

## 依赖环境

- PyTorch
- scikit-learn
- lifelines
- pandas
- numpy
- tensorboard

## 示例数据

`sample_data/`目录包含24个示例特征文件（1.pkl - 24.pkl）供演示使用。

## 日志说明

训练日志保存至`logs/lymph_survival_train.log`，包含：
- 每批次损失值
- 每轮C-Index和AUC指标
- 验证结果详情

---

*注：README和注释使用了AI辅助润色。*