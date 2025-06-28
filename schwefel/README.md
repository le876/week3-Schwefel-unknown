# Schwefel函数神经网络逼近项目

本项目旨在使用神经网络模型逼近高维Schwefel函数，采用基于函数分解的创新方法，利用Schwefel函数在各维度上的独立性特征，通过分解-组合策略实现高效、精确的函数逼近。

## 项目概述

Schwefel函数是一个经典的多模态优化测试函数，具有以下特点：
- 函数形式：`f(x) = 418.9829 * d - Σ(x_i * sin(sqrt(|x_i|)))`
- 输入维度：20维
- 定义域：每个维度 [-500, 500]
- 特性：各维度独立，适合分解方法

## 项目结构

```
schwefel/
├── data/                           # 数据目录
│   └── raw/                       # 原始数据文件
│       ├── Schwefel_x_train.npy  # 训练集输入
│       ├── Schwefel_y_train.npy  # 训练集标签
│       ├── Schwefel_x_test.npy   # 测试集输入
│       └── Schwefel_y_test.npy   # 测试集标签
├── models/                         # 模型保存目录
├── results/                        # 结果保存目录
├── visualizations/                 # 可视化结果目录
├── analysis_results/               # 数据分析结果
├── decomposed_nn.py               # 分解神经网络模型
├── auto_decomposed_nn.py          # 自动分解神经网络模型
├── train_rf.py                    # 随机森林训练脚本
├── train_gbdt.py                  # 梯度提升树训练脚本
├── data_analysis.py               # 数据分析脚本
├── visualization_utils.py         # 可视化工具
├── model_explanation.md           # 模型详细说明文档
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明文档
```

## 核心方法

### 1. 分解神经网络 (Decomposed Neural Network)

基于Schwefel函数各维度独立的特性，设计了分解神经网络：

- **子网络设计**：每个维度使用独立的子网络处理
- **网络结构**：多层感知机，包含BatchNorm、LeakyReLU激活和Dropout正则化
- **训练策略**：分别训练各维度的贡献值，最后求和得到总预测

### 2. 自动分解神经网络 (Auto-Decomposed Neural Network)

改进的分解方法，使用参数共享：

- **参数共享**：所有维度共享同一个子网络参数
- **并行训练**：同时处理所有维度，提高训练效率
- **混合损失函数**：结合MSE和Pearson相关系数损失

### 3. 传统机器学习方法

- **随机森林**：基于树的集成方法，具有特征工程
- **梯度提升树**：XGBoost和LightGBM实现
- **特征工程**：针对Schwefel函数设计的专门特征

## 环境设置

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖包：
- numpy >= 1.20.0
- scipy >= 1.7.0  
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0
- joblib >= 1.0.0

### PyTorch安装（神经网络模型需要）

```bash
# CPU版本
pip install torch torchvision

# GPU版本（如果有CUDA）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 1. 数据分析

```bash
python schwefel/data_analysis.py
```

生成数据分析报告和可视化图表，包括：
- 特征分布分析
- 相关性分析
- 维度贡献分析

### 2. 训练分解神经网络

```bash
python schwefel/decomposed_nn.py
```

### 3. 训练自动分解神经网络

```bash
python schwefel/auto_decomposed_nn.py
```

### 4. 训练随机森林模型

```bash
python schwefel/train_rf.py
```

### 5. 训练梯度提升树模型

```bash
python schwefel/train_gbdt.py
```

## 评估指标

- **均方误差 (MSE)**：衡量预测值与真实值的数值差异
- **Pearson相关系数**：衡量预测趋势的准确性，目标 ≥ 0.9
- **决定系数 (R²)**：解释方差比例

## 实验结果

通过多种方法的对比实验，自动分解神经网络取得了最佳性能：

- **测试集Pearson相关系数**：0.89+
- **训练效率**：参数共享大幅减少模型复杂度
- **可解释性**：能够分析各维度的贡献

## 可视化输出

训练过程会自动生成以下可视化：

1. **训练曲线**：损失函数和相关系数随训练轮次的变化
2. **预测散点图**：预测值vs真实值的对比
3. **维度贡献图**：各维度对最终预测的贡献分析
4. **特征重要性图**：基于树模型的特征重要性排序

所有结果保存在 `visualizations/` 目录下，按时间戳命名。

## 技术特点

1. **创新的分解方法**：充分利用Schwefel函数的数学特性
2. **参数共享策略**：减少模型参数，提高泛化能力
3. **混合损失函数**：同时优化数值精度和相关性
4. **完整的实验框架**：从数据分析到模型评估的完整流程

## 文件说明

- `model_explanation.md`：详细的模型架构和训练策略说明
- `decomposed_nn.py`：传统分解神经网络实现
- `auto_decomposed_nn.py`：改进的自动分解神经网络
- `train_rf.py`：随机森林基线模型
- `train_gbdt.py`：梯度提升树模型
- `data_analysis.py`：数据探索性分析
- `visualization_utils.py`：可视化工具函数
