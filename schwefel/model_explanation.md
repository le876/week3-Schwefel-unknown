# Schwefel函数神经网络分解模型总结

## 1. 项目概述

本项目旨在使用神经网络模型来逼近高维Schwefel函数。与传统方法不同，我们采用了一种基于函数分解的方法，利用Schwefel函数在各维度上的独立性特征，通过分解-组合的方式实现高效、精确的函数逼近。

## 2. 模型架构

### 2.1 主要设计思想

模型基于以下关键思想设计：

1. **函数分解**：Schwefel函数在各维度上是独立的，因此可以将其分解为多个一维子函数的和。
2. **参数共享**：由于每个维度上的函数形式相同，使用一个共享参数的子网络处理所有维度。
3. **组合策略**：通过简单求和将各维度的输出合并，得到最终预测结果。

### 2.2 网络结构详解

#### 2.2.1 子网络 (SubNetwork)

子网络负责处理单个维度的输入，结构如下：

```python
class SubNetwork(nn.Module):
    def __init__(self, hidden_size=256):
        # 输入层
        self.layer1 = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # 隐藏层1
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # 隐藏层2
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # 隐藏层3
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, 1)
```

每个隐藏层包含以下组件：
- **线性变换**：进行特征转换
- **批量归一化**：加速训练并提高稳定性
- **LeakyReLU激活**：引入非线性，同时避免神经元"死亡"问题
- **Dropout**：随机丢弃部分神经元，防止过拟合

#### 2.2.2 自动分解网络 (AutoDecomposedNetwork)

这是整个模型的主体架构，包含：

1. **共享子网络**：处理所有维度的单一子网络
2. **前向传播逻辑**：
   - 训练模式：可以针对单一维度进行训练
   - 推理模式：处理所有维度并汇总结果

```python
class AutoDecomposedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=256):
        self.input_dim = input_dim
        self.shared_subnet = SubNetwork(hidden_size)
```

#### 2.2.3 工作流程图解

```
输入(20维) → 维度拆分 → 子网络(共享参数) → 各维度输出 → 求和 → 最终预测
```

## 3. 训练方法

### 3.1 并行训练策略

我们采用"并行训练"策略，同时考虑所有维度的贡献：

1. 对每个样本的所有维度同时通过共享子网络处理
2. 将所有维度的输出相加得到总预测
3. 计算总预测与真实值的损失，并反向传播更新参数

这种方法相比于"依次训练"每个维度的策略，能够更全面地捕捉数据特征，提高模型性能。

### 3.2 混合损失函数

为了同时优化预测值与真实值之间的数值差异和相关性，我们使用了混合损失函数：

```
Loss = α × MSE + (1-α) × (1-Pearson相关系数)
```

其中：
- **MSE**：均方误差，衡量预测值与真实值的数值差异
- **Pearson相关系数**：衡量预测值与真实值的线性相关程度
- **α**：权衡因子，设置为0.5，平衡两种损失的比重

### 3.3 优化器与学习率调度

- **优化器**：Adam优化器，初始学习率为0.001，权重衰减为0.0001
- **学习率调度**：使用ReduceLROnPlateau策略，当验证损失不再下降时，将学习率乘以0.7
- **早停机制**：如果连续50轮验证集性能没有提升，则停止训练

## 4. 评估指标与可视化

### 4.1 主要评估指标

- **均方误差(MSE)**：直接衡量预测值与真实值的差异
- **Pearson相关系数**：衡量预测趋势的准确性，范围[-1,1]，越接近1表示预测越准确

### 4.2 可视化内容

模型训练过程中会生成以下可视化：

1. **训练与验证损失曲线**：展示模型的学习进度和是否过拟合
2. **Pearson相关系数曲线**：展示模型预测能力的提升过程
3. **预测值vs真实值散点图**：直观展示预测准确性
4. **维度贡献可视化**：展示各维度对最终预测的贡献比例

## 5. 实验结果

通过优化模型结构和训练策略，我们在测试集上实现了以下性能：

- **MSE**：较低的均方误差值
- **Pearson相关系数**：0.89+，表明模型能够非常好地捕捉Schwefel函数的特性

## 6. 关键技术要点总结

1. **参数共享**：大幅减少模型参数量，提高训练效率，同时提供更多训练信号
2. **混合损失函数**：同时优化数值准确性和趋势一致性
3. **批量归一化**：加速收敛并提高训练稳定性
4. **学习率动态调整**：根据验证性能动态调整学习率，避免训练陷入局部最优
5. **正则化技术**：Dropout和权重衰减，有效防止过拟合

## 7. 适用场景与局限性

### 适用场景
- 具有特定结构（如各维度独立）的高维函数逼近
- 需要解释各维度贡献的场景
- 样本数量有限的建模任务

### 局限性
- 对于维度间有复杂交互的函数可能效果有限
- 需要对函数特性有一定先验知识才能设计最佳架构

## 8. 未来改进方向

1. **自适应维度权重**：动态调整各维度的重要性权重
2. **更先进的激活函数**：尝试如Mish、Swish等新型激活函数
3. **多任务学习**：同时预测函数值及其导数
4. **集成策略**：训练多个模型并集成结果，提高稳定性和准确性

## 9. 模型代码示例

```python
class SubNetwork(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        # 输入层
        self.layer1 = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # 隐藏层
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # 更多层...
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # 更多层...
        x = self.output_layer(x)
        return x

class AutoDecomposedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=256):
        super().__init__()
        self.input_dim = input_dim
        self.shared_subnet = SubNetwork(hidden_size)
        
    def forward(self, x, current_dim=None):
        # 实现核心逻辑...
```

## 10. 术语解释

- **批量归一化(Batch Normalization)**：通过归一化每一层的输入来加速深度网络训练的技术
- **LeakyReLU**：修正线性单元的一种变体，允许负值输入产生较小的负输出，避免"死亡ReLU"问题
- **Dropout**：一种正则化技术，在训练过程中随机"丢弃"一部分神经元，防止过拟合
- **Adam优化器**：结合了动量法和RMSProp的优点，自适应调整每个参数的学习率
- **早停(Early Stopping)**：当验证集性能不再提升时停止训练，防止过拟合
- **Pearson相关系数**：衡量两个变量线性相关程度的统计量，取值范围[-1,1] 