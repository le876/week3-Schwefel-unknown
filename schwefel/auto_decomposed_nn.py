import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import copy
import math

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置matplotlib样式
plt.style.use('default')  
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class SchwefelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        print(f"\n数据集形状：")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SubNetwork(nn.Module):
    def __init__(self, hidden_size=256):  
        super().__init__()
        # 输入层
        self.layer1 = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),  # 改回LeakyReLU
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
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x

class AutoDecomposedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=256):
        super().__init__()
        self.input_dim = input_dim
        
        # 创建一个共享的子网络
        self.shared_subnet = SubNetwork(hidden_size)
        
    def forward(self, x, current_dim=None):
        batch_size = x.size(0)
        
        if current_dim is not None:
            # 训练模式：只处理当前维度
            input_dim_i = x[:, current_dim:current_dim+1]  # (batch_size, 1)
            output_dim_i = self.shared_subnet(input_dim_i)  # (batch_size, 1)
            return output_dim_i
        else:
            # 推理模式：处理所有维度并将结果相加
            outputs = []
            for i in range(self.input_dim):
                input_dim_i = x[:, i:i+1]  # (batch_size, 1)
                output_dim_i = self.shared_subnet(input_dim_i)  # (batch_size, 1)
                outputs.append(output_dim_i)
            
            # 将所有维度的输出相加
            summed_output = torch.sum(torch.cat(outputs, dim=1), dim=1, keepdim=True)
            return summed_output
    
    def forward_all_dims(self, x):
        """并行计算所有维度，返回每个维度的输出和总输出"""
        batch_size = x.size(0)
        outputs = []
        
        for i in range(self.input_dim):
            input_dim_i = x[:, i:i+1]  # (batch_size, 1)
            output_dim_i = self.shared_subnet(input_dim_i)  # (batch_size, 1)
            outputs.append(output_dim_i)
        
        # 所有维度输出拼接成一个张量
        all_outputs = torch.cat(outputs, dim=1)  # (batch_size, input_dim)
        
        # 所有维度输出求和
        summed_output = torch.sum(all_outputs, dim=1, keepdim=True)  # (batch_size, 1)
        
        return summed_output, all_outputs

def pearson_correlation_loss(y_pred, y_true):
    """计算Pearson相关系数损失"""
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))
    return 1.0 - cost

def hybrid_loss(y_pred, y_true, alpha=0.5):  # 调整alpha权重为0.5
    """混合MSE和Pearson相关系数损失"""
    mse = torch.mean((y_pred - y_true) ** 2)
    corr_loss = pearson_correlation_loss(y_pred, y_true)
    return alpha * mse + (1 - alpha) * corr_loss

def load_data(data_dir='schwefel/data/raw'):
    """加载数据"""
    X_train = np.load(os.path.join(data_dir, 'Schwefel_x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'Schwefel_y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'Schwefel_x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'Schwefel_y_test.npy'))
    
    # 打印原始数据形状
    print("原始数据形状：")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 数据标准化
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))
    
    # 打印标准化后的数据形状
    print("\n标准化后的数据形状：")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 打印划分后的数据形状
    print("\n划分后的数据形状：")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler

def create_output_dir():
    """创建输出目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'schwefel/visualizations/auto_decomposed_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def train_model(model, train_loader, val_loader, num_epochs, device, output_dir):
    """优化后的训练函数"""
    best_val_corr = 0
    train_losses = []
    val_losses = []
    train_correlations = []
    val_correlations = []
    patience = 50  # 恢复合理的早停耐心值
    patience_counter = 0
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # 定义损失函数
    criterion = hybrid_loss
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10, verbose=True
    )
    
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            total_output, _ = model.forward_all_dims(X)
            
            # 计算损失
            loss = criterion(total_output, y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(total_output.cpu().detach().numpy())
            train_targets.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_corr = np.corrcoef(np.array(train_preds).flatten(), 
                                 np.array(train_targets).flatten())[0, 1]
        train_losses.append(train_loss)
        train_correlations.append(train_corr)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                
                # 前向传播
                total_output, _ = model.forward_all_dims(X)
                
                # 计算损失
                loss = criterion(total_output, y)
                val_loss += loss.item()
                val_preds.extend(total_output.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_corr = np.corrcoef(np.array(val_preds).flatten(), 
                               np.array(val_targets).flatten())[0, 1]
        val_losses.append(val_loss)
        val_correlations.append(val_corr)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型和早停
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停: {epoch+1}轮无改善')
                break
        
        # 每10轮输出训练信息
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Pearson: {train_corr:.4f}, Val Pearson: {val_corr:.4f}, '
                  f'LR: {current_lr:.6f}')
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_correlations, label='Training Pearson')
    plt.plot(val_correlations, label='Validation Pearson')
    plt.title(f'Pearson Correlations (Best Val: {best_val_corr:.4f} at Epoch {best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # 记录训练历史
    with open(os.path.join(output_dir, 'training_history.txt'), 'w') as f:
        f.write(f'Best Validation Pearson: {best_val_corr:.4f} at Epoch {best_epoch+1}\n\n')
        f.write('Epoch,TrainLoss,ValLoss,TrainPearson,ValPearson\n')
        for i in range(len(train_losses)):
            f.write(f'{i+1},{train_losses[i]:.6f},{val_losses[i]:.6f},'
                    f'{train_correlations[i]:.6f},{val_correlations[i]:.6f}\n')
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    
    return model

def evaluate_model(model, test_loader, device, output_dir):
    """评估模型"""
    model_output_dir = output_dir
    
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            # 完整预测（所有维度）
            y_pred, _ = model.forward_all_dims(X)
            
            predictions.extend(y_pred.cpu().numpy())
            true_values.extend(y.cpu().numpy())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # 计算性能指标
    mse = np.mean((predictions - true_values) ** 2)
    correlation = np.corrcoef(predictions.flatten(), true_values.flatten())[0, 1]
    
    # 绘制预测值vs真实值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predictions, alpha=0.5)
    plt.plot([true_values.min(), true_values.max()], 
             [true_values.min(), true_values.max()], 
             'r--', label='Perfect Prediction')
    plt.title(f'Predicted vs True Values\nTest Pearson: {correlation:.4f}, MSE: {mse:.4f}')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(os.path.join(model_output_dir, 'prediction_scatter.png'))
    plt.close()
    
    # 可视化每个维度单独的预测结果
    plt.figure(figsize=(15, 10))
    
    # 选择一个小的测试样本进行可视化
    sample_size = min(100, len(test_loader.dataset))
    sample_indices = np.random.choice(len(test_loader.dataset), sample_size, replace=False)
    
    sample_X = test_loader.dataset.X[sample_indices].to(device)
    sample_y = test_loader.dataset.y[sample_indices].to(device)
    
    # 预测每个维度的贡献
    dim_preds = []
    with torch.no_grad():  # 确保不会计算梯度
        for i in range(model.input_dim):
            pred_i = model(sample_X, current_dim=i).cpu().detach().numpy()
            dim_preds.append(pred_i)
    
    # 计算总预测
    total_pred = np.sum(dim_preds, axis=0)
    
    # 显示维度贡献
    for i in range(model.input_dim):
        plt.subplot(4, 5, i+1)
        contribution_ratio = np.abs(dim_preds[i]) / (np.abs(total_pred) + 1e-10)  # 添加小值避免除零
        plt.hist(contribution_ratio, bins=20, alpha=0.7)
        plt.title(f'Dim {i+1} Contribution')
        plt.tight_layout()
    
    plt.savefig(os.path.join(model_output_dir, 'dimension_contributions.png'))
    plt.close()
    
    # 可视化每个维度的平均贡献
    mean_contributions = [np.mean(np.abs(dim_preds[i])) for i in range(model.input_dim)]
    total_contribution = sum(mean_contributions)
    contribution_percentages = [100 * cont / total_contribution for cont in mean_contributions]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, model.input_dim + 1), contribution_percentages)
    plt.xlabel('Dimension')
    plt.ylabel('Average Contribution (%)')
    plt.title('Average Contribution of Each Dimension')
    plt.xticks(range(1, model.input_dim + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(model_output_dir, 'dimension_average_contribution.png'))
    plt.close()
    
    # 保存评估结果
    with open(os.path.join(model_output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f'Test MSE: {mse:.4f}\n')
        f.write(f'Test Pearson Correlation: {correlation:.4f}\n')
        f.write('\nDimension Contributions (%):\n')
        for i, cont in enumerate(contribution_percentages):
            f.write(f'Dimension {i+1}: {cont:.2f}%\n')
    
    return mse, correlation

def main():
    # 创建输出目录
    output_dir = create_output_dir()
    
    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler = load_data()
    
    # 创建数据加载器
    train_dataset = SchwefelDataset(X_train, y_train)
    val_dataset = SchwefelDataset(X_val, y_val)
    test_dataset = SchwefelDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # 恢复到128的batch size
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = AutoDecomposedNetwork(input_dim=20, hidden_size=256).to(device)
    
    # 训练模型
    print("开始训练...")
    model = train_model(model, train_loader, val_loader,
                         num_epochs=300, device=device, output_dir=output_dir)
    
    # 评估模型
    mse, correlation = evaluate_model(model, test_loader, device, output_dir)
    
    print(f'结果已保存到 {output_dir}')
    print(f'测试集 MSE: {mse:.4f}')
    print(f'测试集 Pearson 相关系数: {correlation:.4f}')

if __name__ == '__main__':
    main() 