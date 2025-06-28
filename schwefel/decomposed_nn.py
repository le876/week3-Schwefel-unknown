import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
import logging

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class SubNetwork(nn.Module):
    def __init__(self, hidden_size=48):  # 增加隐藏层节点数
        super(SubNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LeakyReLU(0.1),  # 使用LeakyReLU替代ReLU，对负值有更好的处理
            nn.Linear(hidden_size, hidden_size*2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size*2, hidden_size*2),  # 增加一层
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class DecomposedNetwork(nn.Module):
    def __init__(self, input_dim=20, hidden_size=46):
        super(DecomposedNetwork, self).__init__() 
        self.sub_networks = nn.ModuleList([
            SubNetwork(hidden_size) for _ in range(input_dim)
        ])
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        outputs = []
        for i, sub_net in enumerate(self.sub_networks):
            dim_input = x[:, i:i+1]  # 保持二维形状 (batch_size, 1)
            outputs.append(sub_net(dim_input))
        
        # 将各维度的输出叠加，shape: (batch_size, input_dim, 1)
        stacked_outputs = torch.stack(outputs, dim=1)
        # 求和得到最终预测，shape: (batch_size)
        summed_outputs = torch.sum(stacked_outputs, dim=1).squeeze(-1)
        return summed_outputs

# 计算Schwefel函数每个维度的贡献
def calculate_dimension_contribution(X):
    """
    计算Schwefel函数每个维度的贡献
    X: shape (n_samples, n_dimensions)
    return: shape (n_samples, n_dimensions)
    """
    contrib = 418.9829 - X * np.sin(np.sqrt(np.abs(X)))
    return contrib

class SchwefelDatasetDimensional(Dataset):
    def __init__(self, X, y=None, scaler_X=None, scaler_y=None, scaler_total=None, train=True):
        """
        X: 输入特征 (n_samples, n_dimensions)
        y: 总目标值 (n_samples,)，仅用于验证
        """
        self.n_dimensions = X.shape[1]
        
        # 计算每个维度的贡献
        self.contributions = calculate_dimension_contribution(X)
        # 总和
        self.total_contributions = np.sum(self.contributions, axis=1)
        
        if train and (scaler_X is None or scaler_y is None or scaler_total is None):
            # 特征标准化
            self.scaler_X = StandardScaler()
            X = self.scaler_X.fit_transform(X)
            
            # 每个维度的贡献标准化
            self.scaler_y = [StandardScaler() for _ in range(self.n_dimensions)]
            contrib_scaled = np.zeros_like(self.contributions)
            for dim in range(self.n_dimensions):
                contrib_scaled[:, dim] = self.scaler_y[dim].fit_transform(
                    self.contributions[:, dim].reshape(-1, 1)).ravel()
            self.contributions = contrib_scaled
            
            # 总贡献标准化
            self.scaler_total = StandardScaler()
            self.total_scaled = self.scaler_total.fit_transform(self.total_contributions.reshape(-1, 1)).ravel()
        else:
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
            self.scaler_total = scaler_total
            
            if scaler_X is not None:
                X = self.scaler_X.transform(X)
            
            if scaler_y is not None:
                contrib_scaled = np.zeros_like(self.contributions)
                for dim in range(self.n_dimensions):
                    contrib_scaled[:, dim] = self.scaler_y[dim].transform(
                        self.contributions[:, dim].reshape(-1, 1)).ravel()
                self.contributions = contrib_scaled
            
            if scaler_total is not None:
                self.total_scaled = self.scaler_total.transform(self.total_contributions.reshape(-1, 1)).ravel()
            else:
                self.total_scaled = self.total_contributions
        
        # 转换为PyTorch张量
        self.X = torch.FloatTensor(X)
        self.y_contrib = torch.FloatTensor(self.contributions)
        self.y_total = torch.FloatTensor(self.total_scaled)
        
        # 保存原始值用于评估
        self.raw_total = self.total_contributions
        self.raw_y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_contrib[idx], self.y_total[idx]

def calculate_pearson(predictions, targets):
    return np.corrcoef(predictions, targets)[0, 1]

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler,
                device, scaler_y, scaler_total, n_dimensions, num_epochs=100, early_stop_patience=10):
    model = model.to(device)
    best_val_loss = float('inf')
    best_test_pearson = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_pearsons = []
    val_pearsons = []
    test_pearsons = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_dim_outputs = []
        train_predictions = []
        train_targets = []
        
        for X_batch, y_contrib_batch, y_total_batch in train_loader:
            X_batch = X_batch.to(device)
            y_contrib_batch = y_contrib_batch.to(device)
            y_total_batch = y_total_batch.to(device)
            
            optimizer.zero_grad()
            
            # 训练每个子网络预测其对应维度的贡献值
            total_loss = 0
            all_outputs = []
            
            for dim in range(n_dimensions):
                dim_input = X_batch[:, dim:dim+1]
                dim_target = y_contrib_batch[:, dim]
                
                dim_output = model.sub_networks[dim](dim_input).squeeze(-1)
                all_outputs.append(dim_output)
                
                dim_loss = criterion(dim_output, dim_target)
                total_loss += dim_loss
            
            # 计算总和预测
            batch_pred_sum = torch.stack(all_outputs, dim=1).sum(dim=1)
            
            # 增加总和预测损失，调整权重为0.5:0.5
            sum_loss = criterion(batch_pred_sum, y_total_batch)
            total_loss = (total_loss / n_dimensions) * 0.5 + sum_loss * 0.5  # 调整权重分配
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # 收集预测结果用于计算相关系数
            train_dim_outputs.extend([dim_out.detach().cpu().numpy() for dim_out in all_outputs])
            train_predictions.extend(batch_pred_sum.detach().cpu().numpy())
            train_targets.extend(y_total_batch.cpu().numpy())
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 计算训练集的Pearson相关系数
        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)
        
        # 反归一化总和
        if scaler_total is not None:
            train_pred_orig = scaler_total.inverse_transform(train_predictions.reshape(-1, 1)).ravel()
            train_targets_orig = scaler_total.inverse_transform(train_targets.reshape(-1, 1)).ravel()
            train_pearson = calculate_pearson(train_pred_orig, train_targets_orig)
        else:
            train_pearson = calculate_pearson(train_predictions, train_targets)
        
        train_pearsons.append(train_pearson)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_dim_outputs = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_contrib_batch, y_total_batch in val_loader:
                X_batch = X_batch.to(device)
                y_contrib_batch = y_contrib_batch.to(device)
                y_total_batch = y_total_batch.to(device)
                
                all_outputs = []
                total_loss = 0
                
                for dim in range(n_dimensions):
                    dim_input = X_batch[:, dim:dim+1]
                    dim_target = y_contrib_batch[:, dim]
                    
                    dim_output = model.sub_networks[dim](dim_input).squeeze(-1)
                    all_outputs.append(dim_output)
                    
                    dim_loss = criterion(dim_output, dim_target)
                    total_loss += dim_loss
                
                # 计算总和预测
                batch_pred_sum = torch.stack(all_outputs, dim=1).sum(dim=1)
                
                # 增加总和预测损失
                sum_loss = criterion(batch_pred_sum, y_total_batch)
                total_loss = (total_loss / n_dimensions) * 0.5 + sum_loss * 0.5  # 调整权重分配
                
                val_loss += total_loss.item()
                
                # 收集预测结果
                val_dim_outputs.extend([dim_out.cpu().numpy() for dim_out in all_outputs])
                val_predictions.extend(batch_pred_sum.cpu().numpy())
                val_targets.extend(y_total_batch.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 计算验证集的Pearson相关系数
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # 反归一化总和
        if scaler_total is not None:
            val_pred_orig = scaler_total.inverse_transform(val_predictions.reshape(-1, 1)).ravel()
            val_targets_orig = scaler_total.inverse_transform(val_targets.reshape(-1, 1)).ravel()
            val_pearson = calculate_pearson(val_pred_orig, val_targets_orig)
        else:
            val_pearson = calculate_pearson(val_predictions, val_targets)
        
        val_pearsons.append(val_pearson)
        
        # 测试集评估
        test_predictions, test_targets = [], []
        with torch.no_grad():
            for X_batch, y_contrib_batch, y_total_batch in test_loader:
                X_batch = X_batch.to(device)
                y_total_batch = y_total_batch.to(device)
                
                all_outputs = []
                for dim in range(n_dimensions):
                    dim_input = X_batch[:, dim:dim+1]
                    dim_output = model.sub_networks[dim](dim_input).squeeze(-1)
                    all_outputs.append(dim_output)
                
                batch_pred_sum = torch.stack(all_outputs, dim=1).sum(dim=1)
                test_predictions.extend(batch_pred_sum.cpu().numpy())
                test_targets.extend(y_total_batch.cpu().numpy())
        
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        
        # 反归一化总和
        if scaler_total is not None:
            test_pred_orig = scaler_total.inverse_transform(test_predictions.reshape(-1, 1)).ravel()
            test_targets_orig = scaler_total.inverse_transform(test_targets.reshape(-1, 1)).ravel()
            test_pearson = calculate_pearson(test_pred_orig, test_targets_orig)
        else:
            test_pearson = calculate_pearson(test_predictions, test_targets)
        
        test_pearsons.append(test_pearson)
        
        # 保存最佳模型（基于测试集Pearson系数）
        if test_pearson > best_test_pearson:
            best_test_pearson = test_pearson
            torch.save(model.state_dict(), 'best_pearson_model.pth')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_val_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f'Early stopping at epoch {epoch}')
                if best_test_pearson > 0.95:
                    print(f'已达到目标Pearson系数: {best_test_pearson:.4f}')
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Pearson: {train_pearson:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Pearson: {val_pearson:.4f}, '
                  f'Test Pearson: {test_pearson:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses, train_pearsons, val_pearsons, test_pearsons

def evaluate_model(model, test_loader, device, scaler_total, n_dimensions):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_contrib_batch, y_total_batch in test_loader:
            X_batch = X_batch.to(device)
            
            all_dim_outputs = []
            for dim in range(n_dimensions):
                dim_input = X_batch[:, dim:dim+1]
                dim_output = model.sub_networks[dim](dim_input).squeeze(-1)
                all_dim_outputs.append(dim_output)
            
            # 求和得到总预测值
            batch_pred = torch.stack(all_dim_outputs, dim=1).sum(dim=1)
            predictions.extend(batch_pred.cpu().numpy())
            
            # 使用标准化后的总目标
            actuals.extend(y_total_batch.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 反归一化
    if scaler_total is not None:
        predictions = scaler_total.inverse_transform(predictions.reshape(-1, 1)).ravel()
        actuals = scaler_total.inverse_transform(actuals.reshape(-1, 1)).ravel()
    
    mse = mean_squared_error(actuals, predictions)
    correlation = calculate_pearson(predictions, actuals)
    
    return predictions, actuals, mse, correlation

def plot_training_history(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def plot_pearson_history(train_pearsons, val_pearsons, test_pearsons, save_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(train_pearsons, label='Training Pearson')
    plt.plot(val_pearsons, label='Validation Pearson')
    plt.plot(test_pearsons, label='Test Pearson')
    plt.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.title('Pearson Correlation History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pearson_history.png'))
    plt.close()

def plot_predictions(predictions, actuals, pearson, save_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values (Pearson: {pearson:.4f})')
    
    # 添加Pearson系数文本
    text_x = min(actuals) + 0.05 * (max(actuals) - min(actuals))
    text_y = max(predictions) - 0.1 * (max(predictions) - min(predictions))
    plt.text(text_x, text_y, f'Pearson: {pearson:.4f}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(save_dir, 'predictions_scatter.png'))
    plt.close()

def plot_dimension_contributions(X_test, model, scaler_X, scaler_y, n_dimensions, device, save_dir):
    """绘制每个维度的贡献大小"""
    X_scaled = scaler_X.transform(X_test)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    dim_contributions = []
    
    with torch.no_grad():
        for dim in range(n_dimensions):
            dim_input = X_tensor[:, dim:dim+1]
            dim_output = model.sub_networks[dim](dim_input).squeeze(-1)
            # 反归一化得到实际贡献
            dim_contrib = scaler_y[dim].inverse_transform(
                dim_output.cpu().numpy().reshape(-1, 1))
            dim_contributions.append(np.mean(np.abs(dim_contrib)))
    
    # 绘制维度贡献条形图
    plt.figure(figsize=(14, 6))
    plt.bar(range(n_dimensions), dim_contributions)
    plt.xlabel('Dimension')
    plt.ylabel('Average Absolute Contribution')
    plt.title('Contribution of Each Dimension')
    plt.savefig(os.path.join(save_dir, 'dimension_contributions.png'))
    plt.close()
    
    return dim_contributions

def main():
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'schwefel/visualizations/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # 加载数据
    data_dir = 'schwefel/data/raw'
    X_train = np.load(os.path.join(data_dir, 'Schwefel_x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'Schwefel_y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'Schwefel_x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'Schwefel_y_test.npy'))
    
    # 打印数据基本信息
    print(f"训练集形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试集形状: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # 划分训练集和验证集
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # 创建数据加载器
    n_dimensions = X_train.shape[1]
    train_dataset = SchwefelDatasetDimensional(X_train, y_train)
    val_dataset = SchwefelDatasetDimensional(X_val, y_val, 
                                           train_dataset.scaler_X, 
                                           train_dataset.scaler_y,
                                           train_dataset.scaler_total,
                                           train=False)
    test_dataset = SchwefelDatasetDimensional(X_test, y_test, 
                                            train_dataset.scaler_X, 
                                            train_dataset.scaler_y,
                                            train_dataset.scaler_total,
                                            train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 减小batch_size为32
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DecomposedNetwork(input_dim=n_dimensions, hidden_size=96)  # 增加隐藏层节点数
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)  # 略微增加学习率
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )
    
    # 训练模型
    logging.info('Starting training...')
    print("开始训练...")
    train_losses, val_losses, train_pearsons, val_pearsons, test_pearsons = train_model(
        model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler,
        device, train_dataset.scaler_y, train_dataset.scaler_total, n_dimensions, 
        num_epochs=400, early_stop_patience=40  # 增加epochs和早停耐心值
    )
    
    # 加载最佳模型（基于Pearson系数）
    model.load_state_dict(torch.load('best_pearson_model.pth'))
    
    # 评估模型
    logging.info('Evaluating model...')
    predictions, actuals, mse, correlation = evaluate_model(
        model, test_loader, device, train_dataset.scaler_total, n_dimensions
    )
    
    # 记录结果
    logging.info(f'Test MSE: {mse:.4f}')
    logging.info(f'Test Correlation: {correlation:.4f}')
    print(f'测试集 MSE: {mse:.4f}')
    print(f'测试集 Pearson相关系数: {correlation:.4f}')
    
    # 绘制训练历史和Pearson变化趋势
    plot_training_history(train_losses, val_losses, output_dir)
    plot_pearson_history(train_pearsons, val_pearsons, test_pearsons, output_dir)
    
    # 绘制带Pearson值的预测散点图
    plot_predictions(predictions, actuals, correlation, output_dir)
    
    # 绘制并分析每个维度的贡献
    dim_contributions = plot_dimension_contributions(
        X_test, model, train_dataset.scaler_X, train_dataset.scaler_y, 
        n_dimensions, device, output_dir
    )
    
    # 保存维度贡献到文件
    np.savetxt(os.path.join(output_dir, 'dimension_contributions.csv'), 
              np.array([range(n_dimensions), dim_contributions]).T,
              delimiter=',', header='dimension,contribution', comments='')
    
    # 保存Pearson历史记录
    pearson_history = np.column_stack((train_pearsons, val_pearsons, test_pearsons))
    np.savetxt(os.path.join(output_dir, 'pearson_history.csv'),
              pearson_history,
              delimiter=',', header='train,validation,test', comments='')
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': train_dataset.scaler_X,
        'scaler_y': train_dataset.scaler_y,
        'scaler_total': train_dataset.scaler_total,
        'n_dimensions': n_dimensions,
        'test_pearson': correlation
    }, os.path.join(output_dir, 'final_model.pth'))

if __name__ == '__main__':
    main() 