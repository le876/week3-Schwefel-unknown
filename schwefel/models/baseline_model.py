import numpy as np
import os
import sys
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.load_data import get_data

def train_model(model_type='rf', add_squared_features=False, **model_params):
    """
    训练指定类型的模型
    
    参数:
    - model_type: 模型类型，可选 'rf'(随机森林), 'gbr'(梯度提升), 'linear'(线性回归), 
                 'ridge'(岭回归), 'lasso'(Lasso回归), 'svr'(支持向量回归), 'mlp'(神经网络)
    - add_squared_features: 是否添加平方特征
    - model_params: 模型参数
    
    返回:
    - model: 训练好的模型
    - metrics: 包含各种评估指标的字典
    """
    # 加载数据
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = get_data(add_squared_features=add_squared_features)
    
    # 根据指定类型创建模型
    if model_type == 'rf':
        model = RandomForestRegressor(**model_params)
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(**model_params)
    elif model_type == 'linear':
        model = LinearRegression(**model_params)
    elif model_type == 'ridge':
        model = Ridge(**model_params)
    elif model_type == 'lasso':
        model = Lasso(**model_params)
    elif model_type == 'svr':
        model = SVR(**model_params)
    elif model_type == 'mlp':
        model = MLPRegressor(**model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    model.fit(x_train, y_train)
    
    # 在训练集、验证集和测试集上进行预测
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)
    
    # 计算评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_pearson = pearsonr(y_train, y_train_pred)[0]
    val_pearson = pearsonr(y_val, y_val_pred)[0]
    test_pearson = pearsonr(y_test, y_test_pred)[0]
    
    metrics = {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'train_pearson': train_pearson,
        'val_pearson': val_pearson,
        'test_pearson': test_pearson
    }
    
    # 保存模型和评估指标
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tuning_results', timestamp)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    joblib.dump(model, os.path.join(model_dir, f"{model_type}_model.pkl"))
    
    # 保存评估指标
    with open(os.path.join(model_dir, 'metrics.txt'), 'w') as f:
        f.write(f"模型类型: {model_type}\n")
        f.write(f"是否使用平方特征: {add_squared_features}\n")
        f.write(f"模型参数: {model_params}\n\n")
        f.write(f"训练集MSE: {train_mse:.6f}\n")
        f.write(f"验证集MSE: {val_mse:.6f}\n")
        f.write(f"测试集MSE: {test_mse:.6f}\n\n")
        f.write(f"训练集R²: {train_r2:.6f}\n")
        f.write(f"验证集R²: {val_r2:.6f}\n")
        f.write(f"测试集R²: {test_r2:.6f}\n\n")
        f.write(f"训练集Pearson相关系数: {train_pearson:.6f}\n")
        f.write(f"验证集Pearson相关系数: {val_pearson:.6f}\n")
        f.write(f"测试集Pearson相关系数: {test_pearson:.6f}\n")
    
    # 绘制预测值与真实值的散点图
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.title(f'Training Set (Pearson: {train_pearson:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title(f'Validation Set (Pearson: {val_pearson:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Test Set (Pearson: {test_pearson:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'prediction_scatter.png'))
    
    # 绘制残差图
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_train_pred, y_train - y_train_pred, alpha=0.5)
    plt.hlines(y=0, xmin=y_train_pred.min(), xmax=y_train_pred.max(), colors='r', linestyles='--')
    plt.title('Training Set Residuals')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_val_pred, y_val - y_val_pred, alpha=0.5)
    plt.hlines(y=0, xmin=y_val_pred.min(), xmax=y_val_pred.max(), colors='r', linestyles='--')
    plt.title('Validation Set Residuals')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5)
    plt.hlines(y=0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), colors='r', linestyles='--')
    plt.title('Test Set Residuals')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'residual_plot.png'))
    
    return model, metrics

if __name__ == "__main__":
    # 测试基线模型
    model, metrics = train_model(
        model_type='rf',
        add_squared_features=True,
        n_estimators=100,
        random_state=42
    )
    
    print("模型训练完成")
    print(f"测试集Pearson相关系数: {metrics['test_pearson']:.6f}") 