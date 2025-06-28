import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

def calculate_metrics(y_true, y_pred):
    """
    计算回归模型的各种评估指标
    
    参数:
    - y_true: 真实值
    - y_pred: 预测值
    
    返回:
    - metrics: 包含各种评估指标的字典
    """
    # 均方误差 (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # 均方根误差 (RMSE)
    rmse = np.sqrt(mse)
    
    # 平均绝对误差 (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 决定系数 (R²)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson相关系数
    pearson_corr, p_value = pearsonr(y_true, y_pred)
    
    # 平均绝对百分比误差 (MAPE)
    # 避免除以零
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # 相对绝对误差 (RAE)
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
    
    # 相对平方误差 (RSE)
    rse = np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'p_value': p_value,
        'mape': mape,
        'rae': rae,
        'rse': rse
    }
    
    return metrics

def print_metrics(metrics, dataset_name=""):
    """
    打印评估指标
    
    参数:
    - metrics: 包含各种评估指标的字典
    - dataset_name: 数据集名称
    """
    if dataset_name:
        print(f"--- {dataset_name} 评估指标 ---")
    else:
        print("--- 评估指标 ---")
    
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    print(f"Pearson相关系数: {metrics['pearson_corr']:.6f} (p值: {metrics['p_value']:.6f})")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"RAE: {metrics['rae']:.6f}")
    print(f"RSE: {metrics['rse']:.6f}")
    print()

def save_metrics_to_file(metrics, file_path, dataset_name=""):
    """
    将评估指标保存到文件
    
    参数:
    - metrics: 包含各种评估指标的字典
    - file_path: 保存文件的路径
    - dataset_name: 数据集名称
    """
    with open(file_path, 'w') as f:
        if dataset_name:
            f.write(f"--- {dataset_name} 评估指标 ---\n")
        else:
            f.write("--- 评估指标 ---\n")
        
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"R²: {metrics['r2']:.6f}\n")
        f.write(f"Pearson相关系数: {metrics['pearson_corr']:.6f} (p值: {metrics['p_value']:.6f})\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"RAE: {metrics['rae']:.6f}\n")
        f.write(f"RSE: {metrics['rse']:.6f}\n")

def compare_metrics(metrics_list, model_names, file_path=None):
    """
    比较多个模型的评估指标
    
    参数:
    - metrics_list: 包含多个模型评估指标的列表
    - model_names: 模型名称列表
    - file_path: 保存比较结果的文件路径，如果为None则不保存
    """
    # 打印比较表格
    print("--- 模型评估指标比较 ---")
    print(f"{'模型':<15} {'MSE':<10} {'RMSE':<10} {'R²':<10} {'Pearson':<10}")
    print("-" * 55)
    
    for metrics, name in zip(metrics_list, model_names):
        print(f"{name:<15} {metrics['mse']:<10.6f} {metrics['rmse']:<10.6f} {metrics['r2']:<10.6f} {metrics['pearson_corr']:<10.6f}")
    
    # 保存比较结果
    if file_path:
        with open(file_path, 'w') as f:
            f.write("--- 模型评估指标比较 ---\n")
            f.write(f"{'模型':<15} {'MSE':<10} {'RMSE':<10} {'R²':<10} {'Pearson':<10}\n")
            f.write("-" * 55 + "\n")
            
            for metrics, name in zip(metrics_list, model_names):
                f.write(f"{name:<15} {metrics['mse']:<10.6f} {metrics['rmse']:<10.6f} {metrics['r2']:<10.6f} {metrics['pearson_corr']:<10.6f}\n")

if __name__ == "__main__":
    # 测试评估指标计算
    y_true = np.array([3, -0.5, 2, 7, 4.2])
    y_pred = np.array([2.5, 0.0, 2, 8, 4.5])
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "测试数据集") 