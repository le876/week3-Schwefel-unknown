import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """
    绘制预测值与真实值的散点图
    
    参数:
    - y_true: 真实值
    - y_pred: 预测值
    - title: 图表标题
    - save_path: 保存图表的路径，如果为None则不保存
    """
    # 计算Pearson相关系数
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # 添加对角线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f"{title} (Pearson: {pearson_corr:.4f})")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.grid(True, alpha=0.3)
    
    # 添加文本说明
    plt.text(0.05, 0.95, f"Pearson Correlation: {pearson_corr:.4f}", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    
    plt.close()

def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    """
    绘制残差图
    
    参数:
    - y_true: 真实值
    - y_pred: 预测值
    - title: 图表标题
    - save_path: 保存图表的路径，如果为None则不保存
    """
    # 计算残差
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title(title)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.grid(True, alpha=0.3)
    
    # 添加残差统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    plt.text(0.05, 0.95, f"Mean Residual: {mean_residual:.4f}\nStd Residual: {std_residual:.4f}", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    
    plt.close()

def plot_residual_histogram(y_true, y_pred, title="Residual Histogram", save_path=None, bins=30):
    """
    绘制残差直方图
    
    参数:
    - y_true: 真实值
    - y_pred: 预测值
    - title: 图表标题
    - save_path: 保存图表的路径，如果为None则不保存
    - bins: 直方图的箱数
    """
    # 计算残差
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=bins, alpha=0.7, edgecolor='black')
    
    # 添加垂直线表示均值
    plt.axvline(x=np.mean(residuals), color='r', linestyle='--', label=f'Mean: {np.mean(residuals):.4f}')
    
    plt.title(title)
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加残差统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    plt.text(0.05, 0.95, f"Mean: {mean_residual:.4f}\nStd Dev: {std_residual:.4f}", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    
    plt.close()

def plot_qq_plot(y_true, y_pred, title="Q-Q Plot of Residuals", save_path=None):
    """
    绘制残差的Q-Q图，用于检查残差的正态性
    
    参数:
    - y_true: 真实值
    - y_pred: 预测值
    - title: 图表标题
    - save_path: 保存图表的路径，如果为None则不保存
    """
    from scipy import stats
    
    # 计算残差
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # 创建Q-Q图
    stats.probplot(residuals, plot=plt)
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    
    plt.close()

def create_comprehensive_residual_analysis(y_true, y_pred, output_dir, prefix=""):
    """
    创建全面的残差分析图表
    
    参数:
    - y_true: 真实值
    - y_pred: 预测值
    - output_dir: 输出目录
    - prefix: 文件名前缀
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 预测值与真实值的散点图
    plot_predictions_vs_actual(
        y_true, y_pred, 
        title="Predictions vs Actual Values",
        save_path=os.path.join(output_dir, f"{prefix}predictions_vs_actual.png")
    )
    
    # 残差图
    plot_residuals(
        y_true, y_pred, 
        title="Residual Plot",
        save_path=os.path.join(output_dir, f"{prefix}residual_plot.png")
    )
    
    # 残差直方图
    plot_residual_histogram(
        y_true, y_pred, 
        title="Residual Histogram",
        save_path=os.path.join(output_dir, f"{prefix}residual_histogram.png")
    )
    
    # 残差Q-Q图
    plot_qq_plot(
        y_true, y_pred, 
        title="Q-Q Plot of Residuals",
        save_path=os.path.join(output_dir, f"{prefix}residual_qq_plot.png")
    )
    
    # 绘制残差与特征的关系图（如果有特征数据）
    # 这部分需要在调用函数时提供特征数据

def plot_feature_importance(feature_importance, feature_names=None, title="Feature Importance", save_path=None, top_n=None):
    """
    绘制特征重要性图
    
    参数:
    - feature_importance: 特征重要性数组
    - feature_names: 特征名称列表，如果为None则使用索引
    - title: 图表标题
    - save_path: 保存图表的路径，如果为None则不保存
    - top_n: 显示前n个重要特征，如果为None则显示所有特征
    """
    # 如果没有提供特征名称，则使用索引
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    
    # 创建特征重要性和名称的数据框
    import pandas as pd
    df = pd.DataFrame({'importance': feature_importance, 'feature': feature_names})
    
    # 按重要性排序
    df = df.sort_values('importance', ascending=False)
    
    # 如果指定了top_n，则只显示前n个特征
    if top_n is not None:
        df = df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(df['feature'], df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    # 测试残差图生成
    np.random.seed(42)
    y_true = np.random.normal(0, 1, 100)
    y_pred = y_true + np.random.normal(0, 0.5, 100)  # 添加一些噪声
    
    # 创建测试输出目录
    test_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_plots')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 生成各种残差图
    create_comprehensive_residual_analysis(y_true, y_pred, test_output_dir, prefix="test_") 