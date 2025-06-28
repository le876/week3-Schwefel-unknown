import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用系统默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建输出目录
output_dir = 'data_analysis_results'
os.makedirs(output_dir, exist_ok=True)

# 加载数据
def load_data():
    x_train = np.load('data/raw/x_48_train(1).npy')
    y_train = np.load('data/raw/y_48_train(1).npy')
    x_test = np.load('data/raw/x_48_test(1).npy')
    y_test = np.load('data/raw/y_48_test(1).npy')
    
    return x_train, y_train, x_test, y_test

# 预处理三维数据
def preprocess_3d_data(x_data):
    """将三维数据(samples, seq_length, features)转换为二维数据以便分析"""
    # 获取数据维度
    n_samples, seq_length, n_features = x_data.shape
    
    # 方法1: 使用序列的平均值
    x_mean = np.mean(x_data, axis=1)  # 对序列长度维度取平均
    
    # 方法2: 使用序列的最大值
    x_max = np.max(x_data, axis=1)
    
    # 方法3: 使用序列的最小值
    x_min = np.min(x_data, axis=1)
    
    # 方法4: 使用序列的标准差
    x_std = np.std(x_data, axis=1)
    
    # 合并特征
    x_combined = np.concatenate([
        x_mean, x_max, x_min, x_std
    ], axis=1)
    
    # 创建特征名称
    feature_names = []
    for stat in ['mean', 'max', 'min', 'std']:
        for i in range(n_features):
            feature_names.append(f'X{i+1}_{stat}')
    
    return x_combined, feature_names

# 基本统计信息
def basic_stats(x_train, y_train, feature_names):
    print("特征数据基本统计信息:")
    df_x = pd.DataFrame(x_train, columns=feature_names)
    stats = df_x.describe()
    print(stats)
    
    print("\n目标变量基本统计信息:")
    df_y = pd.DataFrame(y_train, columns=['Y'])
    print(df_y.describe())
    
    # 保存统计信息到文件
    stats.to_csv(os.path.join(output_dir, 'feature_stats.csv'))
    df_y.describe().to_csv(os.path.join(output_dir, 'target_stats.csv'))
    
    return df_x, df_y

# 数据分布可视化
def visualize_distributions(df_x, df_y):
    # 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df_y['Y'], kde=True)
    plt.title('Target Variable Distribution')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    
    # 特征分布 - 由于特征数量可能很多，只绘制前20个
    n_features = min(20, df_x.shape[1])
    n_rows = (n_features + 3) // 4  # 每行4个子图
    
    plt.figure(figsize=(16, n_rows * 3))
    for i in range(n_features):
        plt.subplot(n_rows, 4, i+1)
        sns.histplot(df_x.iloc[:, i], kde=True)
        plt.title(f'Feature {df_x.columns[i]} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    
    # 箱线图 - 由于特征数量可能很多，只绘制前20个
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df_x.iloc[:, :n_features])
    plt.title('Feature Boxplots')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_dir, 'feature_boxplots.png'))
    plt.close()

# 相关性分析
def correlation_analysis(df_x, df_y):
    # 合并特征和目标变量
    df = pd.concat([df_x, df_y], axis=1)
    
    # 计算相关系数矩阵
    corr_matrix = df.corr()
    
    # 特征与目标变量的相关性
    feature_target_corr = corr_matrix['Y'].drop('Y').sort_values(ascending=False)
    print("\n特征与目标变量的相关性:")
    print(feature_target_corr.head(20))  # 只显示前20个
    
    # 绘制特征与目标变量的相关性条形图 - 只显示前20个最相关的特征
    plt.figure(figsize=(12, 8))
    feature_target_corr.head(20).plot(kind='bar')
    plt.title('Top 20 Feature Correlation with Target')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.savefig(os.path.join(output_dir, 'feature_target_correlation.png'))
    plt.close()
    
    # 保存相关性结果
    feature_target_corr.to_csv(os.path.join(output_dir, 'feature_target_correlation.csv'))
    
    # 绘制热力图 - 只显示与目标变量相关性最高的20个特征
    top_features = feature_target_corr.abs().sort_values(ascending=False).head(20).index
    plt.figure(figsize=(14, 12))
    sns.heatmap(df[list(top_features) + ['Y']].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix (Top 20 Features)')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    return feature_target_corr

# 散点图分析
def scatter_plot_analysis(df_x, df_y, top_features=5):
    # 获取与目标变量相关性最高的特征
    corr_series = pd.Series(index=df_x.columns)
    for col in df_x.columns:
        corr_series[col] = pearsonr(df_x[col], df_y['Y'])[0]
    
    top_corr_features = corr_series.abs().sort_values(ascending=False).index[:top_features]
    
    # 绘制散点图
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_corr_features):
        plt.subplot(2, 3, i+1)
        plt.scatter(df_x[feature], df_y['Y'], alpha=0.5)
        plt.title(f'{feature} vs Target (corr: {corr_series[feature]:.3f})')
        plt.xlabel(feature)
        plt.ylabel('Target')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_scatter.png'))
    plt.close()

# 主成分分析
def pca_analysis(x_train):
    # 标准化数据
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    
    # 执行PCA
    pca = PCA()
    pca_result = pca.fit_transform(x_scaled)
    
    # 计算解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 绘制解释方差比 - 只显示前50个主成分或全部（如果少于50个）
    n_components = min(50, len(explained_variance_ratio))
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, n_components + 1), explained_variance_ratio[:n_components], alpha=0.7, label='Individual')
    plt.step(range(1, n_components + 1), cumulative_variance_ratio[:n_components], where='mid', label='Cumulative')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    plt.close()
    
    # 计算需要的主成分数量以解释95%的方差
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"\n需要{n_components_95}个主成分来解释95%的方差")
    
    # 绘制前两个主成分的散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First Two Principal Components')
    plt.savefig(os.path.join(output_dir, 'pca_first_two_components.png'))
    plt.close()
    
    return pca, explained_variance_ratio, n_components_95

# 特征重要性分析（使用随机森林）
def feature_importance_analysis(x_train, y_train, feature_names):
    from sklearn.ensemble import RandomForestRegressor
    
    # 训练随机森林模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    
    # 创建特征重要性数据框
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n随机森林特征重要性 (前20个):")
    print(feature_importance_df.head(20))
    
    # 绘制特征重要性条形图 - 只显示前20个最重要的特征
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Feature Importance (Random Forest) - Top 20')
    plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
    plt.close()
    
    # 保存特征重要性结果
    feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance_rf.csv'), index=False)
    
    return feature_importance_df

# 非线性关系分析
def nonlinearity_analysis(df_x, df_y, top_features=5):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # 获取与目标变量相关性最高的特征
    corr_series = pd.Series(index=df_x.columns)
    for col in df_x.columns:
        corr_series[col] = pearsonr(df_x[col], df_y['Y'])[0]
    
    top_corr_features = corr_series.abs().sort_values(ascending=False).index[:top_features]
    
    results = []
    
    # 对每个顶级特征进行多项式拟合
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_corr_features):
        X = df_x[feature].values.reshape(-1, 1)
        y = df_y['Y'].values
        
        # 线性拟合
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_linear = lr.predict(X)
        r2_linear = r2_score(y, y_pred_linear)
        
        # 二次多项式拟合
        poly2 = PolynomialFeatures(degree=2)
        X_poly2 = poly2.fit_transform(X)
        lr2 = LinearRegression()
        lr2.fit(X_poly2, y)
        y_pred_poly2 = lr2.predict(X_poly2)
        r2_poly2 = r2_score(y, y_pred_poly2)
        
        # 三次多项式拟合
        poly3 = PolynomialFeatures(degree=3)
        X_poly3 = poly3.fit_transform(X)
        lr3 = LinearRegression()
        lr3.fit(X_poly3, y)
        y_pred_poly3 = lr3.predict(X_poly3)
        r2_poly3 = r2_score(y, y_pred_poly3)
        
        results.append({
            'Feature': feature,
            'R2_Linear': r2_linear,
            'R2_Poly2': r2_poly2,
            'R2_Poly3': r2_poly3
        })
        
        # 绘制拟合结果
        plt.subplot(2, 3, i+1)
        
        # 排序以便绘图
        sort_idx = np.argsort(X.flatten())
        X_sorted = X.flatten()[sort_idx]
        y_sorted = y[sort_idx]
        y_pred_linear_sorted = y_pred_linear[sort_idx]
        y_pred_poly2_sorted = y_pred_poly2[sort_idx]
        y_pred_poly3_sorted = y_pred_poly3[sort_idx]
        
        plt.scatter(X_sorted, y_sorted, alpha=0.5, label='Data')
        plt.plot(X_sorted, y_pred_linear_sorted, 'r-', label=f'Linear (R²={r2_linear:.3f})')
        plt.plot(X_sorted, y_pred_poly2_sorted, 'g-', label=f'Poly2 (R²={r2_poly2:.3f})')
        plt.plot(X_sorted, y_pred_poly3_sorted, 'b-', label=f'Poly3 (R²={r2_poly3:.3f})')
        plt.title(f'{feature} vs Target')
        plt.xlabel(feature)
        plt.ylabel('Target')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nonlinearity_analysis.png'))
    plt.close()
    
    # 创建结果数据框
    nonlinearity_df = pd.DataFrame(results)
    print("\n非线性关系分析:")
    print(nonlinearity_df)
    
    # 保存非线性分析结果
    nonlinearity_df.to_csv(os.path.join(output_dir, 'nonlinearity_analysis.csv'), index=False)
    
    return nonlinearity_df

# 时序数据可视化
def visualize_time_series(x_train, y_train):
    """可视化时序数据的特征"""
    # 选择几个样本进行可视化
    sample_indices = [0, 1, 2, 3, 4]  # 前5个样本
    n_samples = len(sample_indices)
    n_features = x_train.shape[2]
    
    # 为每个特征创建一个图
    for feature_idx in range(n_features):
        plt.figure(figsize=(15, 10))
        for i, sample_idx in enumerate(sample_indices):
            plt.subplot(n_samples, 1, i+1)
            plt.plot(x_train[sample_idx, :, feature_idx])
            plt.title(f'Sample {sample_idx}, Feature {feature_idx+1}, Target: {y_train[sample_idx]:.2f}')
            plt.xlabel('Time Step')
            plt.ylabel(f'Feature {feature_idx+1} Value')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'time_series_feature_{feature_idx+1}.png'))
        plt.close()
    
    # 创建一个样本的所有特征的图
    sample_idx = 0  # 第一个样本
    plt.figure(figsize=(15, 10))
    for feature_idx in range(n_features):
        plt.subplot(n_features, 1, feature_idx+1)
        plt.plot(x_train[sample_idx, :, feature_idx])
        plt.title(f'Sample {sample_idx}, Feature {feature_idx+1}')
        plt.xlabel('Time Step')
        plt.ylabel(f'Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series_all_features_sample_0.png'))
    plt.close()

# 主函数
def main():
    print("开始Unknown48数据集分析...")
    
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()
    print(f"训练集形状: {x_train.shape}, {y_train.shape}")
    print(f"测试集形状: {x_test.shape}, {y_test.shape}")
    
    # 可视化时序数据
    print("\n生成时序数据可视化...")
    visualize_time_series(x_train, y_train)
    
    # 预处理三维数据为二维
    print("\n预处理三维数据为二维...")
    x_train_2d, feature_names = preprocess_3d_data(x_train)
    print(f"预处理后的训练集形状: {x_train_2d.shape}")
    
    # 转换为DataFrame
    df_x = pd.DataFrame(x_train_2d, columns=feature_names)
    df_y = pd.DataFrame(y_train, columns=['Y'])
    
    # 基本统计信息
    print("\n计算基本统计信息...")
    df_x, df_y = basic_stats(x_train_2d, y_train, feature_names)
    
    # 数据分布可视化
    print("\n生成数据分布可视化...")
    visualize_distributions(df_x, df_y)
    
    # 相关性分析
    print("\n进行相关性分析...")
    feature_target_corr = correlation_analysis(df_x, df_y)
    
    # 散点图分析
    print("\n生成散点图分析...")
    scatter_plot_analysis(df_x, df_y)
    
    # 主成分分析
    print("\n进行主成分分析...")
    pca, explained_variance_ratio, n_components_95 = pca_analysis(x_train_2d)
    
    # 特征重要性分析
    print("\n进行特征重要性分析...")
    feature_importance_df = feature_importance_analysis(x_train_2d, y_train, feature_names)
    
    # 非线性关系分析
    print("\n进行非线性关系分析...")
    nonlinearity_df = nonlinearity_analysis(df_x, df_y)
    
    print("\n数据分析完成！结果保存在", output_dir, "目录中")

if __name__ == "__main__":
    main() 