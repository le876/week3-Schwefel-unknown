import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib参数
plt.style.use('default')  # 使用默认样式
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用系统默认字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True

# 创建输出目录
def create_analysis_dir():
    """创建时间戳分析目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'schwefel/analysis_results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# 加载数据
def load_data(data_dir='schwefel/data/raw'):
    """加载数据集"""
    X_train = np.load(os.path.join(data_dir, 'Schwefel_x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'Schwefel_y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'Schwefel_x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'Schwefel_y_test.npy'))
    
    # 创建验证集
    val_ratio = 0.25
    val_size = int(len(X_train) * val_ratio)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"训练集形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"验证集形状: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"测试集形状: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 基本统计信息
def basic_stats(X, y, output_dir, dataset_name):
    """基本统计描述"""
    n_samples, n_features = X.shape
    
    # 创建特征DataFrame
    df = pd.DataFrame(X, columns=[f'X_{i}' for i in range(n_features)])
    df['target'] = y
    
    # 基本统计量
    stats_df = df.describe().T
    stats_df['skew'] = df.skew()
    stats_df['kurtosis'] = df.kurtosis()
    
    # 保存统计结果
    stats_df.to_csv(f'{output_dir}/{dataset_name}_basic_stats.csv')
    
    # 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=30)
    plt.title(f'{dataset_name} Target Distribution')
    plt.xlabel('Target Value')
    plt.savefig(f'{output_dir}/{dataset_name}_target_dist.png')
    plt.close()
    
    # 特征分布概览
    plt.figure(figsize=(15, 10))
    for i in range(min(16, n_features)):
        plt.subplot(4, 4, i+1)
        sns.histplot(X[:, i], kde=True, bins=20)
        plt.title(f'X_{i} Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_features_dist.png')
    plt.close()
    
    # 箱线图检查异常值
    plt.figure(figsize=(15, 8))
    plt.boxplot(X, vert=False)
    plt.yticks(range(1, n_features+1), [f'X_{i}' for i in range(n_features)])
    plt.title(f'{dataset_name} Features Boxplot')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_features_boxplot.png')
    plt.close()
    
    # 相关性分析
    corr_matrix = np.corrcoef(X.T)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                xticklabels=[f'X_{i}' for i in range(n_features)],
                yticklabels=[f'X_{i}' for i in range(n_features)])
    plt.title(f'{dataset_name} Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_correlation_matrix.png')
    plt.close()
    
    # 与目标变量的相关性
    target_corr = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_features), np.abs(target_corr))
    plt.xlabel('Feature Index')
    plt.ylabel('|Correlation with Target|')
    plt.title(f'{dataset_name} Feature-Target Correlation')
    plt.savefig(f'{output_dir}/{dataset_name}_feature_target_corr.png')
    plt.close()
    
    # PCA降维可视化
    if n_features > 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Target Value')
        plt.title(f'{dataset_name} PCA Visualization')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.savefig(f'{output_dir}/{dataset_name}_pca_visualization.png')
        plt.close()
        
        # 累积方差解释率
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
        plt.grid(True)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title(f'{dataset_name} PCA Cumulative Explained Variance')
        plt.savefig(f'{output_dir}/{dataset_name}_pca_cum_variance.png')
        plt.close()
    
    return stats_df

# 数据分布可视化
def visualize_distributions(df_x, df_y):
    # 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df_y['target'], kde=True)
    plt.title('Target Variable Distribution')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    
    # 特征分布
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_x.columns):
        plt.subplot(5, 4, i+1)
        sns.histplot(df_x[col], kde=True)
        plt.title(f'Feature {col} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    
    # 箱线图
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_x)
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
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 特征与目标变量的相关性
    feature_target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
    print("\n特征与目标变量的相关性:")
    print(feature_target_corr)
    
    # 绘制特征与目标变量的相关性条形图
    plt.figure(figsize=(12, 8))
    feature_target_corr.plot(kind='bar')
    plt.title('Feature Correlation with Target')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.savefig(os.path.join(output_dir, 'feature_target_correlation.png'))
    plt.close()
    
    # 保存相关性结果
    feature_target_corr.to_csv(os.path.join(output_dir, 'feature_target_correlation.csv'))
    
    return feature_target_corr

# 散点图分析
def scatter_plot_analysis(df_x, df_y, top_features=5):
    # 获取与目标变量相关性最高的特征
    corr_series = pd.Series(index=df_x.columns)
    for col in df_x.columns:
        corr_series[col] = pearsonr(df_x[col], df_y['target'])[0]
    
    top_corr_features = corr_series.abs().sort_values(ascending=False).index[:top_features]
    
    # 绘制散点图
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_corr_features):
        plt.subplot(2, 3, i+1)
        plt.scatter(df_x[feature], df_y['target'], alpha=0.5)
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
    
    # 绘制解释方差比
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='Cumulative')
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
def feature_importance_analysis(X_train, y_train, X_val, y_val, output_dir):
    """特征重要性分析"""
    n_features = X_train.shape[1]
    feature_names = [f'X_{i}' for i in range(n_features)]
    
    # 1. 统计检验 - F值
    f_values, p_values = f_regression(X_train, y_train)
    
    # 2. 互信息
    mi_values = mutual_info_regression(X_train, y_train)
    
    # 3. 基于树模型的特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importance = rf.feature_importances_
    
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_importance = gb.feature_importances_
    
    # 4. 基于线性模型的特征重要性
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train_scaled, y_train)
    lasso_importance = np.abs(lasso.coef_)
    
    # 整合所有特征重要性方法的结果
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'F_Value': f_values,
        'P_Value': p_values,
        'Mutual_Info': mi_values,
        'Random_Forest': rf_importance,
        'Gradient_Boosting': gb_importance,
        'Lasso_Coef': lasso_importance
    })
    
    # 保存特征重要性数据
    importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    # 可视化不同方法的特征重要性
    methods = ['F_Value', 'Mutual_Info', 'Random_Forest', 'Gradient_Boosting', 'Lasso_Coef']
    plt.figure(figsize=(15, 12))
    
    for i, method in enumerate(methods, 1):
        plt.subplot(len(methods), 1, i)
        
        # 按重要性排序
        sorted_idx = np.argsort(-importance_df[method])
        sorted_features = [feature_names[i] for i in sorted_idx]
        
        plt.barh(range(n_features), importance_df[method][sorted_idx])
        plt.yticks(range(n_features), [sorted_features[i] for i in range(n_features)])
        plt.title(f'Feature Importance - {method}')
        plt.tight_layout(pad=1.0)
    
    plt.savefig(f'{output_dir}/feature_importance_comparison.png')
    plt.close()
    
    # 特征重要性的相关性
    importance_corr = importance_df[methods].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(importance_corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Feature Importance Methods')
    plt.savefig(f'{output_dir}/importance_methods_correlation.png')
    plt.close()
    
    return importance_df

# 非线性关系分析
def nonlinearity_analysis(df_x, df_y, top_features=5):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # 获取与目标变量相关性最高的特征
    corr_series = pd.Series(index=df_x.columns)
    for col in df_x.columns:
        corr_series[col] = pearsonr(df_x[col], df_y['target'])[0]
    
    top_corr_features = corr_series.abs().sort_values(ascending=False).index[:top_features]
    
    results = []
    
    # 对每个顶级特征进行多项式拟合
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_corr_features):
        X = df_x[feature].values.reshape(-1, 1)
        y = df_y['target'].values
        
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

def analyze_schwefel_function(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    """分析Schwefel函数特性"""
    # 计算每个维度的贡献
    n_dimensions = X_train.shape[1]
    
    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
    
    results = {}
    
    for name, (X, y) in datasets.items():
        # 计算每个维度的贡献
        contributions = calculate_schwefel_contribution(X)
        # 计算总贡献
        calculated_total = np.sum(contributions, axis=1)
        
        # 计算与真实值的MSE和相关性
        mse = np.mean((calculated_total - y) ** 2)
        corr = np.corrcoef(calculated_total, y)[0, 1]
        
        # 计算每个维度的平均绝对贡献
        mean_abs_contrib = np.mean(np.abs(contributions), axis=0)
        mean_contrib = np.mean(contributions, axis=0)
        std_contrib = np.std(contributions, axis=0)
        
        # 排序得到重要维度
        importance_idx = np.argsort(-mean_abs_contrib)
        top_dims = importance_idx[:5]
        
        # 保存维度统计信息
        dim_stats = pd.DataFrame({
            'dimension': np.arange(n_dimensions),
            'mean_contribution': mean_contrib,
            'mean_abs_contribution': mean_abs_contrib,
            'std_contribution': std_contrib
        })
        dim_stats.to_csv(f'{output_dir}/{name}_contribution_stats.csv', index=False)
        
        # 可视化前5个重要维度的贡献分布
        plt.figure(figsize=(15, 10))
        for i, dim in enumerate(top_dims):
            plt.subplot(2, 3, i+1)
            plt.scatter(X[:, dim], contributions[:, dim], alpha=0.6, s=10)
            plt.title(f'Dimension {dim} Contribution')
            plt.xlabel(f'X_{dim}')
            plt.ylabel('Contribution')
        
        plt.subplot(2, 3, 6)
        plt.bar(range(n_dimensions), mean_abs_contrib)
        plt.title('Mean Absolute Contribution by Dimension')
        plt.xlabel('Dimension')
        plt.ylabel('Mean |Contribution|')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{name}_top_dimensions.png')
        plt.close()
        
        # 可视化真实值与计算值的对比
        plt.figure(figsize=(10, 8))
        plt.scatter(y, calculated_total, alpha=0.7, s=20)
        min_val = min(np.min(y), np.min(calculated_total))
        max_val = max(np.max(y), np.max(calculated_total))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f'{name.capitalize()} Set: True vs Calculated Values')
        plt.xlabel('True Values')
        plt.ylabel('Calculated Values')
        plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nCorr: {corr:.4f}', 
                 transform=plt.gca().transAxes, fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.savefig(f'{output_dir}/{name}_true_vs_calculated.png')
        plt.close()
        
        results[name] = {
            'mse': mse,
            'correlation': corr,
            'top_dimensions': top_dims.tolist(),
            'top_mean_abs_contribs': mean_abs_contrib[top_dims].tolist()
        }
    
    return results

def calculate_schwefel_contribution(X):
    """计算Schwefel函数每个维度的贡献"""
    return 418.9829 - X * np.sin(np.sqrt(np.abs(X)))

def analyze_nonlinearity(X_train, y_train, output_dir):
    """非线性和交互性分析"""
    n_samples, n_features = X_train.shape
    
    # 1. 使用二阶多项式特征的性能提升
    X_poly = np.zeros((n_samples, n_features*2))
    X_poly[:, :n_features] = X_train
    X_poly[:, n_features:] = X_train ** 2
    
    results = {}
    
    # 基础模型性能
    gb_base = GradientBoostingRegressor(n_estimators=50, random_state=42)
    base_score = np.mean(cross_val_score(gb_base, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    results['Base'] = -base_score
    
    # 多项式特征性能
    gb_poly = GradientBoostingRegressor(n_estimators=50, random_state=42)
    poly_score = np.mean(cross_val_score(gb_poly, X_poly, y_train, cv=5, scoring='neg_mean_squared_error'))
    results['Polynomial'] = -poly_score
    
    # 2. 特征交互分析 (分析一些重要特征对)
    # 使用RF选择Top 5重要特征
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    top_features = np.argsort(-rf.feature_importances_)[:5]
    
    # 可视化前几个重要特征对的交互
    interaction_pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            interaction_pairs.append((top_features[i], top_features[j]))
    
    n_pairs = len(interaction_pairs)
    fig_rows = (n_pairs + 1) // 2
    plt.figure(figsize=(15, 5*fig_rows))
    
    for i, (f1, f2) in enumerate(interaction_pairs):
        plt.subplot(fig_rows, 2, i+1)
        plt.scatter(X_train[:, f1], X_train[:, f2], c=y_train, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(label='Target Value')
        plt.xlabel(f'X_{f1}')
        plt.ylabel(f'X_{f2}')
        plt.title(f'Interaction: X_{f1} vs X_{f2}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_interactions.png')
    plt.close()
    
    # 3. 局部关系分析 - K近邻稠密区与目标值关系
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    avg_distances = np.mean(distances, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_distances, y_train, alpha=0.6)
    plt.xlabel('Average Distance to 20 Nearest Neighbors')
    plt.ylabel('Target Value')
    plt.title('Density vs Target Analysis')
    plt.savefig(f'{output_dir}/density_vs_target.png')
    plt.close()
    
    # 计算特征间非线性关系的强度 (基于距离相关系数)
    nonlinear_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i+1, n_features):
            # 使用斯皮尔曼相关系数作为非线性关系的粗略评估
            nonlinear_matrix[i, j] = abs(stats.spearmanr(X_train[:, i], X_train[:, j])[0])
            nonlinear_matrix[j, i] = nonlinear_matrix[i, j]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(nonlinear_matrix, annot=False, cmap='YlGnBu',
                xticklabels=[f'X_{i}' for i in range(n_features)],
                yticklabels=[f'X_{i}' for i in range(n_features)])
    plt.title('Feature Nonlinear Relationship Strength')
    plt.savefig(f'{output_dir}/nonlinear_relationship_matrix.png')
    plt.close()
    
    return results, nonlinear_matrix

def generate_summary_report(schwefel_results, output_dir):
    """生成分析总结报告"""
    with open(f'{output_dir}/analysis_summary.md', 'w') as f:
        f.write('# Schwefel函数数据集分析总结\n\n')
        
        f.write('## 训练集分析\n\n')
        f.write(f"* MSE (均方误差): {schwefel_results['train']['mse']:.2f}\n")
        f.write(f"* 相关系数: {schwefel_results['train']['correlation']:.2f}\n")
        f.write('* 基于绝对平均贡献的前5个重要维度:\n')
        for i, dim in enumerate(schwefel_results['train']['top_dimensions']):
            contrib = schwefel_results['train']['top_mean_abs_contribs'][i]
            f.write(f"  - 维度 {dim}: {contrib:.2f}\n")
        f.write('\n')
        
        f.write('## 验证集分析\n\n')
        f.write(f"* MSE (均方误差): {schwefel_results['val']['mse']:.2f}\n")
        f.write(f"* 相关系数: {schwefel_results['val']['correlation']:.2f}\n")
        f.write('* 基于绝对平均贡献的前5个重要维度:\n')
        for i, dim in enumerate(schwefel_results['val']['top_dimensions']):
            contrib = schwefel_results['val']['top_mean_abs_contribs'][i]
            f.write(f"  - 维度 {dim}: {contrib:.2f}\n")
        f.write('\n')
        
        f.write('## 测试集分析\n\n')
        f.write(f"* MSE (均方误差): {schwefel_results['test']['mse']:.2f}\n")
        f.write(f"* 相关系数: {schwefel_results['test']['correlation']:.2f}\n")
        f.write('* 基于绝对平均贡献的前5个重要维度:\n')
        for i, dim in enumerate(schwefel_results['test']['top_dimensions']):
            contrib = schwefel_results['test']['top_mean_abs_contribs'][i]
            f.write(f"  - 维度 {dim}: {contrib:.2f}\n")
    
    print(f"分析总结报告已保存至: {output_dir}/analysis_summary.md")

# 主函数
def main():
    print("开始Schwefel数据集分析...")
    
    # 创建输出目录
    output_dir = create_analysis_dir()
    print(f"分析结果将保存在: {output_dir}")
    
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # 数据概览分析
    print("执行基本统计分析...")
    train_stats = basic_stats(X_train, y_train, output_dir, 'train')
    val_stats = basic_stats(X_val, y_val, output_dir, 'validation')
    test_stats = basic_stats(X_test, y_test, output_dir, 'test')
    
    # Schwefel函数分析
    print("分析Schwefel函数特性...")
    schwefel_results = analyze_schwefel_function(
        X_train, y_train, X_val, y_val, X_test, y_test, output_dir
    )
    
    # 特征重要性分析
    print("执行特征重要性分析...")
    importance_df = feature_importance_analysis(X_train, y_train, X_val, y_val, output_dir)
    
    # 非线性和交互性分析
    print("执行非线性和交互性分析...")
    nonlinear_results, nonlinear_matrix = analyze_nonlinearity(X_train, y_train, output_dir)
    
    # 生成总结报告
    generate_summary_report(schwefel_results, output_dir)
    
    print(f"数据分析完成！结果已保存至 {output_dir}")

if __name__ == '__main__':
    main() 